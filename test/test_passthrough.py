#!/usr/bin/env python3
'''
test_passthrough.py - Unit tests for FUSE_PASSTHROUGH support.

Exercises the new ``pyfuse3.passthrough_open()`` /
``pyfuse3.passthrough_close()`` API and the ``FileInfo.backing_id``
field. When the running kernel (>= 6.9) and libfuse (>= 3.17) support
FUSE_PASSTHROUGH, reads issued against a passthrough-mounted file are
served directly by the kernel from the backing fd and the user-space
``Operations.read`` handler is never invoked.

The test skips with an informative message when the
kernel/libfuse/filesystem combination doesn't support passthrough — in
that case ``passthrough_open()`` returns 0 and the fallback path
exercises the normal Python read handler.

Copyright © 2024 Materia project. This work may be distributed under
the terms of the GNU LGPL.
'''

import sys

import pytest

if __name__ == '__main__':
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

import errno
import logging
import multiprocessing
import os
import stat
import threading

import trio

import pyfuse3
from pyfuse3 import (
    EntryAttributes,
    FileHandleT,
    FileInfo,
    FUSEError,
    InodeT,
    ReaddirToken,
    RequestContext,
)
from util import cleanup, fuse_test_marker, umount, wait_for_mount

pytestmark = fuse_test_marker()


BACKING_DATA = b'kernel passthrough test payload\n' * 32


def get_mp():
    if hasattr(multiprocessing, 'get_context'):
        mp = multiprocessing.get_context('fork')
    else:
        mp = multiprocessing  # type: ignore[assignment]
    if threading.active_count() != 1:
        raise RuntimeError('Multi-threaded test running is not supported')
    return mp


class PassthroughFs(pyfuse3.Operations):
    '''Minimal single-file FS that opts into FUSE_PASSTHROUGH.

    The backing path is shared from the parent process via the
    cross-process namespace and points at a regular file. The FS
    registers that file's fd as the kernel passthrough backing on
    every ``open()``. Subsequent reads go straight through the
    kernel — ``read()`` here only fires if passthrough was not
    negotiated, which lets the test distinguish the two code paths.
    '''

    enable_passthrough = True

    def __init__(self, cross_process):
        super().__init__()
        self.hello_name = b'message'
        self.hello_inode = pyfuse3.ROOT_INODE + 1
        self.status = cross_process
        self.lookup_cnt = 0
        self.status.read_called = False
        self.status.last_backing_id = 0
        self.status.last_open_errno = 0
        # Map fh -> (backing_id, backing_fd) for release().
        self._open_handles: dict[int, tuple[int, int]] = {}
        self._fh_counter = 0

    async def getattr(self, inode: InodeT, ctx: RequestContext | None = None) -> EntryAttributes:
        entry = EntryAttributes()
        if inode == pyfuse3.ROOT_INODE:
            entry.st_mode = stat.S_IFDIR | 0o755
            entry.st_size = 0
        elif inode == self.hello_inode:
            entry.st_mode = stat.S_IFREG | 0o644
            entry.st_size = len(BACKING_DATA)
        else:
            raise FUSEError(errno.ENOENT)

        entry.st_atime_ns = 0
        entry.st_ctime_ns = 0
        entry.st_mtime_ns = 0
        entry.st_gid = os.getgid()
        entry.st_uid = os.getuid()
        entry.st_ino = inode
        entry.entry_timeout = 0
        entry.attr_timeout = 0
        return entry

    async def forget(self, inode_list):
        for inode, cnt in inode_list:
            if inode == self.hello_inode:
                self.lookup_cnt -= 1
                assert self.lookup_cnt >= 0

    async def lookup(
        self, parent_inode: InodeT, name: bytes, ctx: RequestContext
    ) -> EntryAttributes:
        if parent_inode != pyfuse3.ROOT_INODE or name != self.hello_name:
            raise FUSEError(errno.ENOENT)
        self.lookup_cnt += 1
        return await self.getattr(self.hello_inode, ctx)

    async def opendir(self, inode, ctx):
        if inode != pyfuse3.ROOT_INODE:
            raise FUSEError(errno.ENOENT)
        return FileHandleT(inode)

    async def readdir(self, fh: FileHandleT, start_id: int, token: ReaddirToken) -> None:
        if start_id == 0:
            pyfuse3.readdir_reply(token, self.hello_name, await self.getattr(self.hello_inode), 1)
        return

    async def open(self, inode, flags, ctx):
        if inode != self.hello_inode:
            raise FUSEError(errno.ENOENT)
        if flags & (os.O_RDWR | os.O_WRONLY):
            raise FUSEError(errno.EACCES)

        backing_path = self.status.backing_path
        backing_fd = os.open(backing_path, os.O_RDONLY)
        backing_id = 0
        try:
            backing_id = pyfuse3.passthrough_open(backing_fd)
        except Exception:
            backing_id = 0
        self.status.last_backing_id = backing_id

        self._fh_counter += 1
        fh = self._fh_counter
        self._open_handles[fh] = (backing_id, backing_fd)

        fi = FileInfo(fh=FileHandleT(fh))
        if backing_id > 0:
            fi.backing_id = backing_id
        return fi

    async def read(self, fh, off, size):
        # If passthrough is in effect, the kernel serves reads directly
        # from the backing fd and this handler should NEVER fire. We
        # record the fact so the test can assert it.
        self.status.read_called = True
        return BACKING_DATA[off : off + size]

    async def release(self, fh):
        entry = self._open_handles.pop(int(fh), None)
        if entry is None:
            return
        backing_id, backing_fd = entry
        if backing_id > 0:
            try:
                pyfuse3.passthrough_close(backing_id)
            except Exception:
                pass
        try:
            os.close(backing_fd)
        except OSError:
            pass


@pytest.fixture()
def passthrough_mount(tmpdir):
    yield from _mount_passthrough(tmpdir)


def _mount_passthrough(tmpdir):
    mnt_dir = str(tmpdir.mkdir('mnt'))
    backing_path = str(tmpdir / 'backing.bin')
    with open(backing_path, 'wb') as fh:
        fh.write(BACKING_DATA)

    mp = get_mp()
    with mp.Manager() as mgr:
        cross_process = mgr.Namespace()
        cross_process.backing_path = backing_path
        mount_process = mp.Process(target=_run_fs, args=(mnt_dir, cross_process))
        mount_process.start()
        try:
            wait_for_mount(mount_process, mnt_dir)
            yield (mnt_dir, cross_process)
        except:
            cleanup(mount_process, mnt_dir)
            raise
        else:
            umount(mount_process, mnt_dir)


def _run_fs(mountpoint, cross_process):
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    testfs = PassthroughFs(cross_process)
    fuse_options = set(pyfuse3.default_options)
    fuse_options.add('fsname=pyfuse3_passthrough')
    pyfuse3.init(testfs, mountpoint, fuse_options)
    try:
        trio.run(pyfuse3.main)
    finally:
        pyfuse3.close()


def test_passthrough_open_outside_handler_raises():
    '''passthrough_open() must reject calls outside an open/create handler.'''
    with pytest.raises(RuntimeError, match='inside an open|inside a release'):
        pyfuse3.passthrough_open(0)


def test_passthrough_close_outside_handler_raises():
    '''passthrough_close() must reject calls outside a release handler.'''
    with pytest.raises(RuntimeError, match='inside an open|inside a release'):
        pyfuse3.passthrough_close(0)


def test_passthrough_bypasses_python_read(passthrough_mount):
    '''When passthrough is negotiated, kernel reads bypass the user-space read handler.

    Skips if kernel < 6.9 or libfuse < 3.17 (passthrough_open returns 0).
    '''
    mnt_dir, fs_state = passthrough_mount
    path = os.path.join(mnt_dir, 'message')

    with open(path, 'rb') as fh:
        data = fh.read()

    if fs_state.last_backing_id == 0:
        pytest.skip(
            'FUSE_PASSTHROUGH not negotiated on this system. Requires: '
            'Linux 6.9+ with CONFIG_FUSE_PASSTHROUGH=y, libfuse 3.17+, '
            'and the mounting process must hold CAP_SYS_ADMIN (the '
            'FUSE_DEV_IOC_BACKING_OPEN ioctl is privileged). Unprivileged '
            'user-space FUSE filesystems fall back to the normal '
            'Operations.read path automatically.'
        )

    assert data == BACKING_DATA, 'data read through mount must match backing file'
    assert not fs_state.read_called, (
        'Operations.read should not fire when passthrough is active '
        f'(backing_id={fs_state.last_backing_id})'
    )


def test_passthrough_fallback_when_disabled(tmpdir):
    '''When enable_passthrough=False, reads route through Operations.read as before.

    Confirms the new code is strictly opt-in and doesn't break existing
    filesystems that don't request passthrough.
    '''

    class NoPassthroughFs(PassthroughFs):
        enable_passthrough = False

    mnt_dir = str(tmpdir.mkdir('mnt'))
    backing_path = str(tmpdir / 'backing.bin')
    with open(backing_path, 'wb') as fh:
        fh.write(BACKING_DATA)

    mp = get_mp()
    with mp.Manager() as mgr:
        cross_process = mgr.Namespace()
        cross_process.backing_path = backing_path
        mount_process = mp.Process(
            target=_run_no_passthrough,
            args=(mnt_dir, cross_process),
        )
        mount_process.start()
        try:
            wait_for_mount(mount_process, mnt_dir)
            path = os.path.join(mnt_dir, 'message')
            with open(path, 'rb') as fh:
                data = fh.read()
            assert data == BACKING_DATA
            assert cross_process.read_called, (
                'Operations.read must fire when enable_passthrough is False'
            )
        except:
            cleanup(mount_process, mnt_dir)
            raise
        else:
            umount(mount_process, mnt_dir)


def _run_no_passthrough(mountpoint, cross_process):
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    class NoPassthroughFs(PassthroughFs):
        enable_passthrough = False

    testfs = NoPassthroughFs(cross_process)
    fuse_options = set(pyfuse3.default_options)
    fuse_options.add('fsname=pyfuse3_no_passthrough')
    pyfuse3.init(testfs, mountpoint, fuse_options)
    try:
        trio.run(pyfuse3.main)
    finally:
        pyfuse3.close()
