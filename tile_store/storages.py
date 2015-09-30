#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tile storage implementation.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import contextlib
import hashlib
import logging
import multiprocessing
import os
import struct
import zipfile

logger = logging.getLogger(__name__)
lock = multiprocessing.Lock()


def prepare(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass


class AbstractStorage(object):
    def __init__(self, path, mode='r'):
        self.path = path
        self.mode = mode


class FileStorage(AbstractStorage):
    """ Store tiles in a simple tms tree. """
    def get_tile_path(self, key):
        x, y, z = key
        return os.path.join(self.path, str(z), str(x), str(y))

    def __setitem__(self, key, value):
        path = self.get_tile_path(key)
        prepare(path)
        open(path, 'w').write(value)

    def __getitem__(self, key):
        path = self.get_tile_path(key)
        try:
            return open(path).read()
        except IOError:
            raise KeyError()


class ZipFileStorage(AbstractStorage):
    """
    Store tiles in a balanced tree of zipfiles.
    """
    @contextlib.contextmanager
    def acquire(self):
        if self.mode == 'a':
            lock.acquire()
        try:
            yield
        finally:
            if self.mode == 'a':
                lock.release()

    def get_path_and_key(self, key):
        # unique zipfile paths per 16 x 16 images
        x, y, z = key
        p, q = x >> 4, y >> 4
        md5 = hashlib.md5(struct.pack('3q', p, q, z)).hexdigest()
        path = os.path.join(self.path, md5[0:2], md5[2:4], md5[4:] + '.zip')

        # unique entries in zipfiles
        return path, hashlib.md5(struct.pack('3q', x, y, z)).hexdigest()

    def __setitem__(self, key, value):
        path, zkey = self.get_path_and_key(key)
        prepare(path)
        with self.acquire():
            with zipfile.ZipFile(path, mode='a') as archive:
                zkeys = archive.namelist()
                if zkey not in zkeys:
                    archive.writestr(zkey, value)
                    return

            # move all other keys to new zipfile
            ext = '.in'
            with zipfile.ZipFile(path) as source:
                with zipfile.ZipFile(path + ext, mode='a') as target:
                    target.writestr(zkey, value)
                    zkeys.remove(zkey)
                    for zkey in zkeys:
                        target.writestr(zkey, source.read(zkey))
            os.rename(path + ext, path)

    def __getitem__(self, key):
        path, zkey = self.get_path_and_key(key)
        with self.acquire():
            try:
                with zipfile.ZipFile(path) as archive:
                    return archive.read(zkey)
            except IOError:
                raise KeyError()
