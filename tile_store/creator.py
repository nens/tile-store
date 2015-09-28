# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Create a tilemap from a GDAL datasource.

Resampling methods can be one of:
bilinear, cubic, average, nearestneighbour, mode, lanczos, cubicspline
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import io
import itertools
import logging
import math
import multiprocessing
import sys

from tile_store import datasets
from tile_store import gdal
from tile_store import osr

from tile_store import storages

from PIL import Image
from osgeo import gdal_array
import numpy as np

logger = logging.getLogger(__name__)

LIM = 2 * 6378137 * math.pi
WKT = osr.GetUserInputAsWKT(str('epsg:3857'))
GRA = {n[4:].lower(): getattr(gdal, n)
       for n in dir(gdal) if n.startswith('GRA')}

master = None


def initializer(path):
    """ Assign master dataset to a global variable. """
    global master
    master = gdal.Open(path)


def func(job):
    """ Make tile. """
    tile, count = job
    tile.make()
    return job


class BBox(object):
    def __init__(self, dataset):
        # analyze
        w = dataset.RasterXSize
        h = dataset.RasterYSize
        g = dataset.GetGeoTransform()
        coords = map(gdal.ApplyGeoTransform, 4 * [g], 2 * [0, w], [0, 0, h, h])

        # transform
        source = osr.SpatialReference(dataset.GetProjection())
        target = osr.SpatialReference(WKT)
        ct = osr.CoordinateTransformation(source, target)
        x, y = zip(*ct.TransformPoints(coords))[:2]

        self.x1, self.y1, self.x2, self.y2 = min(x), min(y), max(x), max(y)


class DummyPool(object):
    """ Dummy pool in case multiprocessing is not used. """
    imap = itertools.imap

    def __init__(self, initializer, initargs):
        initializer(*initargs)


class Tile(object):
    """
    Base tile class that allows loading from storage and conversion to
    a gdal dataset.
    """
    def __init__(self, x, y, z, storage):
        self.x = x
        self.y = y
        self.z = z
        self.storage = storage

    def __str__(self):
        return '<{n}: {z}/{x}/{y}>'.format(x=self.x,
                                           y=self.y,
                                           z=self.z,
                                           n=self.__class__.__name__)

    def get_geo_transform(self):
        """ Return GeoTransform """
        s = LIM / 2 ** self.z
        p = s * self.x - LIM / 2
        q = LIM / 2 - s * self.y
        a = s / 256
        d = -a
        return p, a, 0, q, 0, d

    def as_dataset(self):
        """ Return image as gdal dataset. """
        # load
        try:
            data = self.storage[self.x, self.y, self.z]
        except KeyError:
            return

        # convert to rgba array
        array = np.array(Image.open(io.BytesIO(data))).transpose(2, 0, 1)
        if len(array) == 3:
            # add alpha
            array = np.vstack([array, np.full_like(array[:1], 255)])

        # return as dataset
        dataset = gdal_array.OpenArray(array)
        dataset.SetProjection(WKT)
        dataset.SetGeoTransform(self.get_geo_transform())
        return dataset


class TargetTile(Tile):
    """
    A tile that can build from sources and save to storage.
    """
    def __init__(self, quality, method,  **kwargs):
        super(TargetTile, self).__init__(**kwargs)
        self.quality = quality
        self.method = method

    def make(self):
        """ Make tile and store data on data attribute. """
        # target
        array = np.zeros((4, 256, 256), dtype='u1')
        kwargs = {'projection': WKT,
                  'geo_transform': self.get_geo_transform()}

        with datasets.Dataset(array, **kwargs) as target:
            gra = GRA[self.method]
            for source in self.get_sources():
                gdal.ReprojectImage(source, target, None, None, gra, 0, 0.125)

        # nothing
        if (array[-1] == 0).all():
            self.data = None
            return

        buf = io.BytesIO()
        if (array[-1] < 255).any():
            image = Image.fromarray(array.transpose(1, 2, 0))
            image.save(buf, format='PNG')
        else:
            image = Image.fromarray(array[:3].transpose(1, 2, 0))
            image.save(buf, format='JPEG', quality=self.quality)

        # save
        self.storage[self.x, self.y, self.z] = buf.getvalue()


class BaseTile(TargetTile):
    """
    A tile that has itself and a master as sources.
    """
    def __init__(self, **kwargs):
        """ Same as target tile, but preload. """
        super(BaseTile, self).__init__(**kwargs)

    def get_sources(self):
        """ Yield both this tile and the master, for stitching. """
        yield master
        dataset = self.as_dataset()
        if dataset is not None:
            yield dataset


class OverviewTile(TargetTile):
    """
    A tile that has its subtiles as sources.
    """
    def __init__(self, **kwargs):
        """ Same as target tile, but store preloaded subtiles. """
        super(OverviewTile, self).__init__(**kwargs)

    def get_subtiles(self):
        z = 1 + self.z
        for dy, dx in itertools.product([0, 1], [0, 1]):
            x = dx + 2 * self.x
            y = dy + 2 * self.y
            yield Tile(x=x, y=y, z=z, storage=self.storage)

    def get_sources(self):
        for subtile in self.get_subtiles():
            dataset = subtile.as_dataset()
            if dataset is not None:
                yield dataset


class Level(object):
    """ A single zoomlevel of tiles. """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        x, y = self.get_xranges()
        return len(x) * len(y)

    def __iter__(self):
        """ Return tile generator. """
        kwargs = self.kwargs.copy()
        kwargs.pop('bbox')

        z = kwargs.pop('zoom')
        t = kwargs.pop('tile')
        for y, x in itertools.product(*self.get_xranges()):
            yield t(x=x, y=y, z=z, **kwargs)

    def get_xranges(self):
        s = LIM / 2 ** self.kwargs['zoom']  # edge length
        h = LIM / 2                         # half the earth

        bbox = self.kwargs['bbox']

        x1 = int(math.floor((h + bbox.x1) / s))
        y1 = int(math.floor((h - bbox.y2) / s))
        x2 = int(math.ceil((h + bbox.x2) / s))
        y2 = int(math.ceil((h - bbox.y1) / s))

        return xrange(y1, y2), xrange(x1, x2)


class Pyramid(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return sum(len(l) for l in self)

    def __iter__(self):
        """ Return level generator. """
        kwargs = self.kwargs.copy()
        gra1 = kwargs.pop('gra1')
        gra2 = kwargs.pop('gra2')

        # yield baselevel
        zoom = kwargs.pop('zoom')
        yield Level(tile=BaseTile, zoom=zoom, method=gra1, **kwargs)

        # yield other levels
        for zoom in reversed(range(zoom)):
            yield Level(tile=OverviewTile, zoom=zoom, method=gra2, **kwargs)


def tiles(source_path, target_path, classic, single, verbose, **kwargs):
    """ Create tiles. """
    # inspect dataset
    dataset = gdal.Open(source_path)
    bbox = BBox(dataset)

    # select storage
    Storage = storages.FileStorage if classic else storages.ZipFileStorage
    storage = Storage(path=target_path, mode='a')

    pyramid = Pyramid(storage=storage, bbox=bbox, **kwargs)

    # separate counts for baselevel and the remaining levels
    total1 = len(iter(pyramid).next())
    total2 = len(pyramid)

    # create worker pool
    Pool = DummyPool if single else multiprocessing.Pool
    pool = Pool(initializer=initializer, initargs=[source_path])

    count = itertools.count(1)
    for level_count, level in enumerate(pyramid):

        # progress information
        if level_count == 0:
            logger.info('Generating Base Tiles:')
        elif level_count == 1:
            logger.info('Generating Overview Tiles:')

        for tile, tile_count in pool.imap(func, itertools.izip(level, count)):

            # progress bar
            if verbose:
                logger.debug(tile)
            elif tile_count > total1:
                progress = (tile_count - total1) / (total2 - total1)
                gdal.TermProgress_nocb(progress)
            else:
                progress = tile_count / total1
                gdal.TermProgress_nocb(progress)


def get_parser():
    """ Return argument parser. """
    FormatterClass1 = argparse.ArgumentDefaultsHelpFormatter
    FormatterClass2 = argparse.RawDescriptionHelpFormatter

    class FormatterClass(FormatterClass1, FormatterClass2):
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=FormatterClass)

    # positional
    parser.add_argument('source_path', metavar='SOURCE')
    parser.add_argument('target_path', metavar='TARGET')
    parser.add_argument('zoom', metavar='ZOOM', type=int)

    # optional
    parser.add_argument('-b', '--base', dest='gra1', default='cubic',
                        help='GDAL resampling for base tiles.')
    parser.add_argument('-c', '--classic', action='store_true',
                        help='Use classic storage instead of ZipFileStorage')
    parser.add_argument('-o', '--overview', dest='gra2', default='cubic',
                        help='GDAL resampling for overview tiles.')
    parser.add_argument('-q', '--quality', default=95,
                        type=int, help='JPEG quality for non-edge tiles')
    parser.add_argument('-s', '--single', action='store_true',
                        help='disable multiprocessing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print debug-level log messages')
    return parser


def main():
    """ Call tiles with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs['verbose']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        tiles(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1


if __name__ == '__main__':
    exit(main())
