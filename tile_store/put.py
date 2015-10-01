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
import json
import logging
import math
import multiprocessing
import os
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
WGS84 = osr.GetUserInputAsWKT(str('epsg:4326'))
MERCATOR = osr.GetUserInputAsWKT(str('epsg:3857'))
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
        target = osr.SpatialReference(MERCATOR)
        ct = osr.CoordinateTransformation(source, target)
        x, y = zip(*ct.TransformPoints(coords))[:2]

        self.x1, self.y1, self.x2, self.y2 = min(x), min(y), max(x), max(y)

    def get_xranges(self, zoom):
        """ Return tile ranges for a zoom level. """
        s = LIM / 2 ** zoom  # edge length
        h = LIM / 2          # half the earth

        x1 = int(math.floor((h + self.x1) / s))
        y1 = int(math.floor((h - self.y2) / s))
        x2 = int(math.ceil((h + self.x2) / s))
        y2 = int(math.ceil((h - self.y1) / s))

        return xrange(y1, y2), xrange(x1, x2)

    def get_extent(self):
        """ Return wgs84 extent. """
        coords = (self.x1, self.y1), (self.x2, self.y2)
        source = osr.SpatialReference(MERCATOR)
        target = osr.SpatialReference(WGS84)
        ct = osr.CoordinateTransformation(source, target)
        return [c for p in ct.TransformPoints(coords) for c in p[:2]]


class Config(object):
    def __init__(self, path, zoom):
        self.path = os.path.join(path, 'tiles.json')

        # check for config
        try:
            self.config = json.load(open(self.path))
        except IOError:
            self.config = {'zoom': zoom}
            return

        # check for zoom
        if zoom != self.config['zoom']:
            raise ValueError(self.config['zoom'])

    def update(self, extent):
        """ Update and save config. """
        # calculate new extent
        if 'extent' not in self.config:
            self.config['extent'] = extent
        else:
            x1, y1, x2, y2 = zip(extent, self.config['extent'])
            self.config['extent'] = min(x1), min(y1), max(x2), max(y2)

        # save
        path = self.path + '.in'
        json.dump(self.config, open(path, 'w'))
        os.rename(path, self.path)


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
        dataset.SetProjection(MERCATOR)
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
        kwargs = {'projection': MERCATOR,
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
        dataset = self.as_dataset()
        if dataset is not None:
            yield dataset
        yield master


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
    def __init__(self, bbox, zoom, tile, **kwargs):
        self.tile = tile
        self.zoom = zoom
        self.kwargs = kwargs
        self.xranges = bbox.get_xranges(zoom)

    def __len__(self):
        x, y = self.xranges
        return len(x) * len(y)

    def __iter__(self):
        """ Return tile generator. """
        for y, x in itertools.product(*self.xranges):
            yield self.tile(x=x, y=y, z=self.zoom, **self.kwargs)


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


def put(source_path, target_path, classic, single, verbose, zoom, **kwargs):
    """ Create tiles. """
    # inspect target
    try:
        config = Config(path=target_path, zoom=zoom)
    except ValueError as error:
        logging.info('Existing target has zoom {}'.format(error))
        return

    # inspect dataset
    dataset = gdal.Open(source_path)
    bbox = BBox(dataset)

    # select storage
    Storage = storages.FileStorage if classic else storages.ZipFileStorage
    storage = Storage(path=target_path, mode='a')

    pyramid = Pyramid(bbox=bbox, storage=storage, zoom=zoom, **kwargs)

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

    # update config
    config.update(bbox.get_extent())


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
    """ Call put with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs['verbose']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        put(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1


if __name__ == '__main__':
    exit(main())
