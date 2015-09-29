tile-store
==========

Introduction
------------

This library stores tiles in a directory structure. Currently two types
of storage are available:

FileStorage: /path/to/tiles/z/x/y

ZipFileStorage: path/to/tiles/07/90/b260b82ce2f2d56f84a4035e6a71.zip
                with zipfile entries like b381526dc6b8942ef945a06365fea598

The second storage offers a much more balanced folder structure and
limits the total amount of files by grouping up to 256 images together
in uncompressed zipfiles. Of course when reading there is an extra seek
operation involved for the zipfile index and the zoom structure is no
longer obvious from inspection of the main folder.

Neither type stores the image with an extension. Tiles that contain any
transparency (such as at edges of the datasource) are stored with PNG
compression, whereas the others are stored with JPEG compression. It
is up to the user of the library to determine the type from the first
bytes. See for example:

https://en.wikipedia.org/wiki/Magic_number_(programming)


Usage
-----

The sources needs to be a 4-band (rgba) GDAL datasource. If only 3-band
data is available, the easiest way to add alpha to it is to make a
virtual dataset with added alpha::

    $ gdalbuildvrt source.vrt source.tif -addalpha

To create a bunch of tiles from an rgba source use the creator script::

    $ bin/tile-creator source.tif path/to/tiles 15

To use the created storage in another application::

    >>> from tile_store import storages
        storage = storages.ZipFileStorage('path/to/tiles')
        image_data = storage[x, y, z]

To determine a suitable zoom one could use the zoom table::

    $ bin/tile-table


Post-nensskel setup TODO
------------------------

Here are some instructions on what to do after you've created the project with
nensskel.

- Add a new jenkins job at
  http://buildbot.lizardsystem.nl/jenkins/view/djangoapps/newJob or
  http://buildbot.lizardsystem.nl/jenkins/view/libraries/newJob . Job name
  should be "tile-store", make the project a copy of the existing "lizard-wms"
  project (for django apps) or "nensskel" (for libraries). On the next page,
  change the "github project" to ``https://github.com/nens/tile-store/`` and
  "repository url" fields to ``git@github.com:nens/tile-store.git`` (you might
  need to replace "nens" with "lizardsystem"). The rest of the settings should
  be OK.

- The project is prepared to be translated with Lizard's
  `Transifex <http://translations.lizard.net/>`_ server. For details about
  pushing translation files to and fetching translation files from the
  Transifex server, see the ``nens/translations`` `documentation
  <https://github.com/nens/translations/blob/master/README.rst>`_.
