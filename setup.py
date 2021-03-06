from setuptools import setup

version = '0.5.dev0'

long_description = '\n\n'.join([
    open('README.rst').read(),
    open('CREDITS.rst').read(),
    open('CHANGES.rst').read(),
    ])

install_requires = [
    'setuptools',
    # 'gdal', We do need GDAL but installation via pip does not work optimally.
    #         If you use pip, best to install pygdal. We do not specify pygdal
    #         here to be compatible with installations that use gdal already.
    'Pillow',
    'numpy',
    ],

tests_require = [
    'nose',
    'coverage',
    ]

setup(name='tile-store',
      version=version,
      description=('Storage library with management'
                   'scripts for storage of big tiled maps.'),
      long_description=long_description,
      # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[],
      keywords=[],
      author='arjan.verkerk',
      author_email='arjan.verkerk@nelen-schuurmans.nl',
      url='',
      license='GPL',
      packages=['tile_store'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={'test': tests_require},
      entry_points={
          'console_scripts': [
              'tile-table = tile_store.table:main',
              'tile-put   = tile_store.put:main',
          ]},
      )
