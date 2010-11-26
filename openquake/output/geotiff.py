# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
"""
A trivial implementation of the GeoTiff format,
using GDAL.

In order to make this run, you'll need GDAL installed,
and on the Mac I couldn't get the brew recipe to work.
I recommend the DMG framework at 
http://www.kyngchaos.com/software:frameworks.

I had to add the installed folders to 
PYTHONPATH in my .bash_profile file to get them to load.
"""

import numpy
from osgeo import osr, gdal

from openquake import logs
from openquake import writer

LOG = logs.LOG

GDAL_FORMAT = "GTiff"
GDAL_PIXEL_DATA_TYPE = gdal.GDT_Float32
SPATIAL_REFERENCE_SYSTEM = "WGS84"
TIFF_BAND = 4
TIFF_LONGITUDE_ROTATION = 0
TIFF_LATITUDE_ROTATION = 0

class GeoTiffFile(writer.FileWriter):
    """Rough implementation of the GeoTiff format,
    based on http://adventuresindevelopment.blogspot.com/2008/12/
                python-gdal-adding-geotiff-meta-data.html
    """
    
    format = GDAL_FORMAT
    normalize = True
    
    def __init__(self, path, image_grid, init_value=numpy.nan, normalize=False):
        self.grid = image_grid
        self.normalize = normalize
        # NOTE(fab): GDAL initializes the image as columns x rows.
        # numpy arrays, however, have usually rows as first axis,
        # and columns as second axis (as it is the convention for
        # matrices in maths)
        
        # initialize raster to init_value values (default in NaN)
        self.raster = numpy.ones((self.grid.rows, self.grid.columns),
                                 dtype=numpy.float) * init_value
        self.alpha_raster = numpy.ones((self.grid.rows, self.grid.columns),
                                 dtype=numpy.float) * 32.0
        self.target = None
        super(GeoTiffFile, self).__init__(path)
        
    def _init_file(self):
        driver = gdal.GetDriverByName(self.format)

        # NOTE(fab): use GDAL data type GDT_Float32 for science data
        pixel_type = GDAL_PIXEL_DATA_TYPE
        if self.normalize:
            pixel_type = gdal.GDT_Byte
        self.target = driver.Create(self.path, self.grid.columns, 
            self.grid.rows, TIFF_BAND, pixel_type)
        
        corner = self.grid.region.upper_left_corner

        # this is the order of arguments to SetGeoTransform()
        # top left x, w-e pixel resolution, rotation, 
        # top left y, rotation, n-s pixel resolution
        # rotation is 0 if image is "north up" 
        # taken from http://www.gdal.org/gdal_tutorial.html

        # NOTE(fab): the last parameter (grid spacing in N-S direction) is 
        # negative, because the reference point for the image is the 
        # upper left (north-western) corner
        self.target.SetGeoTransform(
            [corner.longitude, self.grid.cell_size, TIFF_LONGITUDE_ROTATION, 
             corner.latitude, TIFF_LATITUDE_ROTATION, -self.grid.cell_size])

        # set the reference info 
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS(SPATIAL_REFERENCE_SYSTEM)
        self.target.SetProjection(srs.ExportToWkt())

        # This doesn't work with the eventlet tpool stuff.
        # self.file = tpool.Proxy(open(self.path, 'w'))
    
    def write(self, cell, value):
        """Stores the cell values in the NumPy array for later 
        serialization. Make sure these are zero-based cell addresses."""
        self.raster[int(cell[0]), int(cell[1])] = float(value)
        # Set AlphaLayer
        if value:
            self.alpha_raster[int(cell[0]), int(cell[1])] = 255.0

    def close(self):
        """Make sure the file is flushed, and send exit event"""
        
        # NOTE(fab): numpy raster does not have to be transposed, although
        # it has rows x columns
        if self.normalize:
            self.raster = self.raster * 254.0 / self.raster.max()

        self.target.GetRasterBand(1).WriteArray(self.raster)
        self.target.GetRasterBand(2).Fill(0.0)
        self.target.GetRasterBand(3).Fill(0.0)

        # Write alpha channel
        self.target.GetRasterBand(4).WriteArray(self.alpha_raster)

        self.target = None  # This is required to flush the file
        self.finished.send(True)
    
    def serialize(self, iterable):
        # TODO(JMC): Normalize the values
        maxval = max(iterable.values())
        for key, val in iterable.items():
            if self.normalize:
                val = val/maxval*254
            self.write((key.column, key.row), val)
        self.close()
