# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:18:14 2020
@author: Clare
Accuracy assessment should be completed in two parts,
1. Creation of stratified sample points,
    where the actual sample points file is checked and amended in QGIS
2. Accuracy assessment measures code
"""

import numpy as np
import gdal
import ogr
from sklearn import metrics
import pandas as pd
import os
from eolearn import geometry
import geopandas as gpd

# File directories
fil_dir = r'C:\Users\Clare\Documents\MscDiss'
temp_fn = os.path.join(fil_dir, 'Images', 'tmp')  # Temp files
site_dir = 'RW'  # name of folder for current site
outshp_fn = os.path.join(fil_dir, 'Images', '6_Classified', site_dir+'_actual.shp')

# read in ref image
for stack_temp_dir in os.listdir(os.path.join(fil_dir, 'Images', '6_Classified', site_dir)):
    if stack_temp_dir.endswith('child_c.tif'):
        for clippedfile in os.listdir(os.path.join(fil_dir, 'Images', '3_Clipped',
                                                   site_dir, stack_temp_dir[:-12])):
            if clippedfile.endswith('_B04.tif'):
                img_ref = gdal.Open(os.path.join(fil_dir, 'Images', '3_Clipped',
                                                 site_dir, stack_temp_dir[:-12], clippedfile), gdal.GA_Update)


# STEP 1
# Creates stratified sample point shapefile based on mean child class output raster
mean_class_fn = gdal.Open(os.path.join(fil_dir, 'Images', '6_Classified', site_dir+'_child_mean.tif'))
mean_child_ds = mean_class_fn.ReadAsArray()  # read in mean class img

# create new sample points, weighted to mean class img
sample_ds = geometry.sampling.PointRasterSampler([4, 5, 6, 7, 8, 9, 10, 11, 12, 13], even_sampling=True)
sample_points = sample_ds.sample(mean_child_ds.astype(np.uint8), n_samples=100)

# get geotransformation information
(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = mean_class_fn.GetGeoTransform()
# calculate x and y coorinates of cells
x_coords = sample_points[1] * x_size + upper_left_x + (x_size / 2)  # add half the cell size
y_coords = sample_points[0] * y_size + upper_left_y + (y_size / 2)  # to centre the point

# create pandas df with x and y coordinates of sample points
df = pd.DataFrame({'xcoord': x_coords, 'ycoord': y_coords})

# iterate through each of the child class tifs in the relevant site directory
for child_img_dir in os.listdir(os.path.join(fil_dir, 'Images', '6_Classified', site_dir)):
    if child_img_dir.endswith('child_c.tif'):
        print(child_img_dir)
        # read in child class image
        child_class = gdal.Open(os.path.join(fil_dir, 'Images', '6_Classified',
                                             site_dir, child_img_dir), gdal.GA_Update)
        child_ds = child_class.ReadAsArray()

        acc_gt = np.zeros(np.shape(child_ds))  # create an empty array the same shape as child ds
        num_rows, num_cols = acc_gt.shape  # get row and column number

        # Create sample point labels
        sample_label = []
        for i in range(len(sample_points[0])):
            sample_label.append(child_ds[sample_points[0][i]][sample_points[1][i]])
        print(sample_label)

        # add columns to pandas df with the child class label for each date column
        df[child_img_dir[:-12]+'_lc'] = sample_label

# create point shapefile
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.xcoord, df.ycoord))
print(gdf.head())
gpd.GeoDataFrame.to_file(gdf, outshp_fn)


# STEP 2
# read in and rasterise QGIS verified ground truth points
test_ds = ogr.Open(outshp_fn)
lyr = test_ds.GetLayer()
layerDefinition = lyr.GetLayerDefn()
driver = gdal.GetDriverByName('MEM')  # gdal driver to create new dataset
target_ds = driver.Create('', img_ref.RasterXSize, img_ref.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(img_ref.GetGeoTransform())
target_ds.SetProjection(img_ref.GetProjection())

# iterate through each of the child class tifs in the relevant site directory
for child_img_dir in os.listdir(os.path.join(fil_dir, 'Images', '6_Classified', site_dir)):
    if child_img_dir.endswith('child_c.tif'):
        field_nm = (child_img_dir[:-12]+'_l')  # get field column name for relevant img date
        options = ['ATTRIBUTE='+field_nm]
        print(child_img_dir[:-12])
        # read in checked ground truth points raster, for the relevant dat field
        gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
        truth = target_ds.GetRasterBand(1).ReadAsArray()

        # read in relevant child class and extract values at sample point locations
        child_class = gdal.Open(os.path.join(fil_dir, 'Images', '6_Classified',
                                             site_dir, child_img_dir), gdal.GA_Update)
        child_ds = child_class.ReadAsArray()
        pred = np.copy(truth)
        idx = np.nonzero(truth)  # find np array indices where truth does not equal zero
        pred[idx] = child_ds[idx]  # pred is equal to the child class where the sample points are

        # confusion matrix of predicted vs ground truthed sample pixels, create csv
        cm = metrics.confusion_matrix(truth[idx], pred[idx])
        ct_df = pd.crosstab(truth[idx], pred[idx], rownames=['True'], colnames=['Predicted'], margins=True)
        ct_df.to_csv(os.path.join(r'C:\Users\Clare\Documents\1. University\6. MSc GIS\7. Dissertation\4. Figures',
                                  site_dir+child_img_dir[:-12]+'.csv'))