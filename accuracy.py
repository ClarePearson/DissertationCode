# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:18:14 2020
@author: Clare
Accuracy assessment should be completed in two parts,
1. Creation of stratified sample points,
    in which the actual sample points file is checked and amended in QGIS
2. Accuracy assessment measures code
"""

import numpy as np
import gdal
import ogr
from osgeo import osr
from sklearn import metrics
import pandas as pd
import os
from eolearn import geometry
import geopandas as gpd

"""
PLAN:
4. read back in amended shapefile, rasterise
5. confusion matrix
"""

# STEP 1
# File directories
fil_dir = r'C:\Users\Clare\Documents\MscDiss'
temp_fn = os.path.join(fil_dir, 'Images', 'tmp')  # Temp files
site_dir = 'WB'  # name of folder for current site

mean_class_fn = gdal.Open(os.path.join(fil_dir, 'Images','6_Classified',site_dir+'_child_mean.tif'))
mean_child_ds = mean_class_fn.ReadAsArray()

# create sample points
sample_ds = geometry.sampling.PointRasterSampler(np.unique(mean_child_ds), even_sampling=False)
sample_points = sample_ds.sample(mean_child_ds.astype(np.uint8), n_samples=100)  # ,weighted=True
outshp_fn = os.path.join(fil_dir, 'Images', '6_Classified', site_dir+'_actual.shp')

# get geotransformation information
(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = mean_class_fn.GetGeoTransform()

# calculate x and y coorinates of cells
x_coords = sample_points[1] * x_size + upper_left_x + (x_size / 2)  # add half the cell size
y_coords = sample_points[0] * y_size + upper_left_y + (y_size / 2)  # to centre the point

# create geopandas dataframe and create shapefile
df = pd.DataFrame({'xcoord': x_coords, 'ycoord': y_coords})

for stack_temp_dir in os.listdir(os.path.join(fil_dir, 'Images', '6_Classified', site_dir)):
    if stack_temp_dir.endswith('child_c.tif'):
        print(stack_temp_dir)
        # read in child class
        child_class = gdal.Open(os.path.join(fil_dir, 'Images', '6_Classified',
                                             site_dir, stack_temp_dir), gdal.GA_Update)
        child_ds = child_class.ReadAsArray()

        # read in ref image
        for clippedfile in os.listdir(os.path.join(fil_dir, 'Images', '3_Clipped',
                                                   site_dir, stack_temp_dir[:-12])):
            if clippedfile.endswith('_B04.tif'):
                img_ref = gdal.Open(os.path.join(fil_dir, 'Images', '3_Clipped',
                                                 site_dir, stack_temp_dir[:-12], clippedfile), gdal.GA_Update)

        # declare names for files to be created
        outrast_fn = os.path.join(fil_dir, 'GT', site_dir, stack_temp_dir[:-12]+'_predict.tif')

        acc_gt = np.zeros(np.shape(child_ds))
        num_rows, num_cols = acc_gt.shape

        # Get sample point labels
        sample_label = []
        for i in range(len(sample_points[0])):
            #print(child_ds[sample_points[0][i]][sample_points[1][i]])
            sample_label.append(child_ds[sample_points[0][i]][sample_points[1][i]])
        print(sample_label)

        for i in range(len(sample_points[0])):
            acc_gt[sample_points[0][i]][sample_points[1][i]] = sample_label[i]

        # rasterise sample points
        driverTiff = gdal.GetDriverByName('GTiff')
        imgds = driverTiff.Create(outrast_fn,
                                  img_ref.RasterXSize, img_ref.RasterYSize,
                                  1, gdal.GDT_Float32)
        imgds.SetGeoTransform(img_ref.GetGeoTransform())
        imgds.SetProjection(img_ref.GetProjection())
        imgds.GetRasterBand(1).SetNoDataValue(-9999.0)
        imgds.GetRasterBand(1).WriteArray(acc_gt)
        imgds = None

        # Read in sample point raster
        sample_rast = gdal.Open(outrast_fn)
        sample_rast_ds = sample_rast.GetRasterBand(1)
        sample_rast_np = sample_rast.ReadAsArray().astype(np.float)

        # create geopandas dataframe and create shapefile
        df[stack_temp_dir[:-12]+'_lc'] = sample_label

# create shapefile
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.xcoord, df.ycoord))
print(gdf.head())
gpd.GeoDataFrame.to_file(gdf, outshp_fn)




driver = ogr.GetDriverByName("ESRI Shapefile")

# STEP 2
# read in and rasterise test datapoints
test_ds = ogr.Open(outshp_fn)
lyr = test_ds.GetLayer()
layerDefinition = lyr.GetLayerDefn()

# for i in range(layerDefinition.GetFieldCount()):
#     print (layerDefinition.GetFieldDefn(i).GetName())

# gdal driver to create new dataset
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', img_ref.RasterXSize, img_ref.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(img_ref.GetGeoTransform())
target_ds.SetProjection(img_ref.GetProjection())

# rasterise
for stack_temp_dir in os.listdir(os.path.join(fil_dir, 'Images', '6_Classified', site_dir)):
    if stack_temp_dir.endswith('child_c.tif'):
        # declare names for files to be created
        field_nm = (stack_temp_dir[:-12]+'_l')
        options = ['ATTRIBUTE='+field_nm]

        gdal.RasterizeLayer(target_ds, [1], lyr, options=options)  # where saving to, band, input vector
        truth = target_ds.GetRasterBand(1).ReadAsArray()  # retrieve the rasterised data

        # find locations where truth does not equal zero
        idx = np.nonzero(truth)

        #pixel by pixel comfucsionmatrix, not segment
        cm = metrics.confusion_matrix(truth[idx], child_ds[idx])
        pd.crosstab(truth[idx], child_ds[idx], rownames=['True'], colnames=['Predicted'], margins=True)

        # pixel accuracy
        print(cm)

        print(cm.diagonal())
        print(cm.sum(axis=0))

        # per class accuracy
        accuracy = cm.diagonal() / cm.sum(axis=0)
        print(accuracy)

