# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:18:14 2020
@author: Clare
Code adapted from https://opensourceoptions.com
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

"""
PLAN:
1. create stratified sample of child class - two copies, one site_date_actual next site_date_predict
2. polygonise, where lc is the number representing the lc class
3. check in qgis, ammending numebrs where wrong
4. read back in amended shapefile, rasterise
5. confusion matrix
6. iterate through for all files
"""

# STEP 1
# File directories
fil_dir = r'C:\Users\Clare\Documents\MscDiss\Images'
temp_fn = os.path.join(fil_dir, 'tmp')  # Temp files
outrast_fn = r'C:\Users\Clare\Documents\MscDiss\GT\test.tif'
stack_temp_ds = gdal.Open(os.path.join(r'C:\Users\Clare\Documents\MscDiss\Images\3_Clipped\WB\20191208\RT_T30UYD_20191208T111441_B04.tif'), gdal.GA_Update)
site_dir = 'WB'  # name of folder for current site

child_ds = gdal.Open(r'C:\Users\Clare\Documents\MscDiss\Images\6_Classified\WB\20200206_child_c.tif')
child_ds = child_ds.ReadAsArray()

acc_gt = np.zeros(np.shape(child_ds))
num_rows, num_cols = acc_gt.shape

# Sample points
sample_ds = geometry.sampling.PointSampler(child_ds, no_data_value=None, ignore_labels=None)
sample_ds.labels()
sample_points = sample_ds.sample(nsamples = 200, weighted = True)

for i in range(len(sample_points[0])):
    acc_gt[sample_points[1][i]][sample_points[2][i]] = sample_points[0][i]

# rasterise
driverTiff = gdal.GetDriverByName('GTiff')
imgds = driverTiff.Create(outrast_fn,
                          stack_temp_ds.RasterXSize, stack_temp_ds.RasterYSize,
                          1, gdal.GDT_Float32)
imgds.SetGeoTransform(stack_temp_ds.GetGeoTransform())
imgds.SetProjection(stack_temp_ds.GetProjection())
imgds.GetRasterBand(1).SetNoDataValue(-9999.0)
imgds.GetRasterBand(1).WriteArray(acc_gt)
imgds = None

# create shapefile
class_names = np.unique(acc_gt)
class_names = class_names[1:len(class_names)]




# STEP 2
# Reading in files
test_ds = ogr.Open(os.path.join(fil_dir, 'GT','WB_GT.shp'))
driverTiff = gdal.GetDriverByName('GTiff')
lyr = test_ds.GetLayer()
pred = pred_ds.GetRasterBand(1).ReadAsArray()

# read in and rasterise test datapoints
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', stack_temp_ds.RasterXSize, stack_temp_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(stack_temp_ds.GetGeoTransform())
target_ds.SetProjection(stack_temp_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
truth = target_ds.GetRasterBand(1).ReadAsArray()

# find locations where truth does not equal zero
idx = np.nonzero(truth)

#pixel by pixel comfucsionmatrix, not segment
cm = metrics.confusion_matrix(truth[idx], pred[idx])
pd.crosstab(truth[idx], pred[idx], rownames=['True'], colnames=['Predicted'], margins=True)
 
# pixel accuracy
print(cm)
 
print(cm.diagonal())
print(cm.sum(axis=0))

# per class accuracy
accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)