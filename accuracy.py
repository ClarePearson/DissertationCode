# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:18:14 2020

@author: Clare

Code adapted from https://opensourceoptions.com

"""

import numpy as np
import gdal
import ogr
from sklearn import metrics
import pandas as pd

# File directories
fil_dir = r'C:\Users\clare\MscDiss\Images' #
temp_fn = os.path.join(fil_dir, 'tmp')  # Temp files
site_dir = 'WB'  # name of folder for current site
test_ds = ogr.Open(os.path.join(fil_dir, 'GT','WB_GT.shp'))

for stack_temp_dir in os.listdir(os.path.join(fil_dir, '4_Stacked', site_dir)):
    if stack_temp_dir.endswith('_stack_temp.kea'):
        # image stack fro driver and size details
        stack_temp_ds = gdal.Open(os.path.join(fil_dir, '4_Stacked',site_dir, stack_temp_dir), gdal.GA_Update)
        # predicted
        pred_ds = gdal.Open('C:\\Users\\Clare\Documents\\MscDiss\\\Malham\\clipS2A\\classified.tif')

# Reading in files
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