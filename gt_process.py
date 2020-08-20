"""
Created on Mon May 25 19:20:00 2020

@author: Clare

Code from https://opensourceoptions.com

"""
import gdal
import numpy as np
import geopandas as gpd
import pandas as pd

# read ground truth points shapefile to geopandas geodataframe
gd_f = gpd.read_file ('C:\\Users\\clare\\MscDiss\\GT\\WB_GT.shp')
# get names of land cover classes/labels
class_names = gd_f['lc'].unique()
print('class names', class_names)

# raster cant hold string, so have to create corresponding id number
# create a unique id (integer) for each land cover class/label
class_ids = np.arange(class_names.size) + 1
print('class ids', class_ids)

# create a pandas data frame of the labels and ids and save to csv
df = pd.DataFrame({'lc': class_names, 'id': class_ids})
df.to_csv('C:\\Users\\Clare\\MscDiss\\GT\\WB_class_lookup.csv')
print('gd_f without ids',gd_f.head())
# add a new column to geodatafame with the id for each class/label
gd_f['id']= gd_f['lc'].map(dict(zip(class_names,class_ids)))
print('gd_f with ids',gd_f.head())


# split into training and test subsets
gdf_train = gd_f.sample(frac=0.7)
gdf_test = gd_f.drop(gdf_train.index)
print('gd_f shape', gd_f.shape,'training shape', gdf_train.shape, 'test', gdf_test.shape)
gdf_train.to_file('C:\\Users\\Clare\\MscDiss\\GT\\WB_gt_train.shp')
gdf_test.to_file('C:\\Users\\Clare\\MscDiss\\GT\\WB_gt_test.shp')