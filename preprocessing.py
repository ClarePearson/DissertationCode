import os
import rasterio
from rasterio.merge import merge

directory = 'C:\\Users\\clare\\MscDiss\\DEM\\OS Terrain 50m\\terrain-50-dtm_3581326'

os.listdir(directory)

dem_list = []

for folder in os.listdir(directory):
    for file in os.listdir(str(directory) + '\\' + str(folder)):
        if file.endswith(".asc"):
            dem_list.append(str(directory) + '\\' + str(folder) + '\\' + str(file))
        else:
            continue

dem_ds = []
for fp in dem_list:
    src = rasterio.open(fp)
    dem_ds.append(src)

mosaic, out_trans = merge(dem_ds)