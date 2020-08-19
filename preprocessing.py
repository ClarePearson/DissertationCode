import os
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import gdal
import ogr
import earthpy
from earthpy import spatial
import geopandas as gpd
import numpy as np
from rasterio.plot import plotting_extent
from rasterio.warp import calculate_default_transform, reproject, Resampling

# input files
ramsarshp_fn = "C:\\Users\\clare\\MscDiss\\RAMSARsites\\Broadlands.shp"  # Shapefile of area taken from RASMAR all shapefile
flowacc_fn = "C:\\Users\\clare\\MscDiss\\DEM\\OS Terrain 50m\\flowacc.tif"

# to be created
clipout_fn = "C:\\Users\\clare\\MscDiss\\Hydrology\\WBpourpoint.tif"  # output of flow acc clipping to ramsarshp_fn

# TODO: add exceptions for where file directories and files already exist
# --------------------- #
# clipping flow accumulation raster
ramsar = gpd.read_file(ramsarshp_fn)
# clip flow acc by shapefile
with rasterio.open(flowacc_fn) as src_raster:
    cropped_raster, cropped_meta = spatial.crop_image(src_raster, ramsar)

crop_affine = cropped_meta["transform"]
# Create spatial plotting extent for the cropped layer
chm_extent = plotting_extent(cropped_raster[0], crop_affine)
cropped_meta.update({'transform': crop_affine,
                       'height': cropped_raster.shape[1],
                       'width': cropped_raster.shape[2],
                       'nodata': -999.99})
# write tif file of clipped bit
with rasterio.open(clipout_fn, 'w', **cropped_meta) as ff:
    ff.write(cropped_raster[0], 1)

# create copy of flow accumulation raster but values are 0
"""
flowacc_ds = gdal.Open(flowacc_fn)
format = "GTiff"
driver = gdal.GetDriverByName( format )
dst_ds = driver.CreateCopy(blankflowacc_fn, flowacc_ds, 0 )

# append clipped raster to blank flowacc raster
import sys
sys.path.insert(0, r"C:\\Users\\clare\\anaconda3\\envs\\dissertation\\Lib\\site-packages\\GDAL-3.0.2-py3.7-win-amd64.egg-info\\scripts")
import gdal_merge
sys.argv = ['','-o',pourpoint_fn,blankflowacc_fn, clipout_fn]
gdal_merge.main()
"""

# --------------------- #
# reproject image rasters to british national grid
def reproject_et(inpath, outpath, new_crs):
    dst_crs = new_crs # CRS for web meractor

    with rasterio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


bng = 'EPSG:27700'
raw_img_dir = r"C:\Users\clare\MscDiss\Images\0_RAW\WB"
reproj_img_dir = r"C:\Users\clare\MscDiss\Images\1_Reprojected\WB"

for image_dir in os.listdir(raw_img_dir):
    # print (image_dir)
    path = os.path.join(reproj_img_dir, image_dir[-15:-7])
    try:
        os.mkdir(path)
    except:
        pass
    for image_band in os.listdir(raw_img_dir + '\\' + image_dir):
        if image_band.endswith(('B02.jp2', 'B03.jp2', 'B04.jp2', 'B05.jp2', 'B06.jp2', 'B07.jp2',
                                'B08.jp2', 'B8A.jp2', 'B11.jp2', 'B12.jp2')):
            reproject_et(os.path.join(raw_img_dir, image_dir, image_band),
                         os.path.join(path, image_band),
                         bng)


# -------------- #
# crete text file for batch
images_dir = r"C:\Users\clare\MscDiss\Images"
dos_img_dir = r"C:\Users\clare\MscDiss\Images\2_DOS\WB"
reproj_img_dir = r"C:\Users\clare\MscDiss\Images\1_Reprojected\WB"

batch_txt_fn = r"C:\Users\clare\MscDiss\Images\repro_DOS.txt"
f=open(batch_txt_fn, "a+")

for folder in os.listdir(reproj_img_dir):
    print(os.path.join(reproj_img_dir, folder))
    # print(os.path.join(reproj_img_dir, folder, 'MTD_TL.xml'))
    print(os.path.join(images_dir,'2_DOS','MT', folder))
    try:
        # os.mkdir(os.path.join(images_dir,'2_DOS','WB', folder))
        text = str("sentinel2_conversion;input_dir : '"+ os.path.join(reproj_img_dir, folder)
                   + "';mtd_safl1c_file_path : '" + os.path.join(reproj_img_dir, folder, 'MTD_TL.xml')
                   +"';apply_dos1 : 1;dos1_only_blue_green : 0;use_nodata : 0;nodata_value : 0;create_bandset : 0;output_dir : '"
                   + os.path.join(images_dir,'2_DOS','WB', folder) + "';band_set : 0"+"\n")
        f.write(text)
    except:
        pass

f.close()

# -------------- #
# clip files by polygon
clipped_img_dir = r"C:\Users\clare\MscDiss\Images\3_Clipped\WB"
basinpoly_fn = r"C:\Users\clare\MscDiss\Malham\upslope\mt_basin.shp"  # shapefile of polygonised basin from r.water.source

ramsar = gpd.read_file(ramsarshp_fn )

for image_dir in os.listdir(dos_img_dir):
    # print(image_dir)
    path = os.path.join(clipped_img_dir, image_dir)
    try:
        os.mkdir(path)
        for image_band in os.listdir(dos_img_dir + '\\' + image_dir):
            # print(os.path.join(path, image_band))
            # clip flow acc by shapefile
            with rasterio.open(os.path.join(dos_img_dir, image_dir, image_band)) as src_raster:
                cropped_raster, cropped_meta = spatial.crop_image(src_raster, ramsar)

            crop_affine = cropped_meta["transform"]
            cropped_meta.update({'transform': crop_affine,
                                   'height': cropped_raster.shape[1],
                                   'width': cropped_raster.shape[2],
                                   'nodata': -999.99})
            # write tif file of clipped bit
            with rasterio.open(os.path.join(path, image_band), 'w', **cropped_meta) as ff:
                ff.write(cropped_raster[0], 1)
    except:pass


# --------------------- #
# Merging all the dem tiles
"""
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
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                 }
                )
out_fp = r"C:\\Users\\clare\\MscDiss\\DEM\\OS Terrain 50m\\ukDEM.tif"
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)
"""