"""
Preprocessing to OBIA methods
Each of three sections must be run in the following sequence:
1. Image reprojection
2. Creation of text file for batch processing of QGIS Semi-Automatic Classifictaion plugin (SCP)
    Dark Object Subtraction (DOS)
3. Run batch text file in QGIS
4. Clip files to extent of ramsar shapefile
"""

import os
import rasterio
from earthpy import spatial
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling

# TODO: automate with RW/MT/WB
# Input files for reprojection
bng = 'EPSG:27700' # British NAtional Gris Projection code
raw_img_dir = r"C:\Users\Clare\Documents\MscDiss\Images\0_RAW\RW"  # File directory for raw images
reproj_img_dir = r"C:\Users\Clare\Documents\MscDiss\Images\1_Reprojected\RW"  # File directory for reprojected images

# Input files for SCP DOS batch text file creation
images_dir = r"C:\Users\Clare\Documents\MscDiss\Images"
dos_img_dir = r"C:\Users\Clare\Documents\MscDiss\Images\2_DOS\RW"
batch_txt_fn = r"C:\Users\Clare\Documents\MscDiss\Images\repro_DOS.txt"

# Input files for clipping images to RAMSAR site extent
clipped_img_dir = r"C:\Users\Clare\Documents\MscDiss\Images\3_Clipped\RW"
# Shapefile of area taken from RASMAR all shapefile
ramsarshp_fn = "C:\\Users\\Clare\\Documents\\MscDiss\\RAMSARsites\\Rutlands.shp"


def reproject_et(inpath, outpath, new_crs):
    """
    reprojects created reprojected output image from input inputted image
    :param inpath: String, file path of input image to be reprojected
    :param outpath: String, file path of output image to be created
    :param new_crs: String, Projection system to be reprojected to
    """
    dst_crs = new_crs  # CRS for web meractor

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

# -------------- #
# Reprojection
# Iterate through raw data files and reproject selected image bands
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
# Create text file for SCP DOS batch
f=open(batch_txt_fn, "a+")
for folder in os.listdir(reproj_img_dir):
    print(os.path.join(reproj_img_dir, folder))
    # print(os.path.join(reproj_img_dir, folder, 'MTD_TL.xml'))
    print(os.path.join(images_dir,'2_DOS','RW', folder))
    try:
        # os.mkdir(os.path.join(images_dir,'2_DOS','RT', folder))
        text = str("sentinel2_conversion;input_dir : '"+ os.path.join(reproj_img_dir, folder)
                   + "';mtd_safl1c_file_path : '" + os.path.join(reproj_img_dir, folder, 'MTD_TL.xml')
                   +"';apply_dos1 : 1;dos1_only_blue_green : 0;use_nodata : 0;nodata_value : 0;create_bandset : 0;output_dir : '"
                   + os.path.join(images_dir,'2_DOS','RW', folder) + "';band_set : 0"+"\n")
        f.write(text)
    except:
        pass

f.close()

# -------------- #
# clip files by polygon
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

