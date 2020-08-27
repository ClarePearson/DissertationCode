"""
Code to establish patterns in land cover change
"""
import os
import gdal
import numpy as np

# Directories
fil_dir = r'C:\Users\Clare\Documents\MscDiss\Images' #
temp_fn = os.path.join(fil_dir, 'tmp')  # Temp files
site_dir = 'MT'  # name of folder for current site


def checkShapeSame(numpy_3d):
    """
    Checks all files are the same extent
    :param numpy_3d: 3D numpy array where each item is a different child class img
    :return: True if all are same extent, else false
    """
    shape = []
    for i in numpy_3d:
        shape.append(np.shape(i))
    return len(set(shape)) == 1


def rasterise_layer (file_dir, site_dir, imgSizeRef, raster, output_nm):
    """
    :param file_dir: string, general file directory of project
    :param site_dir: string, name of folder for site
    :param imgSizeRef: gdal dataset, reference image size
    :param raster: numpy array, of image to be rasterised
    :param output_nm: string, to add on end of site dir to produce output file name, must end in '.tif'
    """
    driverTiff = gdal.GetDriverByName('GTiff')
    imgds = driverTiff.Create(os.path.join(file_dir, '6_Classified', site_dir+output_nm),
                              imgSizeRef.RasterXSize, imgSizeRef.RasterYSize,
                              1, gdal.GDT_Float32)
    imgds.SetGeoTransform(img.GetGeoTransform())
    imgds.SetProjection(img.GetProjection())
    imgds.GetRasterBand(1).SetNoDataValue(-9999.0)
    imgds.GetRasterBand(1).WriteArray(raster)
    imgds = None


# read in child_c files as numpy arrays and stack
stack_ds = []
for stack_temp_dir in os.listdir(os.path.join(fil_dir, '6_Classified', site_dir)):
    if stack_temp_dir.endswith('child_c.tif'):
        print(stack_temp_dir)
        img = gdal.Open(os.path.join(fil_dir, '6_Classified',
                                               site_dir, stack_temp_dir), gdal.GA_Update)  # as gdal array
        stack_ds.append(img.ReadAsArray())  # for numpy array
        print(np.unique(img.ReadAsArray(), return_counts=True))
"""
if site_dir == 'MT':
    for i in range(len(stack_ds)):
        stack_ds[i][stack_ds[i] == 9] = 19
        stack_ds[i][stack_ds[i] == 12] = 22
        stack_ds[i][stack_ds[i] == 8] = 9
        stack_ds[i][stack_ds[i] == 11] = 12
        stack_ds[i][stack_ds[i] == 22] = 11
        stack_ds[i][stack_ds[i] == 19] = 8
"""
"""
if not checkShapeSame(stack_ds):
    print ("Rasters different sizes, ")
    exit()
else:
    pass
"""
# create new file, same extent
change_rast = np.zeros(np.shape(stack_ds[0]))
change_count = np.zeros(np.shape(stack_ds[0]))
seasonal = np.zeros(np.shape(stack_ds[0]))

num_rows, num_cols = change_rast.shape

# Pixel Iteration
for row in range(num_rows-1):
    for cell in range(num_cols-1):
        cell_list = []
        # Stack values for each pixel
        for i in range(len(stack_ds)):
            cell_list.append(int(stack_ds[i][row][cell]))
        change_count[row][cell] = len(np.unique(cell_list))  # get number of unique classes per cell
        if len(set(cell_list)) == 1:  # if only one land cover type over all date images
            change_rast[row][cell] = 1
            seasonal[row][cell] = int(np.unique(cell_list))  # if lc stays same, set to original lc
        elif np.count_nonzero(np.array(cell_list) == np.bincount(cell_list).argmax()) == (len(cell_list)-1):
            seasonal[row][cell] = np.bincount(cell_list).argmax()  # if lc stays same except one, set to lc
        elif {4, 5, 6} & set(cell_list):  # if water is present at any point
            if not list(np.unique([ele for ele in cell_list if ele not in {4, 5, 6}])):
                seasonal[row][cell] = 1  # if only different classes of water, set to 1
            elif {7} & set(cell_list):  # if water and bare ground
                if {10, 13} & set(cell_list):
                    if {8, 11} & set(cell_list):
                        if {9, 12} & set(cell_list): seasonal[row][cell] = 14  # water, bare ground and all 3 veg types
                        else: seasonal[row][cell] = 15  # water, bare ground and grass and shrub
                    elif {9, 12} & set(cell_list): seasonal[row][cell] = 16  # water, bare ground and grass and forest
                    else: seasonal[row][cell] = 17  # water, bare ground and grass
                elif {8, 11} & set(cell_list):
                    if {9, 12} & set(cell_list):seasonal[row][cell] = 18  # water, bare ground and shrub and forest
                    else:seasonal[row][cell] = 19  # water, bare ground and shrub
                elif {9, 12} & set(cell_list):seasonal[row][cell] = 20  # water, bare ground and forest
            elif {10, 13} & set(cell_list):
                if {8, 11} & set(cell_list):
                    if {9, 12} & set(cell_list):
                        seasonal[row][cell] = 21  # water and all three veg types
                    else: seasonal[row][cell] = 22  # water, grass and shrub
                elif {9, 12} & set(cell_list):seasonal[row][cell] = 23  # water, grass and forest
                else: seasonal[row][cell] = 24  # water and grass
            elif{8, 11} & set(cell_list):
                if {9, 12} & set(cell_list): seasonal[row][cell] = 25  # water, shrub and forest
                else: seasonal[row][cell] = 26  # water and shrub
            elif {9, 12} & set(cell_list): seasonal[row][cell] = 27  # water and forest
        elif {7} & set(cell_list):  # if bare ground is present at any point
            if {10, 13} & set(cell_list):
                if {8, 11} & set(cell_list):
                    if {9, 12} & set(cell_list): seasonal[row][cell] = 28  # bare ground and all three veg types
                    else: seasonal[row][cell] = 29  # bare ground, grass and shrub
                elif {9, 12} & set(cell_list): seasonal[row][cell] = 30  # bare ground, grass and forest
                else: seasonal[row][cell] = 31  # bare ground and grass
            elif{8, 11} & set(cell_list):
                if {9, 12} & set(cell_list): seasonal[row][cell] = 32  # bare ground , shrub and forest
                else: seasonal[row][cell] = 33 # bare ground and shrub
            elif {9, 12} & set(cell_list): seasonal[row][cell] = 34  # bare ground and forest
        elif {8, 9, 10, 11, 12, 13} & set(cell_list):  # if bare ground is present at any point
            if {10, 13} & set(cell_list):
                if {8, 11} & set(cell_list):
                    if {9, 12} & set(cell_list): seasonal[row][cell] = 35  # all three veg types
                    else: seasonal[row][cell] = 36  # grass and shrub
                elif {9, 12} & set(cell_list): seasonal[row][cell] = 37  # grass and forest
                else: seasonal[row][cell] = 38  # grass and productive grass
            elif{8, 11} & set(cell_list):
                if {9, 12} & set(cell_list): seasonal[row][cell] = 39  # shrub and forest
                else: seasonal[row][cell] = 40  # shrub and productive shrub
            elif {9, 12} & set(cell_list): seasonal[row][cell] = 41  # forest and productive forest

# rasterise
rasterise_layer(fil_dir, site_dir, img, change_rast, '_change.tif')
rasterise_layer(fil_dir, site_dir, img, change_count, '_changecount.tif')
rasterise_layer(fil_dir, site_dir, img, seasonal, '_seasonal.tif')
