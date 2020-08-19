
import os
import gdal
import numpy as np

# Directories
fil_dir = r'C:\Users\clare\MscDiss\Images' #
temp_fn = os.path.join(fil_dir, 'tmp')  # Temp files
site_dir = 'WB'  # name of folder for current site

stack_ds = []
# read in child_c files as numpy arrays and stack
for stack_temp_dir in os.listdir(os.path.join(fil_dir, '6_Classified', site_dir)):
    if stack_temp_dir.endswith('child_c.tif'):
        print(stack_temp_dir)
        img = gdal.Open(os.path.join(fil_dir, '6_Classified',
                                               site_dir, stack_temp_dir), gdal.GA_Update) # as gdal array
        stack_ds.append(img.ReadAsArray()) # for numpy array

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


if not checkShapeSame(stack_ds):
    print ("Rasters different sizes, ")
    exit()
else:
    pass

# create new file, same extent
change_rast = np.zeros(np.shape(stack_ds[0]))

num_rows, num_cols = change_rast.shape

# Pixel Iteration
for row in range(num_rows-1):
    if row % 100 == 0:
        print('Row number processed: ' + str(row))
    for cell in range(num_cols-1):
        # print('cell = ' + str(cell) + ' row = ' + str(row))
        cell_list = []
        # Stack values for each pixel
        for i in range(len(stack_ds)):
            cell_list.append(int(stack_ds[i][row][cell]))
        # print(cell_list)
        if not len(set(cell_list)) == 1:
            change_rast[row][cell] = 1

# if water at somepoint then not - seasonally inundated.


driverTiff = gdal.GetDriverByName('GTiff')
imgds = driverTiff.Create(r"C:\Users\clare\MscDiss\Images\6_Classified\WB_change.tif",
                          img.RasterXSize, img.RasterYSize,
                          1, gdal.GDT_Float32)
imgds.SetGeoTransform(img.GetGeoTransform())
imgds.SetProjection(img.GetProjection())
imgds.GetRasterBand(1).SetNoDataValue(-9999.0)
imgds.GetRasterBand(1).WriteArray(change_rast)
imgds = None