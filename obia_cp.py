"""
TODO: child hier class using objects and spectral info
TODO: Add other statistics to RAT: proximity to water, area
TODO: Empty the file location?
TODO: Cleanup

"""
import os
import gdal
import ogr
import rios
from rios import rat
import numpy as np
import rsgislib
from rsgislib import imageutils
from rsgislib import rastergis
from rsgislib.rastergis import ratutils
from rsgislib.segmentation import segutils
# import sklearn  # preprocessing/measure
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from skimage import measure

# ------------------------------------------------- #
# Define File Names
temp_fn = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib'  # Temp files
segments_fn = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\segments.kea'  # Output Segmentation
fil_dir = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\'  # TODO: Merge with temp_fn
stan_rast_fn = str(fil_dir + 'stack_stan.kea')
stack_temp = str(fil_dir + 'stack_temp.kea')
stack_fn = str(fil_dir + 'stack.kea')
parent_fn = str(fil_dir + 'parent_c.kea')
rfveg_fn = str(fil_dir + 'rfveg_c.tif')
child_fn = str(fil_dir + 'child_c.tif')

# ------------------------------------------------- #
# Image File stacking - create image stack of sentinel MT images
# Str of image band file to be stacked
# MUST BE IN THIS ORDER
sentImgList = ['C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B02.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B03.tif ',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B04.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B05.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B06.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B07.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B08.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B8A.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B9.tif',
               'C:\\Users\\clare\\MscDiss\\Malham\\clipS2A\\S20200419_B10.tif',
               ]

# Str of image bad names to be stacked - ORDER IMPORTANT
sentBandNames = ['Band02', 'Band03', 'Band04', 'Band05',
                 'Band06', 'Band07', 'Band08', 'Band8A', 'Band09', 'Band10']
# Create Image Stack
imageutils.stackImageBands(sentImgList, sentBandNames, stack_temp, None, 0, 'KEA', rsgislib.TYPE_32FLOAT)
# Read in GDAL file
stack_temp_ds = gdal.Open(stack_temp, gdal.GA_Update)  # as gdal array
# stack_arr = stack_ds.ReadAsArray() # for numpy array


def rasterShape(gdalRasterStack):
    """
    :param gdalRasterStack:
    :return: 3 int list of 1) number of bands, 2) number of rows and 3) number of columns
    """
    nbands = gdalRasterStack.RasterCount
    nrows = gdalRasterStack.RasterYSize
    ncols = gdalRasterStack.RasterXSize
    return nbands, nrows, ncols


def rescaler(band_array):
    """
    Rescales the input band to 0 and 1
    :param band_array: numpy array
    :return: rescaled numpy array between 0 and 1
    """
    bmin = band_array.min(axis=(0, 1), keepdims=True)
    if bmin < 0:
        band_array = band_array + -bmin
    bmin = band_array.min(axis=(0, 1), keepdims=True)
    bmax = band_array.max(axis=(0, 1), keepdims=True)
    xformed = (band_array - bmin) / (bmax - bmin)
    return xformed


def checknormalisation(band_array):
    """
    check normalisation and band ratio appendix has worked
    :param band_array: input numpy array of raster
    :return: prints band max and min
    """
    print(band_array.min())
    print(band_array.max())


# ------------------------------------------------- #
# Create array of band information
band_data = []
for i in range(1, rasterShape(stack_temp_ds)[0] + 1):
    band = stack_temp_ds.GetRasterBand(i).ReadAsArray()
    # print('band ' + str(i) + ' ' + str(band.max()) + ' ' + str(band.min()))
    band_data.append(band)
# Calculate band ratios
NDVIband = (band_data[6] - band_data[2]) / (band_data[6] + band_data[2])  # NDVI = (NIR – Red)/( NIR + Red)
NDWIband = (band_data[1] - band_data[6]) / (band_data[1] + band_data[6])  # NDWI = (Green - NIR)/(Green + NIR)
band_data.append(NDVIband)
band_data.append(NDWIband)

# Rescaling files between 0 and 1
# Create standardised raster stack for segmentation algorithm
# look through bands in raster stack, rescale and add to array
band_data_stan = []
for i in range(1, rasterShape(stack_temp_ds)[0] + 1):
    band_s = stack_temp_ds.GetRasterBand(i).ReadAsArray()
    band_s = rescaler(band_s)
    # checknormalisation(band)
    band_data_stan.append(band_s)
# band ratio rescale and append to band array
NDVIband = rescaler(NDVIband)
band_data_stan.append(NDVIband)
NDWIband = rescaler(NDWIband)
band_data_stan.append(NDWIband)

NDVIband = None
NDWIband = None
band = None
band_s = None

# ------------------------------------------------- #
# Creating 2 raster stack files of 1) standardised and
# 2) non-standardised bands and band ratios


def rasteriseOutput(stack_dataset, imgName, img, fn_type):
    """
    creates a .kea output file of the inputted image
    :param stack_dataset: gdal dataset of image band
    :param imgName: String of file name of image e.g. '...\\classified_img.kea'
    :param img: numpy array of image - single band only
    :param fn_type: String of file type e.g. KEA or MEM
    """
    driverTiff = gdal.GetDriverByName(fn_type)
    imgds = driverTiff.Create(imgName,
                              stack_dataset.RasterXSize, stack_dataset.RasterYSize,
                              1, gdal.GDT_Float32)
    imgds.SetGeoTransform(stack_dataset.GetGeoTransform())
    imgds.SetProjection(stack_dataset.GetProjection())
    imgds.GetRasterBand(1).SetNoDataValue(-9999.0)
    imgds.GetRasterBand(1).WriteArray(img)
    imgds = None


def stackRast(band_stack, directory, rast_fn, band_nm, ref_img):
    """
    :param band_stack: Numpy array of raster stack
    :param directory: String of file location to put temp and output file in
    :param ref_img: gdal dataset of a reference image, used for driverTiff creation
    :param rast_fn: String of directory and file name of image stack file to be made
    :param band_nm: list of strings containing band names
    :return: Tif file which is a raster stack of standardised input bands and band ratios
    """
    bands = []
    for j in range(band_stack.shape[2]):
        bands.append(str(directory + 'tempband_' + str(j) + '.kea'))  # add file name to list
        rasteriseOutput(ref_img, str(directory + 'tempband_' + str(j) + '.kea'), band_stack[:, :, j], 'KEA')
        # print(str(directory + 'stan_band_' + str(j) + '.tif'))
    # print(bands)
    # band_names = []
    band_names = band_nm.copy()
    band_names.append('NDVI')
    band_names.append('NDWI')
    # print(band_names)
    # print(rast_fn)
    imageutils.stackImageBands(bands, band_names, rast_fn, None, float(-9999.0), 'KEA', rsgislib.TYPE_32FLOAT)

    # remove temp files
    for j in range(len(bands)):
        os.remove(str(fil_dir + 'tempband_' + str(j) + '.kea'))


# convert band array stack to numpy array, third axis is number of bands
band_data = np.dstack(band_data)
band_data_stan = np.dstack(band_data_stan)

stackRast(band_data, fil_dir, stack_fn, sentBandNames, stack_temp_ds)  # Stack normal band data as kea file
stackRast(band_data_stan, fil_dir, stan_rast_fn, sentBandNames, stack_temp_ds)  # Stack standardised band data as kea

print("Normalised raster stack created...")
# ------------------------------------------------- #
# Segmentation process
print("Segmenting image...")
segutils.runShepherdSegmentation(stan_rast_fn, segments_fn, tmpath=temp_fn,
                                 gdalformat='KEA',
                                 noStretch=True,  # Issue when this is false
                                 noStats=True,  # Issue when this is false
                                 numClusters=60,
                                 minPxls=50,
                                 distThres=100,
                                 sampling=100, kmMaxIter=200)
# ------------------------------------------------- #
# Image statistics
print('Calculating image statistics...')
# Spectral stats
rastergis.populateStats(segments_fn)
ratutils.populateImageStats(stack_fn, segments_fn,
                                      calcMean=True,
                                      calcMin=True,
                                      calcMax=True,
                                      calcStDev=True)  # populates clumps file with stats
# reading in RAT to add to
segment_rat = gdal.Open(segments_fn, gdal.GA_Update)  # Read in segments raster as gdal
segment_ds = segment_rat.GetRasterBand(1).GetDefaultRAT()  # Read in RAT
band = segment_rat.GetRasterBand(1).ReadAsArray()  # read in segment raster band, w seg ids
band_data = np.dstack(band)
segment_ids = np.unique(band)  # extract seg ids

# ------------------------------------ #
# create list of object stats, where each row is an object ro1[1] = [1,2,3,4,5]
# TODO: move above where parent class is
obj_stats = []
for i in range(1,int(segment_ds.GetRowCount())):
    row_stat = []
    for j in range(int(segment_ds.GetColumnCount())):
        stat = segment_ds.GetValueAsDouble(i, j)
        row_stat.append(stat)
    obj_stats.append(row_stat)
# ------------------------------------ #
# Geometry
segment_ds.CreateColumn('obj_area', gdal.GFT_Integer, gdal.GFU_Generic)  # col 53
segment_ds.CreateColumn('obj_width', gdal.GFT_Integer, gdal.GFU_Generic)  # col 54
segment_ds.CreateColumn('obj_length', gdal.GFT_Integer, gdal.GFU_Generic)  # col 55
segment_ds.CreateColumn('obj_lwRatio', gdal.GFT_Integer, gdal.GFU_Generic)  # col 55
for i in range(1, int(segment_ds.GetRowCount())):  # iterate down rows for number of rows in input stack
    # print('Object ' + str(i) + ' no of pixels = ' + str(np.count_nonzero(band == i)))
    segment_ds.SetValueAsInt(i, 53, np.count_nonzero(band == i))  # no of pixels
    segment_ds.SetValueAsInt(i, 54, int(max(np.count_nonzero(band == i, axis=1))))  # max width
    segment_ds.SetValueAsInt(i, 55, int(max(np.count_nonzero(band == i, axis=0))))  # max length/height
    segment_ds.SetValueAsInt(i, 56, int(max(np.count_nonzero(band == i, axis=0))/max(np.count_nonzero(band == i, axis=1))*100)) # length/width ratio
# ------------------------------------------------- #
# Hierarchical Classification
# Reading in segment RAT
print("Hierarchical Classification...")

NDWIAvg = rat.readColumn(segment_rat, "NDWIAvg")
Band10Max = rat.readColumn(segment_rat, "Band10Max")
NDVIAvg = rat.readColumn(segment_rat, "NDVIAvg")

# Creating and population parent class column
segment_ds.CreateColumn('p_class', gdal.GFT_Integer, gdal.GFU_Generic)
for i in range(int(segment_ds.GetRowCount())):  # iterate down rows for number of rows in input stack
    if segment_ds.GetValueAsInt(i, (segment_ds.GetColumnCount()-1)) == 0:  # if no class (0) in p_class column
        if NDWIAvg[i] > -0.3:  # if NDWIAvg (col 51) > -0.3 then
            segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 1)  # water,
        elif Band10Max[i] > 0.25:  # if Band10Max (col 42) > 0.25 then
            segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 2)  # rock
        elif NDVIAvg[i] < 0.55:  # if NDVIAvg (col 47) > 0.25 then
            segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 3)  # wetlands
        else:
            segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 4)  # vegetation

# Creating raster of parent class
parent_class = rat.readColumn(segment_rat, "p_class")  # extract parent class ids for each object
parent_class = np.delete(parent_class, 0)  # had to as segment 0 wasnt anything

pclass_ds = np.copy(band)
for seg_id, klass in zip(segment_ids, parent_class):  # dictionary that links segment ids to predicted values
    pclass_ds[pclass_ds == seg_id] = klass
rasteriseOutput(stack_temp_ds, parent_fn, pclass_ds, 'KEA')
print("Parent Class Complete...")
# ------------------------------------------------- #
# rasterising shapefile ground truth data points
# read in training dataset
train_fn = 'C:\\Users\\clare\\MscDiss\\\Malham\\GT\\gt_train.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

# gdal driver to create new dataset
driver = gdal.GetDriverByName('MEM')  # file, raster saved to memory - dont need saved copy
target_ds = driver.Create('', stack_temp_ds.RasterXSize, stack_temp_ds.RasterYSize, 1,
                          gdal.GDT_UInt16)  # name, x,y,band no, format
target_ds.SetGeoTransform(stack_temp_ds.GetGeoTransform())
target_ds.SetProjection(stack_temp_ds.GetProjection())
# TODO: functionise this, incorp with rastOutput func

# rasterise
options = ['ATTRIBUTE=Id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)  # where saving to, band, input vector
# retrieve the rasterised data and print basic stats
data = target_ds.GetRasterBand(1).ReadAsArray()
print('min', data.min(), 'max', data.max(), 'mean', data.mean())
# --------------------------------#
# Assigning land cover types for  ground truth points

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()  # 2d array containg number 0 through 6 for land cover classes
classes = np.unique(ground_truth)[1:]
print('class values', classes)

segments_per_class = {}  # set of sets

##############
for klass in classes:
    # find out which segments belong in each one of classes
    segments_of_class = band[ground_truth == klass]  # which segments correspond to each land cover type
    # print(segments[ground_truth == klass])
    segments_per_class[klass] = set(segments_of_class)
    print('Training segments for class', klass, ':', len(segments_of_class))
    # TODO: Integrate check that there is no overlap between non NDVI PC

# Check if segment appears in two different classes
intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(
        class_segments)  # if anything in accum overlaps with class_segemnts, add to intersection
    accum |= class_segments
    # we want intersection to be 0, showing there are no class intersections
assert len(intersection) == 0, "Segments(s) represent multiple classes"
# ------------------------------------#
# Training classification algorithm

print('Training child classification algorithm...')
train_img = np.copy(band)  # np array copy
threshold = train_img.max() + 1  # greater than max of segments

# loop through classes
for klass in classes:
    # find all the segments associated them, change class to known
    class_label = threshold + klass
    # segment equal to class label so we know its linked to that class
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label  # where train_img equal to segment_id, change to class_label

# alter training image to only show class values
train_img[train_img <= threshold] = 0  # where no training data, equal to zero
train_img[train_img > threshold] -= threshold  # where there is training data, equal to class number

training_labels = []
training_objects = []

for klass in classes:
    # for each object, if  seg_id is in the segments of training data for that class,
    # we get value of that segment for training obj
    class_train_object = [v for i, v in enumerate(obj_stats) if
                          segment_ids[i] in segments_per_class[klass]]  # i = index, v = values of objects
    # new list, that has value of class as many objects as we have
    # if we have 15 segments that represented wat, we would get list of water_id 15 times
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print('Training objects for class', klass, ':', len(class_train_object))
    # train objects should be equal to training segments

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print('Fitting Random Forest Classifier')
predicted = classifier.predict(obj_stats)  # given list of labels for each segments
print('Predicting Classifications')



# Creating child class
parent_ds = gdal.Open(parent_fn, gdal.GA_Update)
parent_arr = parent_ds.GetRasterBand(1).ReadAsArray()
water_mask = parent_arr < 1.5
water_obj = water_mask.astype(int)  # binary array where 1 is water and 0 is not
water_obj = measure.label(water_obj)  # label all separate water objects with dif vals

for i in range(len(np.unique(water_obj))):
    if i == 0:
        pass
    else:
        if (max(np.count_nonzero(water_obj == i, axis=0))/max(np.count_nonzero(water_obj == i, axis=1))) >= 1.5:
            # print("river")
            water_obj[water_obj == i] = 4  # where above is true set water obj to 1
        elif (max(np.count_nonzero(water_obj == i, axis=0))/max(np.count_nonzero(water_obj == i, axis=1))) <= 0.5:
            # print ("river")
            water_obj[water_obj == i] = 4
        elif np.count_nonzero(water_obj == i) >= 800:  # if greater than 8ha (100 pixels):
            print("Lake")
            water_obj[water_obj == i] = 5
        else:
            print("pond")
            water_obj[water_obj == i] = 6

classes = [4, 5, 6]
classes = np.array(classes)
segments_per_class = {}  # set of sets

for klass in classes:
    # find out which segments belong in each one of classes
    segments_of_class = band[water_obj == klass]  # which segments correspond to each land cover type
    # print(segments[ground_truth == klass])
    segments_per_class[klass] = set(segments_of_class)

for klass in classes:
    # find all the segments associated them, change class to known
    # segment equal to class label so we know its linked to that class
    for segment_id in segments_per_class[klass]:
        predicted[segment_id-1] = klass

clf = np.copy(band)
# for each segment we want to list segment Id with reducted values
for segment_id, klass in zip(segment_ids, predicted):  # dictionary that links segment ids to predicted values
    clf[clf == segment_id] = klass

clf = clf.astype(np.float32)

# add in bare rock
rock_mask = parent_arr == 2
np.copyto(clf, np.zeros(clf.shape)+7, where=rock_mask)

print('Prediction applied to numpy array')

# masking ot make sure it comes out properly - bask to show where we have data and where we dont
mask = np.sum(band_data, axis=2)
mask[mask > 0.0] = 1.0  # expect data here
mask[mask == 0.0] = -1.0  # dont expect data
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0

rasteriseOutput(stack_temp_ds, child_fn, clf, 'GTiff')

print("Finished!")


"""
# Create RAT for Objects
stack_rat = gdal.RasterAttributeTable()
for colName in range(len(sentBandNames)):
    # print (sentBandNames[colName])
    stack_rat.CreateColumn(sentBandNames[colName], gdal.GFT_Integer, gdal.GFU_Generic)
    # print(stack_rat.GetNameOfCol(colName))

    # SetValue = row, col, value
    for i in range(rasterShape(stack_ds)[1]): # iterate down rows for number of rows in input stack
        segRAT.SetValueAsInt(row, col, value) # populate first column with segment IDs

#segments_RAT.GetNameOfCol(1)
#segments_RAT.GetColumnCount()
#segments_RAT.GetRowCount()
#segments_RAT.GetValueAsInt(row,col)


## example to associate rat with band
# band = stack_ds.getrasterband(1)
# band = SetDefaultRAT (SegRAT)
# Close the dataset and persist the RAT
# band = ds = None

# GetNameOfCol

## Creating Outputs Files
"""