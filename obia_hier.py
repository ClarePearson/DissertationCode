"""
Created 16/06/2020 by Clare
Based off opensourceoptions obia tutorial
adapted to use rsgislib - shepherds segmentation algorithm

TODO: fix the rescaling issues
TODO: trial different segmentation scales
TODO: hierarchical classification
TODO: workout whats happening with the output image - hole in water
"""

# Import libraries
import rsgislib # image segmentation
from rsgislib.segmentation import segutils
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import time
import os
import gdal # image file format reading, raster att table
import ogr
import numpy as np
import rasterio
from rasterio.plot import show
import rios # simplify raster code processing
import tuiview

# Reading in Files
mt_stack_fn = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\mt_stack.tif'
mt_stack_ds = gdal.Open(mt_stack_fn)  # default read only

# Define File Names
temp_fn = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib' # Temp files
segments_fn_rs = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\mt_segments_rs.tif' # Output Segmentation

# Output drivers
driverTiff = gdal.GetDriverByName('GTiff')

# Count the number of bands in image
nbands = mt_stack_ds.RasterCount
print('bands', mt_stack_ds.RasterCount, 'rows', mt_stack_ds.RasterYSize,
      'columns', mt_stack_ds.RasterXSize)

# Create array of band information
band_data = []
for i in range(1, nbands + 1):
    band = mt_stack_ds.GetRasterBand(i).ReadAsArray()
    # print('band ' + str(i) + ' ' + str(band.max()) + ' ' + str(band.min()))
    #band = exposure.rescale_intensity(band, range_values())
    band_data.append(band)

# calculate band ratios
# NDVI = (NIR – Red)/( NIR + Red)
# S2 = (band7 - band3)/(band7 + band3)
NDVIband = (band_data[6] - band_data[2]) / (band_data[6] + band_data[2])
# print('NDVI band max = ' + str(NDVIband.max()) + ' min = ' + str(NDVIband.min()))
band_data.append(NDVIband)

# NDWI = (Green - NIR)/(Green + NIR)
# S2 = (band 2 - band7/(band 2 + band7)
NDWIband = (band_data[1] - band_data[6]) / (band_data[1] + band_data[6])
# print('NDWI band max = ' + str(NDWIband.max()) + ' min = ' + str(NDWIband.min()))

band_data.append(NDWIband)

# Rescale band data between 0 and 1
# so k-means euclideandistance isnt artificially larger or shorter
#for i in band_data:
    #for j in i:
        #for band in j:
            #print(band) # is each individual


# loop through bands and stack together
band_data = np.dstack(band_data)
# print(band_data.shape)

img = band_data.astype('double')

# -------------------------------------------------#
# segmentation process
#segutils.runShepherdSegmentation(mt_stack_fn, segments_fn_rs, tmpath=temp_fn,
#                             gdalformat='GTiff',
#                             numClusters=60,
#                             minPxls=100,
#                             distThres=100,
#                             sampling=100, kmMaxIter=200)
# -------------------------------------------------#
# image statistics
print('Calculating image statistics...')

rs_segments_ds = gdal.Open(segments_fn_rs)

print('bands', rs_segments_ds.RasterCount, 'rows', rs_segments_ds.RasterYSize,
      'columns', rs_segments_ds.RasterXSize)
rs_segments = rs_segments_ds.ReadAsArray()


def segment_features(segment_pixels):
    #
    # calculate statistics for features
    #
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        statistics = stats.describe(segment_pixels[:, b])
        band_stats = list(statistics.minmax) + list(statistics)[2:]
        if npixels == 1:
            # if the variance is nan, change it to 0
            band_stats[3] = 0.0
        features += band_stats
    return features


obj_start = time.time()
segment_ids = np.unique(rs_segments)
objects = []  # list for all feature statistics
object_ids = []  # link id up with object statistics
for id in segment_ids:
    segment_pixels = img[rs_segments == id]  # where segemnts equals id, get image data
    # print('pixels for id', id, segment_pixels.shape)
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print('Created', len(objects), 'objects with', len(objects[0]), 'variables in', time.time() - obj_start, 'seconds')

# ------------------------------------#
# rasterising shapefile ground truth data points
# read in training dataset
train_fn = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\gt_train.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

# gdal driver to create new dataset
driver = gdal.GetDriverByName('MEM')  # in memory
target_ds = driver.Create('', mt_stack_ds.RasterXSize, mt_stack_ds.RasterYSize, 1,
                          gdal.GDT_UInt16)  # name, x,y,band no, format
target_ds.SetGeoTransform(mt_stack_ds.GetGeoTransform())
target_ds.SetProjection(mt_stack_ds.GetProjection())

# rasterise shapefile, raster saved to memory - dont need saved copy
options = ['ATTRIBUTE=Id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)  # where saving to, band, input vector
# retrieve the rasterized data and print basic stats
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
    segments_of_class = rs_segments[ground_truth == klass]  # which segments correspond to each land cover type
    # print(segments[ground_truth == klass])
    segments_per_class[klass] = set(segments_of_class)
    print('Training segments for class', klass, ':', len(segments_of_class))

# Check if segment appears in two different classes
intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(
        class_segments)  # if anything in accum overlaps with class_segemnts, add to intersection
    accum |= class_segments
    # we want intersection to be 0, showing there are no class intersections
assert len(intersection) == 0, "Segments(s) represent multiple classes"

# ---------------------------------- #
# Parent Class Rule-Based Classifier
print("Classifying Parent Classes...")

def segment_means(segment_pixels):
    # calculate statistics for features
    features = []

    for b in range(nbands):
        seg_mean = segment_pixels[b].mean
        features += seg_mean
    return features

pclf = np.copy(rs_segments)

"""
for id in object_ids:
    segment_pixels = img[pclf == id]  # where segemnts equals id, get image data
    #print('pixels for id', id, segment_pixels.shape) # check iteration
    seg_mean = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        seg_mean.append(segment_pixels[:,b].mean())
    if seg_mean[11]
"""

outimg = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\parentclass_img.tif'
rast = rasterio.open(outimg)
show(rast)  # other bracket comma band layer for multiband ((rast,3))

# ------------------------------------#

# Training classification algorithm
print('Training classification algorithm...')
train_img = np.copy(rs_segments)  # np array copy
threshold = train_img.max() + 1  # greater than max of segments

# loop through classes
for klass in classes:
    # find all the segments associated them, change class to known
    class_label = threshold + klass
    # segment equal to class label so we know its linked to that class
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label  # where train_imd equal to segment_id, change to class_label

# alter training image to onbly show class values
train_img[train_img <= threshold] = 0  # where no training data, equal to zero
train_img[train_img > threshold] -= threshold  # where there is training data, equal to class number

training_labels = []
training_objects = []

for klass in classes:
    # for each object, if  seg_id is in the segments of training data for that class, we'll get the value of that segment for training object
    class_train_object = [v for i, v in enumerate(objects) if
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
predicted = classifier.predict(objects)  # given list of labels for each segments
print('Predicting Classifications')

clf = np.copy(rs_segments)
# for each segment we want to list segment Id with predicted values
for segment_id, klass in zip(segment_ids, predicted):  # dictionary that links segment ids to predicted values
    clf[clf == segment_id] = klass

print('Prediction applied to numpy array')

# masking to make sure it comes out properly - mask to show where we have data and where we dont
mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0  # expect data here
mask[mask == 0.0] = -1.0  # dont expect data
clf = np.multiply(clf, mask) # classified image
clf[clf < 0] = -9999.0

print('Saving classification to raster with gdal')

# ------------------------------#
# Rasterise output
# method only words for 2d arrays

clfds = driverTiff.Create('C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\classified_img.tif',
                          mt_stack_ds.RasterXSize, mt_stack_ds.RasterYSize,
                          1, gdal.GDT_Float32)
clfds.SetGeoTransform(mt_stack_ds.GetGeoTransform())
clfds.SetProjection(mt_stack_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print('Done!')

outimg = 'C:\\Users\\clare\\MscDiss\\Malham\\rsgislib\\classified_img.tif'
rast = rasterio.open(outimg)
show(rast)  # other bracket comma band layer for multiband ((rast,3))


