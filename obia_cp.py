"""
The following code iterates through all the clipped image files in the specified directory and:
1. Stacks the Image
2. Calculates band ratios
3. Creates a dataset of all image layers standardised beteen 0 and 1
4. Segments image into objects
5. Calculates object statistics
6. Creates Broad Image class
7. Reads in, rasterises and conduces Random Forest classification using ground truth points
    Code for this step adapted from https://opensourceoptions.com
8. Subclassifies broad image class layer using RF outputs and hierarchical nomenclature.
"""

import os
import gdal
import ogr
from rios import rat
import numpy as np
import rsgislib
from rsgislib import imageutils
from rsgislib import rastergis
from rsgislib.rastergis import ratutils
from rsgislib.segmentation import segutils
from sklearn.ensemble import RandomForestClassifier
from skimage import measure

# ------------------------------------------------- #
# Define File Names
# Directories
fil_dir = r'C:\Users\Clare\Documents\MscDiss\Images' #
temp_fn = os.path.join(fil_dir, 'tmp')  # Temp files
site_dir = 'RW' # name of folder for current site

# Input Files
# training shape file
train_fn = r'C:\Users\Clare\Documents\MscDiss\GT\RW_gt_train.shp'


def fileStackBands(file_directory, site_directory, date):
    """
    :param file_directory: String of directory of Images folder
    :param site_directory: String of folder name of site e.g. 'MT'
    :param date: String of date folder name e.g. 20200125
    :return: list of strings of file directories to be fed into image stack
    """
    sentImgList = []
    bandend_nm = ['B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif',
                  'B8A.tif', 'B11.tif', 'B12.tif']
    for image_band in os.listdir(os.path.join(file_directory, '3_Clipped', site_directory, date)):  # imgdate_dir
        # print(image_band)

        for band in bandend_nm:
            if image_band.endswith(band):
                sentImgList.insert(bandend_nm.index(band),
                                   os.path.join(file_directory, '3_Clipped', site_directory, date, image_band))
    return sentImgList


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


def rasteriseOutput(stack_dataset, imgName, img, fn_type):
    """
    creates an output file of the inputted image in whatever file type you put in
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


def npStackRast(band_stack, directory, rast_fn, band_nm, ref_img):
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
    band_names.append('NDWIveg')
    band_names.append('SWIRratio')
    band_names.append('RVI')
    band_names.append('NIRGreen')
    # print(band_names)
    imageutils.stackImageBands(bands, band_names, rast_fn, None, float(-9999.0), 'KEA', rsgislib.TYPE_32FLOAT)

    # remove temp files
    for k in range(len(bands)):
        os.remove(str(directory + 'tempband_' + str(k) + '.kea'))


# Image File stacking - create image stack of sentinel MT images
# Iterate through dates and create temp_stack of them in 4_Stacked file location
print("Creating image stack of clipped sentinel MT images...")

sentBandNames = ['Band02', 'Band03', 'Band04', 'Band05',
                 'Band06', 'Band07', 'Band08', 'Band8A', 'Band11', 'Band12']

for img_date_dir in os.listdir(os.path.join(fil_dir, '3_Clipped', site_dir)):
    # print (img_date_dir)
    imageutils.stackImageBands(fileStackBands(fil_dir, site_dir, img_date_dir), sentBandNames,
                               os.path.join(fil_dir, '4_Stacked', site_dir, img_date_dir + '_stack_temp.kea'),
                               None, 0, 'KEA', rsgislib.TYPE_32FLOAT)

# Iterate through stacked images
for stack_temp_dir in os.listdir(os.path.join(fil_dir, '4_Stacked', site_dir)):
    if stack_temp_dir.endswith('_stack_temp.kea'):
        stack_temp_ds = gdal.Open(os.path.join(fil_dir, '4_Stacked',
                                               site_dir, stack_temp_dir), gdal.GA_Update)  # as gdal array

        # Output Files to be created
        stack_dir_fn = os.path.join(fil_dir, '4_Stacked', site_dir, stack_temp_dir[:-15])
        stack_fn = os.path.join(stack_dir_fn + '_stack.kea')
        stan_rast_fn = os.path.join(stack_dir_fn + '_stack_stan.kea')

        segments_fn = os.path.join(fil_dir, '5_Segmented', site_dir, stack_temp_dir[:-15] + '_segments.kea')
        parent_fn = os.path.join(fil_dir, '6_Classified', site_dir, stack_temp_dir[:-15] + '_parent_c.tiff')
        child_fn = os.path.join(fil_dir, '6_Classified', site_dir, stack_temp_dir[:-15] + '_child_c.tif')

        # read in stacked images to numpy array
        print('Calculating band ratios...')
        band_data = []
        for i in range(1, rasterShape(stack_temp_ds)[0] + 1):
            band = stack_temp_ds.GetRasterBand(i).ReadAsArray()
            band_data.append(band)
        # Calculate band ratios
        NDVIband = (band_data[6] - band_data[2]) / (band_data[6] + band_data[2])  # NDVI = (NIR – Red)/( NIR + Red)
        NDWIband = (band_data[1] - band_data[6]) / (band_data[1] + band_data[6])  # NDWI = (Green - NIR)/(Green + NIR)
        NDWIvegband = (band_data[6] - band_data[8]) / (band_data[6] + band_data[8])  # NDWI_veg ratio = (NIR/SWIR 1)
        SWIRratio = band_data[8] / band_data[9]  # SWIR ratio = (SWIR 1/SWIR 2)
        RVIband = (band_data[6] / band_data[2]) # RVI NIR / Red
        NIRGreen = (band_data[6] / band_data[1])  # Vegetation contrast ratio NIR / Green
        # add to numpy array
        band_data.append(NDVIband)
        band_data.append(NDWIband)
        band_data.append(NDWIvegband)
        band_data.append(SWIRratio)
        band_data.append(RVIband)
        band_data.append(NIRGreen)

        # Rescaling files between 0 and 1
        # Create standardised raster stack for segmentation algorithm
        band_data_stan = []
        for i in range(1, rasterShape(stack_temp_ds)[0] + 1):
            band_s = stack_temp_ds.GetRasterBand(i).ReadAsArray()
            band_s = rescaler(band_s)
            # checknormalisation(band)
            band_data_stan.append(band_s)
        # TODO: automate these lists!
        # band ratio rescale and append to band array
        NDVIband = rescaler(NDVIband)
        band_data_stan.append(NDVIband)
        NDWIband = rescaler(NDWIband)
        band_data_stan.append(NDWIband)
        NDWIvegband = rescaler(NDWIvegband)
        band_data_stan.append(NDWIvegband)
        SWIRratio = rescaler(SWIRratio)
        band_data_stan.append(SWIRratio)
        RVIband = rescaler(RVIband)
        band_data_stan.append(RVIband)
        NIRGreen = rescaler(NIRGreen)
        band_data_stan.append(NIRGreen)

        NDVIband = None
        NDWIband = None
        NDWIvegband = None
        SWIRratio = None
        NIRGreen = None
        RVIband = None
        band = None
        band_s = None

        # ------------------------------------------------- #
        # Creating 2 raster stack files of 1) standardised and
        # 2) non-standardised bands and band ratios
        # convert band array stack to numpy array, third axis is number of bands
        band_data = np.dstack(band_data)
        band_data_stan = np.dstack(band_data_stan)

        # Stack normal, then standardised band data as kea file
        npStackRast(band_data, stack_dir_fn, stack_fn, sentBandNames, stack_temp_ds)
        npStackRast(band_data_stan, stack_dir_fn, stan_rast_fn, sentBandNames, stack_temp_ds)
        print("Normalised raster stack created...")
        # ------------------------------------------------- #
        # Segmentation process
        print("Segmenting image...")
        segutils.runShepherdSegmentation(stan_rast_fn, segments_fn, tmpath=temp_fn,
                                         gdalformat='KEA',
                                         noStretch=True,  # Issue when this is false
                                         noStats=True,  # Issue when this is false
                                         numClusters=60,
                                         minPxls=25,
                                         distThres=10,
                                         sampling=100, kmMaxIter=200)
        # ------------------------------------------------- #
        # Image statistics
        print('Calculating image statistics...')
        # Spectral stats
        rastergis.populateStats(segments_fn)
        ratutils.populateImageStats(stan_rast_fn, segments_fn,
                                    calcMean=True,
                                    calcMin=True,
                                    calcMax=True,
                                    calcStDev=True)  # populates clumps file with stats
        # reading in RAT to add to
        segment_rat = gdal.Open(segments_fn, gdal.GA_Update)  # Read in segments raster as gdal
        segment_ds = segment_rat.GetRasterBand(1).GetDefaultRAT()  # Read in RAT
        band = segment_rat.GetRasterBand(1).ReadAsArray()  # read in segment raster band, with seg ids
        band_data = np.dstack(band)
        segment_ids = np.unique(band)  # extract seg ids

        # ------------------------------------ #
        # create list of object stats, where each row is an object ro1[1] = [1,2,3,4,5]
        obj_stats = []
        for i in range(1, int(segment_ds.GetRowCount())):
            row_stat = []
            for j in range(int(segment_ds.GetColumnCount())):
                stat = segment_ds.GetValueAsDouble(i, j)
                row_stat.append(stat)
            obj_stats.append(row_stat)
        # ------------------------------------ #
        # Geometry
        segment_ds.CreateColumn('obj_area', gdal.GFT_Integer, gdal.GFU_Generic)  # col 58
        segment_ds.CreateColumn('obj_width', gdal.GFT_Integer, gdal.GFU_Generic)  # col 59
        segment_ds.CreateColumn('obj_length', gdal.GFT_Integer, gdal.GFU_Generic)  # col 60
        segment_ds.CreateColumn('obj_lwRatio', gdal.GFT_Integer, gdal.GFU_Generic)  # col 61
        for i in range(1, int(segment_ds.GetRowCount())):  # iterate down rows for number of rows in input stack
            # print('Object ' + str(i) + ' no of pixels = ' + str(np.count_nonzero(band == i)))
            colcount = segment_ds.GetColumnCount()
            segment_ds.SetValueAsInt(i, colcount-4, np.count_nonzero(band == i))  # no of pixels
            segment_ds.SetValueAsInt(i, colcount-3, int(max(np.count_nonzero(band == i, axis=1))))  # max width
            segment_ds.SetValueAsInt(i, colcount-2, int(max(np.count_nonzero(band == i, axis=0))))  # max length/height
            segment_ds.SetValueAsInt(i, colcount-1,
                                     int(max(np.count_nonzero(band == i, axis=0))/max(np.count_nonzero(band == i, axis=1))*100))  # length/width ratio

        # ------------------------------------------------- #
        # Hierarchical Classification
        print("Hierarchical Classification...")

        # Reading in segment RAT
        NDWIAvg = rat.readColumn(segment_rat, "NDWIAvg")
        SWIRratioAvg = rat.readColumn(segment_rat, "SWIRratioAvg")
        RVIAvg = rat.readColumn(segment_rat, "RVIAvg")

        # Creating and population parent class column based on rules
        segment_ds.CreateColumn('p_class', gdal.GFT_Integer, gdal.GFU_Generic)
        for i in range(int(segment_ds.GetRowCount())):  # iterate down rows for number of rows in input stack
            if segment_ds.GetValueAsInt(i, (segment_ds.GetColumnCount()-1)) == 0:  # if no class (0) in p_class column
                if NDWIAvg[i] > 0.6:  
                    segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 1)  # water
                elif SWIRratioAvg[i] < 0.4:
                    segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 2)  # rock
                elif RVIAvg[i] < 0.35:
                    segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 3)  # low productivity veg
                else:
                    segment_ds.SetValueAsInt(i, (segment_ds.GetColumnCount()-1), 4)  # high productivity veg

        # Creating raster of parent class
        parent_class = rat.readColumn(segment_rat, "p_class")  # extract parent class ids for each object
        parent_class = np.delete(parent_class, 0)  # Remove empty segment 0
        pclass_ds = np.copy(band)  # create copy of raster of objects

        for seg_id, klass in zip(segment_ids, parent_class):  # dictionary that links segment ids to parent class
            pclass_ds[pclass_ds == seg_id] = klass # create np array, using dictionary to link segments to p class
        rasteriseOutput(stack_temp_ds, parent_fn, pclass_ds, 'GTiff')

        print("Broad Classification Complete...")
        # ------------------------------------------------- #
        # rasterising shapefile of ground truth data points
        # read in training dataset
        train_ds = ogr.Open(train_fn)
        lyr = train_ds.GetLayer()

        # gdal driver to create new dataset
        driver = gdal.GetDriverByName('MEM')  # file, raster saved to memory - dont need saved copy
        target_ds = driver.Create('', stack_temp_ds.RasterXSize, stack_temp_ds.RasterYSize, 1,
                                  gdal.GDT_UInt16)  # name, x,y,band no, format
        target_ds.SetGeoTransform(stack_temp_ds.GetGeoTransform())
        target_ds.SetProjection(stack_temp_ds.GetProjection())

        # rasterise
        options = ['ATTRIBUTE=Id']
        gdal.RasterizeLayer(target_ds, [1], lyr, options=options)  # where saving to, band, input vector
        data = target_ds.GetRasterBand(1).ReadAsArray()  # retrieve the rasterised data
        # --------------------------------#
        # Assigning land cover types for  ground truth points
        # 2d array containing number 1-3 for land cover classes
        ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
        classes = np.unique(ground_truth)[1:]
        print('class values', classes)

        segments_per_class = {}  # set of sets

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
                class_segments)  # if anything in accum overlaps with class_segments, add to intersection
            accum |= class_segments
            # we want intersection to be 0, showing there are no class intersections
        assert len(intersection) == 0, "Segments(s) represent multiple classes"
        # ------------------------------------#
        # Training classification algorithm
        print('Training RF classification algorithm...')
        train_img = np.copy(band)  # np array copy
        threshold = train_img.max() + 1  # greater than max of segments

        # loop through classes
        for klass in classes:
            # find all the segments associated them, change class to known
            class_label = threshold + klass
            # segment equal to class label so we know its linked to that class
            for segment_id in segments_per_class[klass]:
                # where train_img equal to segment_id, change to class_label
                train_img[train_img == segment_id] = class_label

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
        print('Fitting Random Forest Classifier')
        classifier.fit(training_objects, training_labels)
        print('Predicting Classifications')
        predicted = classifier.predict(obj_stats)  # given list of labels for each segments


        # Creating child class
        print('Creating detailed classification layer...')
        clf = np.copy(band)
        # for each segment we want to list segment Id with reducted values
        for segment_id, klass in zip(segment_ids, predicted):  # dictionary that links segment ids to predicted values
            clf[clf == segment_id] = klass
        clf = clf.astype(np.float32)

        # read in parent class file
        parent_ds = gdal.Open(parent_fn, gdal.GA_Update)
        parent_arr = parent_ds.GetRasterBand(1).ReadAsArray()

        # classify vegetation objects
        if (clf.shape == parent_arr.shape) : pass
        else: print("Broad and specific classess different shapes")

        num_rows, num_cols = parent_arr.shape
        # Pixel Iteration
        for row in range(num_rows):
            for cell in range(num_cols):
                cell_list = []
                cell_list.append(parent_arr[row][cell])
                cell_list.append(clf[row][cell])
                if cell_list[0] == 1:pass  # if water parent class, pass
                elif cell_list[0] == 2: pass  # if bare ground parent class, pass
                elif cell_list[0] == 3:  # if low productivity veg parent class
                    for i in [1, 2, 3]:  # iterate through bog (1), forest (2) and grass (3)
                        if cell_list[1] == i:
                            clf[row][cell] = 7 + i
                elif cell_list[0] == 4: # if low productivity veg parent class
                    for i in [1, 2, 3]:  # iterate through bog (1), forest (2) and grass (3)
                        if cell_list[1] == i:
                            clf[row][cell] = 10 + i

        # classify water objects
        water_mask = parent_arr < 1.5  # create a true false mask where everything 2 or above is false, and 1 is true
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
                    # print("Lake")
                    water_obj[water_obj == i] = 5
                else:
                    # print("pond")
                    water_obj[water_obj == i] = 6

        # add in rivr
        riv_mask = water_obj == 4
        np.copyto(clf, np.zeros(clf.shape)+4, where=riv_mask)

        # add in bare rock
        lake_mask = water_obj == 5
        np.copyto(clf, np.zeros(clf.shape)+5, where=lake_mask)

        # add in bare rock
        pond_mask = water_obj == 6
        np.copyto(clf, np.zeros(clf.shape)+6, where=pond_mask)

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

