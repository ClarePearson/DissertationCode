# Dissertation Code

## Dissertation Title
The python scripts in this directory we're writted for my MSc in GIS at the University of Leeds, submitted September 2020. The title of my dissertation was  
*"Assessing hierarchical, object-based image analysis for the classification and dynamic mapping of UK inland Ramsar sites."*

## Abstract
Wetlands are critically important ecosystems that are being severely degraded by human activity. This is partly due to the lack of wetland inventories to support conservation, particularly for inland wetlands, as wetlands are dynamic and heterogeneous and therefore challenging to delineate. Object-based and hierarchical classification methods have shown higher accuracy for wetland land cover classification. Despite this, there have been no UK-based studies assessing the suitability of object-based, hierarchical classification methods to wetland land cover classification and dynamic mapping. This dissertation aimed to develop and evaluate such a method for three UK inland Ramsar wetland sites using open source data and software and producing a dynamic land cover map from the classification outputs. The overall image classification accuracy for the chosen study sites were 66.8%, 81.0% and 86.9%, with the highest errors of omission being for shrub and forest vegetation classes. The dynamic mapping method developed successfully created a map of fuzzy wetland land covers in each study site, highlighting areas of seasonal inundation and vegetation regrowth. The classification would be improved by the development of a segmentation parameter optimisation tool and a greater number and temporal resolution of classified images. More sophisticated pattern recognition algorithms potentially involving error surfaces would improve the dynamic map.

## Data and Software
### Satelite Imagery
Sentinel-2 MSI Level-1C (S2MSI1c) Imagery was used, this can be freely downloaded from https://scihub.copernicus.eu/. This dat aproducts is geometrically and radiometrically corrected and consists of TOA reflectance values. This dissretation used the three visible lighe bands (2-4), four Red Edge bands (5-7 and 8A), NIR band (8) and SWIR bands (11 and 12).

The following table shows the image aquisition dates for each study site used. Images were selected to include a variety of seasons to capture the seasonal variation in wetland vegetation. Images were checked and excluded if there were clouds covering the study area, high contrast shadows or reflectance anomalies.

| Site        |Image Date   |  
|------------ |------------:|  
| Malham Tarn	| 2018/06/29  |   
| 	          | 2019/02/24  |   
|             | 2019/12/01  |   
|             | 2020/04/19  |   
| Bure Marshes| 2015/09/10  |   
| 	          | 2019/12/15  |   
| 	          | 2020/02/06  |   
| 	          | 2020/03/27  |   
| 	          | 2020/04/16  |   
| 	          | 2020/04/26  |   
| 	          | 2020/06/25  |   
| Rutland Water| 	2019/09/19|   
| 	          | 2019/10/02  |   
| 	          | 2019/11/18  |   
| 	          | 2020/03/27  |   
|           	| 2020/04/19  |   
|           	| 2020/05/06  |   
| 	          | 2020/05/29  |   
|           	| 2020/06/25  |   

### RAMSAR site Shapefile
A shapefile delineating all Ramsar sites in England was downloaded from the Natural England – DEFRA Open Data Portal (https://naturalengland-defra.opendata.arcgis.com/). 

### Software and Python Packages
**Software:** Python and QGIS with the Semi-automatic classification plugin (SCP) installed.  
**Packages:** GDAL and OGT, NumPy, RIOS, RSGISLib, TuiView, Scikit-learn, Pandas, GeoPandas, EarthPy and Eo-learn.

## Methods
* Atmospheric Correction using the SCP Dark Object Subtraction algorithm.  
* Band Ratio Calculations: NDVI, NDWI, NDWI (veg) (Green - NIR/Green + NIR), SWIR ratio (SWIR 1 / SWIR 2), RVI (NIR/Red), NIR-Green Ratio(NIR/Green).  
* Image Segmentation   
* 2-step Hierearchical classification.  
  * Step 1: a broader classification using thresholds of mean NDWI (> 0.6 is water), mean SWIR ratio (<0.4 was Bare Ground) and mean RVI (< 0.35 is High productivity veg, > 0.35 is low productivity veg) for each segment.  
  * Step 2: a finer classification, where water classes were reclassified intro river, lakes and ponds based on segment geometry. (if 0.5 >= length:width >- 1.5, river. if area > 8 ha, lake, else pond). Bare groudn remained as bare ground. A random forest was used to determine forest, shrub and grass vegetation types using ground truth points created using Google Earth Pro. Where a vegetation type coincided with the high prodicvitiy vegetation class from step one, it was considered emergent or managed, where coincided with low productivity class it was considered established.  
* Accuracy assessments of image classifications were calculated.  
* All classified images for each study area were stacked into a timeseries. Where different land cover classes were seen over the same pixel in each of the images, a new dynamic class was assigned based on the combination of classes over the timeseries. For example pixels which were classed as water and emergent shrub throughout the timeseries were categorised as seasonally innundated shrublands.  

## Script Descriptions
The scripts in this directory do the following:
1. **preprocessing.py** – reprojects inputted images to British National Grid, creates txt file to be inputted into batch processing QGIS Semi-Automatic Classifictaion plugin (SCP) Dark Object Subtraction (DOS), clips output files of SCP DOS to extent of ramsar shapefile
2. **gt_process.py** – splits ground truthed points into training and test subset
3. **obia.py** – iterates through all the clipped image files in the specified directory and:  
  a. Stacks the Image  
  b. Calculates band ratios  
  c. Creates a dataset of all image layers standardised beteen 0 and 1  
  d. Segments image into objects  
  e. Calculates object statistics  
  f. Creates Broad Image class  
  g. Reads in, rasterises and conduces Random Forest classification using ground truth points Code for this step adapted from https://opensourceoptions.com  
  h. Subclassifies broad image class layer using RF outputs and hierarchical nomenclature.  
4. **accuracy.py** – Creates stratified sample points, reads in and rasterises ground truth points and created a confusion matrix and calculates kappa score.
5. **pattern.py** – stacks classified image files, iterates through each pixel and designates dynamic land cover pattern class based on temporal trends in land cover, createds output dynamic land cover raster.


