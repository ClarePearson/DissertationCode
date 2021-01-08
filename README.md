# Dissertation Code

## Dissertation Title
The python scripts in this directory we're writted for my MSc in GIS at the University of Leeds, submitted September 2020. The title of my dissertation was  
*"Assessing hierarchical, object-based image analysis for the classification and dynamic mapping of UK inland Ramsar sites."*

## Abstract
Wetlands are critically important ecosystems that are being severely degraded by human activity. This is partly due to the lack of wetland inventories to support conservation, particularly for inland wetlands, as wetlands are dynamic and heterogeneous and therefore challenging to delineate. Object-based and hierarchical classification methods have shown higher accuracy for wetland land cover classification. Despite this, there have been no UK-based studies assessing the suitability of object-based, hierarchical classification methods to wetland land cover classification and dynamic mapping. This study aimed to develop and evaluate such a method for three UK inland Ramsar wetland sites using open source data and software, using the classification outputs for a dynamic land cover map. The overall image classification accuracy for the chosen study sites were 66.8%, 81.0% and 86.9%, with the highest errors of omission for shrub and forest vegetation classes. The dynamic mapping method developed successfully created a map of fuzzy wetland land covers in each study site, highlighting areas of seasonal inundation and vegetation regrowth. The classification would be improved by the development of a segmentation parameter optimisation tool and a greater number and temporal resolution of classified images, alongside more sophisticated pattern recognition algorithms would improve the dynamic map.

## Data Description
Sentinel-2A MSI imagery (https://scihub.copernicus.eu/)  

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


