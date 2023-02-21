# -*- coding: utf-8 -*-
"""
Created on Thu June 14 15:31:05 2022

@author: Manuel A. Luck
         luck@ifu.baug.ethz.ch , manuel.luck@gmail.com
         Ph.D. Candidate in Environmental Engineering, ETH Zurich 
"""

# %% Import:
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap as lsc
from matplotlib.colors import Normalize as Normalize

import numpy as np
import pandas as pd

from datetime import datetime

import os
import json
import joblib

from sklearn.ensemble import RandomForestClassifier

from osgeo import gdal, osr, gdalconst

# %% Google API Initialization:
import ee   
# Trigger the authentication flow. 
# ee.Authenticate() # probably necessary to do with conda console -> activate ee -> import ee -> ee.Authenticate -> follow online instructions

# Initialize the library.
ee.Initialize()

# %% Set Parameters:
# Should there be any figures:    
prints          = False # general result images
allResults      = False # after running for different time periods


# Determine which subset of the image should be calculated:
subset          = True # one subset consists of approximately 511x511 pixels due to EE download function
if subset:
    xMax    = 8 # 9
    yMax    = 4 # 5
    xMin    = 7 # 7
    yMin    = 3 # 3
testCountLimit  = np.inf    

# Choose if the method should be applied to subsets with any() slope values >= minAngle:
excludeFlatland = True
if excludeFlatland:
    minAngle = 10

# Decide whether to calculate the graph or just get all the data together
calculateGraph  = True

# %% File Locations:    
AscFiles        = ['D:/Documents/Processing/Weheka/AscIW2_before.tif','D:/Documents/Processing/Weheka/AscIW2_large.tif'] # Asc image pair

DescFiles       = ['D:/Documents/Processing/Weheka/DescIW1_before.tif','D:/Documents/Processing/Weheka/DescIW1_large.tif'] # Desc image pair

TrainingFiles   = ['D:/Documents/Processing/The_Alpine_Gardens_landslide/DInSAR/Coregistered/GT_3resampled.tif'] # Training data (if not RFC is already trained, just as placeholder)
    
GTFiles         = ['D:/Documents/Processing/The_Alpine_Gardens_landslide/DInSAR/Coregistered/GT_3resampled.tif'] # Ground truth data

# %% Naming and Time Period:
runningExtent   = 'large_RFC21'
landCoverYear   = 2017
dateSurvey      = ee.Date('2018-11-17')
dateSurveyYb    = dateSurvey.advance(-12,'month')
dateBefore      = dateSurvey.advance(6,'month')
dateAfter       = dateSurvey.advance(6,'month')

# Different Variations:
# runningExtent   = '16_17_RFC21'                     # 'small'                           # 'large'                           # 'small'                           #'16_17'
# landCoverYear   = 2016                              # 2017                              # 2017                              # 2017                              # 2016
# dateSurvey      = ee.Date('2017-02-06')             # ee.Date('2018-02-06')             # ee.Date('2018-11-17')             # ee.Date('2018-02-06')             # ee.Date('2017-02-06')
# dateSurveyYb    = dateSurvey.advance(-12,'month')   # dateSurvey.advance(-12,'month')   # dateSurvey.advance(-12,'month')   # dateSurvey.advance(-12,'month')   # dateSurvey.advance(-12,'month')
# dateBefore      = dateSurvey.advance(4,'month')     # dateSurvey.advance(4,'month')     # dateSurveyYb.advance(6,'month')   # dateSurvey.advance(4,'month')     # dateSurvey.advance(4,'month')
# dateAfter       = dateSurvey.advance(4,'month')     # dateSurvey.advance(1,'month')     # dateSurvey.advance(6,'month')     # dateSurvey.advance(1,'month')     # dateSurvey.advance(4,'month')


# World Cover Colormap
worldCoverClasses = [[10,20,30,40,50,60,70,80,90,95,100],
                     ['Trees','Shrubland','Grassland','Cropland','Build-up',
                      'Baren','SnowIce','OpenWater','HerbaceousWetland',
                      'Mongroves','MossLichen'],
                     ['#006400','#006400','#ffbb22','#ffbb22','#ffff4c','#ffff4c',
                      '#f096ff','#f096ff','#fa0000','#fa0000','#b4b4b4','#b4b4b4',
                      '#f0f0f0','#f0f0f0','#0064c8','#0064c8','#0096a0',
                      '#00cf75','#fae6a0']]
worldCoverCmap  = lsc.from_list('worldCoverCmap', worldCoverClasses[2])

# %% Global Functions:
def same_len_print(string,maxLength=70):
    '''
    Function to print a string in the python console with a specific max length 
    adding - before and after the string to keep it centered.

    Parameters
    ----------
    string : string
        Text to be printed.
    maxLength : int, optional
        length of one line. The default is 70.

    Returns
    -------
    None.

    '''
    strL        = len(string)
    if np.mod(strL,2) == 0:
        string = ' '+string+' '
    else:
        string = ' '+string+' -'
        
    if strL == 0:
        print('-'*maxLength)
    elif strL+2 <= maxLength:
        linesCount  = int(np.floor((maxLength - strL)/2)-1)
        lines       = '-'*linesCount
        printStr    = lines+string+lines
        
        print(printStr)
    else:
        print(string)
                             
# %% EE Functions
def getTerrainCoords(aoi,proj,stringDEM='CGIAR/SRTM90_V4'):
    """
    Function to "download" a rectangle of a chosen DEM from the google earth engine
    according to a given ee.geometry and ee.projection object.
    Adding slope, apect, latitute, and longitude.
    
    XY pixels coordinates to align to ee.projection file

    Parameters
    ----------
    aoi : ee.geometry
        Area of interest.
    proj : ee.projection
        Projection of image.
    stringDEM : string, optional
        Data Catalog Name. The default is 'CGIAR/SRTM90_V4'.

    Returns
    -------
    terrain : Array of float64
        [:,:,0] = Elevation, 
        [:,:,1] = Slope, 
        [:,:,2] = Aspect.
    coords : Array of float64
        [:,:,0] = Longitude (X), 
        [:,:,1] = Latitude (Y), 
        [:,:,2] = Elevation.
    xy : Array of float64
        [:,:,0] : Pixels X Coordinates
        [:,:,1] : Pixels Y Coordinates

    """
    elevation   = ee.Image(stringDEM).select('elevation').resample('bicubic').reproject(proj).sampleRectangle(aoi)
    elevation   = elevation.get('elevation')
    elevation   = np.array(elevation.getInfo())
    elevation   = np.expand_dims(elevation, 2)
        
    slope       = ee.Terrain.slope(ee.Image(stringDEM).select('elevation')).resample('bicubic').reproject(proj).sampleRectangle(aoi)
    slope       = slope.get('slope')
    slope       = np.array(slope.getInfo())
    slope       = np.expand_dims(slope, 2)
        
    aspect      = ee.Terrain.aspect(ee.Image(stringDEM).select('elevation')).resample('bicubic').reproject(proj).sampleRectangle(aoi)
    aspect      = aspect.get('aspect')
    aspect      = np.array(aspect.getInfo())
    aspect      = np.expand_dims(aspect, 2)
    
    coords      = ee.Image(stringDEM).pixelCoordinates(proj)
    xi          = coords.select('x').reproject(proj).sampleRectangle(aoi)
    yi          = coords.select('y').reproject(proj).sampleRectangle(aoi)
    
    x           = np.expand_dims(np.array(xi.get('x').getInfo()),2)
    y           = np.expand_dims(np.array(yi.get('y').getInfo()),2)
    xy          = np.dstack((x,y)) 
    
    transform   = proj.getInfo()
    
    [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation] = transform['transform']

    
    x = x * xScale + xTranslation
    y = y * yScale + yTranslation

    

    terrain     = np.dstack((elevation,slope,aspect))
    coords      = np.dstack((x,y,elevation))

    return terrain, coords, xy

def getWorldCover(aoi,proj=None,stringWC="ESA/WorldCover/v100"):
    """
    Function to "download" a rectangle of a chosen landcover map from the google earth engine
    according to a given ee.geometry and ee.projection object.
    
    XY pixels coordinates to align to ee.projection file    

    Parameters
    ----------
    aoi : ee.geometry
        Area of interest.
    proj : ee.projection
        Projection of image.
    stringWC : TYPE, optional
        DESCRIPTION. The default is "ESA/WorldCover/v100".

    Returns
    -------
    wC : np array
        Landcover information according to the respective table.
        
    xy : Array of float64
        [:,:,0] : Pixels X Coordinates
        [:,:,1] : Pixels Y Coordinates
        
    """
    if type(proj) == ee.Projection:
        wC   = ee.ImageCollection(stringWC).first().reproject(proj).sampleRectangle(aoi)
    else:
        wC   = ee.ImageCollection(stringWC).first().sampleRectangle(aoi) 
        proj = wC.projection()
        
    wC   = wC.get('Map')
    wC   = np.array(wC.getInfo())
    wC   = np.expand_dims(wC, 2)

    coords      = ee.Image(stringWC).pixelCoordinates(proj)
    xi          = coords.select('x').reproject(proj).sampleRectangle(aoi)
    yi          = coords.select('y').reproject(proj).sampleRectangle(aoi)
    
    x           = np.expand_dims(np.array(xi.get('x').getInfo()),2)
    y           = np.expand_dims(np.array(yi.get('y').getInfo()),2)
    xy          = np.dstack((x,y))    

    return wC, xy

def getCopernicusLandcover(aoi,proj=None,year="2018"):
    """
    Function to "download" a rectangle of a chosen landcover map from the google earth engine
    according to a given ee.geometry and ee.projection object.
    
    XY pixels coordinates to align to ee.projection file    

    Parameters
    ----------
    aoi : ee.geometry
        Area of interest.
    proj : ee.projection, optional
        Projection of image.
    year: string, optional
        DESCRIPTION. The default is "2018".

    Returns
    -------
    wC : np array
        Landcover information according to the simplified table.
        
    xy : Array of float64
        [:,:,0] : Pixels X Coordinates
        [:,:,1] : Pixels Y Coordinates
    """

    if type(year) != str:
        year = str(year)
    stringLC = "COPERNICUS/Landcover/100m/Proba-V-C3/Global/"+year
    
    if type(proj) == ee.Projection:
        lC   = ee.Image(stringLC).select('discrete_classification').reproject(proj).sampleRectangle(aoi)
    else:
        lC   = ee.Image(stringLC).select('discrete_classification').sampleRectangle(aoi) 
        proj = lC.projection()
        
    lC   = lC.get('discrete_classification')
    lC   = np.array(lC.getInfo())
    lC   = np.expand_dims(lC, 2)
    
    wC   = lC
    
    # simplify various forest types
    wC[wC == 200]   = 80
    wC[wC >  100]   = 10

    coords      = ee.Image(stringLC).pixelCoordinates(proj)
    xi          = coords.select('x').reproject(proj).sampleRectangle(aoi)
    yi          = coords.select('y').reproject(proj).sampleRectangle(aoi)
    
    x           = np.expand_dims(np.array(xi.get('x').getInfo()),2)
    y           = np.expand_dims(np.array(yi.get('y').getInfo()),2)
    xy          = np.dstack((x,y))    

    return wC, xy   

def getSentinel2(aoi,startDate=None,endDate=None,proj=None,
                 add_NDVI=True,add_NDWI=False,SR=False,orbit=None,maskClouds='light',
                 bands=['B2','B3','B4','B8','NDVI']):
    """
    

    Parameters
    ----------
    aoi : ee.geometry
        Area of interest.
    proj : ee.projection, optional
        Projection of image.
    startDate : ee.Date, optional
        Start of time period. The default is None.
    endDate : ee.Date, optional
        End of time period. The default is None.
    add_NDVI : bool, optional
        True to add ndvi to each image. The default is True.
    add_NDWI : bool, optional
        True to add ndWi to each image. The default is False.
    SR : bool, optional
        True to use sentinel 2 level 2 date, else levlel 1. The default is False.
    orbit : string, optional
        'Ascending' or 'Descending' to select orbits. The default is None.
    maskClouds : string, optional
        'light' for simple cloud mask based on 'QA' band. The default is 'light'.
        Other options not implemented yet.
    bands : list, optional
        Subset bands which are used. The default is ['B2','B3','B4','B8','NDVI'].

    Returns
    -------
    s2MedianArray : np.array
        Median image for the chosen time period with chosen bands and indices along the 3rd axis.
        
    xy : Array of float64
        [:,:,0] : Pixels X Coordinates
        [:,:,1] : Pixels Y Coordinates        
    

    """
    def addNDVI(img):
        """
        adding NDVI index to S2 images using bands 8 and 4.

        Parameters
        ----------
        img : ee.image

        Returns
        -------
        img : ee.image
            image with added indices.

        """
        img = img.addBands(img.normalizedDifference(ee.List(['B8','B4'])).rename('NDVI'))
        return img

    def addNDWI(img):
        """
        adding NDWI index to S2 images using bands 8 and 11.

        Parameters
        ----------
        img : ee.image

        Returns
        -------
        img : ee.image
            image with added indices.

        """
        img = img.addBands(img.normalizedDifference(ee.List(['B8','B11'])).rename('NDWI'))
        return img     

    def s2cloudMaskLight(img):
        """
        Cloudmasking using the sentinel 2 QA band

        Parameters
        ----------
        img : ee.image
        
        Returns
        -------
        img : ee.image
            image with updated cloud mask.

        """
        cloudBand       = img.select('QA60')
        cloudMask       = cloudBand.bitwiseAnd(1<<10).eq(0)
        cirrusMask      = cloudBand.bitwiseAnd(1<<11).eq(0)
        
        mask            = cloudMask.multiply(cirrusMask)
        
        return img.select('.*').updateMask(mask)
    
    # Switching between level 1 and 2 products:
    if SR:
        collString = 'COPERNICUS/S2_SR'
    else:
        collString = 'COPERNICUS/S2'
        
    # Defining start and end-dates
    if endDate == 'now':
        endDate = ee.Date(datetime.today().strftime('%Y-%m-%d'))


    if maskClouds == 'light': 
        s2Col = (ee.ImageCollection(collString)
                   .filterBounds(aoi)
                   .filterDate(startDate, endDate)
                   .map(s2cloudMaskLight))
    else:
        s2Col = (ee.ImageCollection(collString)
                   .filterBounds(aoi)
                   .filterDate(startDate, endDate))
    if orbit != None:
        s2Col = s2Col.filterMetadata('SENSING_ORBIT_DIRECTION', 'equals', orbit)
        
    if add_NDVI:
        s2Col   = s2Col.map(addNDVI)
    if add_NDWI:
        s2Col   = s2Col.map(addNDWI)

    s2Col       = s2Col.sort('system:time_start')
    s2ColMedian = s2Col.median()
    
    for band in bands:
        b    = s2ColMedian.select(band).reproject(proj).sampleRectangle(aoi)
        b    = b.get(band)
        b    = np.array(b.getInfo())
        b    = np.expand_dims(b, 2)
    
        if band == bands[0]:
            s2MedianArray = b
        else:
            s2MedianArray = np.dstack((s2MedianArray,b))

    coords      = s2ColMedian.pixelCoordinates(proj)
    xi          = coords.select('x').reproject(proj).sampleRectangle(aoi)
    yi          = coords.select('y').reproject(proj).sampleRectangle(aoi)
    
    x           = np.expand_dims(np.array(xi.get('x').getInfo()),2)
    y           = np.expand_dims(np.array(yi.get('y').getInfo()),2)
    xy          = np.dstack((x,y))    

    return s2MedianArray, xy


def GEOjson2Coords(string,moveX=0,moveY=0):
    """
        input:  string (path of .txt file) 

        output: aoi(ee.Geometry.Polygon)
                
        converts GEOjson .txt to dictionary
    """
    with open(string) as f:
        lines = f.readlines()
    f.close()
    
    numbers = []
    for line in lines:
        try:
            numbers.append(float(line.replace(',', '')))
        except:
            pass
    
    xDifference = numbers[0]-numbers[2]
    yDifference = numbers[1]-numbers[5]
    
    newNumbers = [0]*len(numbers)
    for i in range(len(numbers)):
        if i%2 == 0:
            newNumbers[i] = numbers[i]-xDifference*moveX
        elif i%2 != 0:
            newNumbers[i] = numbers[i]-yDifference*moveY
        else:
            newNumbers[i] = numbers[i]
    
    contents=''
    for l in range(len(lines)):
        test = True
        n = 0
        while test and n < len(numbers):
            if str(numbers[n]) in lines[l]:
                lines[l] = lines[l].replace(str(numbers[n]),str(newNumbers[n]))
                test = False
            else:
                n += 1
                
        contents = contents + '\n' + lines[l]

    AOI = json.loads(contents)
    aoi = ee.Geometry.Polygon(AOI['features'][0]['geometry']['coordinates'])
    return aoi

# %% Graph Class:
class graphNet():
    # ------------ ------------ Base Functions ------------ ------------ 
    def __init__(self,data,edgeCreation='byEle'):
        """
        

        Parameters
        ----------
        data : dict
            data is a dict with all the input data stored seperately. 
            data['coords'] has Lat in [:,:,0], Lon in [:,:,1], and Elevation in
            [:,:,2]. data['S1Asc'] consists of the ascending products data['S1Desc']
            the descending, and data['S2'] the optical data.
            data['S1Bands'] and ['S2Bands'] stores the bandnames of S1 and S2 respectively.
            data['Training'] are the training data.
            data['GroundTruth'] are the groundtruth data.
            data['Landcover'] are the surface classes.
            data['Terrain'] has slope in [:,:,1] and aspect in [:,:,2]. 
        edgeCreation : string, optional
            Possible edge creation options are: 'byEle', 'single', and 'double'.     
            byEle uses the elevation from the neighbouring nodes to deterimine.
            the steepest slope. 'single' and 'double' both use the aspect of the
            ee.terrain. While 'single' uses the node closest to the aspect angle,
            'double' integrates both nodes if the angle is between two. 
            The default is 'byEle'.

        Returns
        -------
        Initial graph with nodes and edges/edgelist.

        """        
        
        Lat         = data['Coords'][:,:,0]
        Lon         = data['Coords'][:,:,1]
        DEM         = data['Coords'][:,:,2]
    
        S1Asc       = data['S1Asc']
        S1Desc      = data['S1Desc']
        S2          = data['S2']
        
        self.bandNamesAsc   = [s + 'Asc' for s in data['S1Bands']]
        self.bandNamesDesc  = [s + 'Desc' for s in data['S1Bands']]
        self.bandNamesOpt   = data['S2Bands']
        self.bandNames      = self.bandNamesAsc+self.bandNamesDesc+self.bandNamesOpt
        
        [self.rows, self.cols]      = np.shape(DEM)        
        
        self.edgelist               = []
        self.nodes                  = []

        self.uBandNames             = []
        self.dBandNames             = []        
        
        node                        = 0
            
        for i in range(self.rows):
            for j in range(self.cols):
                self.nodes.append(dict())
                # setting up node
                self.nodes[node]['imageCoords']     = [i,j]
                self.nodes[node]['id']              = node

                # adding topography data to node
                self.nodes[node]['terrain']         = int(data['Landcover'][i,j])
                self.nodes[node]['slope']           = data['Terrain'][i,j,1]
                self.nodes[node]['aspect']          = data['Terrain'][i,j,2]                
                self.nodes[node]['lat']             = Lat[i,j]
                self.nodes[node]['lon']             = Lon[i,j]                
                self.nodes[node]['coords']          = np.hstack([Lat[i,j],Lon[i,j],DEM[i,j]])
                
                # prepare Random ML structure
                self.nodes[node]['GroundTruth']     = data['GroundTruth'][i,j,0]
                self.nodes[node]['predicted Label'] = -1
                self.nodes[node]['training']        = data['Training'][i,j,0]     
                
                # adding remote sensing data to node
                if len(np.shape(S2))>2:
                    for b,bandName in enumerate(self.bandNamesOpt):
                        self.nodes[node][bandName] = S2[i,j,b]
                else:
                    self.nodes[node][self.bandNamesOpt[0]] = S2[i,j]    
                
                if len(np.shape(S1Asc))>2:
                    for b,bandName in enumerate(self.bandNamesAsc):
                        self.nodes[node][bandName] = S1Asc[i,j,b]
                else:
                    self.nodes[node][self.bandNamesAsc[0]] = S1Asc[i,j]                 

                if len(np.shape(S1Desc))>2:
                    for b,bandName in enumerate(self.bandNamesDesc):
                        self.nodes[node][bandName] = S1Asc[i,j,b]
                else:
                    self.nodes[node][self.bandNamesDesc[0]] = S1Desc[i,j]   
                
                # adding masks if available
                if 'MaskA' in data.keys():
                    self.nodes[node]['MaskA']           = data['MaskA'][i,j]
                else:
                    self.nodes[node]['MaskA']           = 0
                    
                if 'MaskD' in data.keys():
                    self.nodes[node]['MaskD']           = data['MaskD'][i,j]
                else:
                    self.nodes[node]['MaskD']           = 0
                    
                if 'MaskO' in data.keys():
                    self.nodes[node]['MaskO']           = data['MaskO'][i,j]
                else:
                    self.nodes[node]['MaskO']           = 0
                    
                # adding additional node structure for later use
                self.nodes[node]['edges']           = []
                self.nodes[node]['upstream']        = []
                self.nodes[node]['downstream']      = []
                self.nodes[node]['neighbours']      = []
                self.nodes[node]['cluster']         = 0
                self.nodes[node]['slopeRatio']      = []

                # adding neighbours id based on a raster image format geometry
                if j < self.cols-1: # all nodes beside the fartest right column
                    self.nodes[node]['neighbours'].append(node+1)
                if j > 0:           # all nodes beside the fartest left column
                    self.nodes[node]['neighbours'].append(node-1)
                
                if i < self.rows-1: # all nodes beside the bottom row
                    self.nodes[node]['neighbours'].append(node+self.cols)
                    if j < self.cols-1: # and the fartest right column
                        self.nodes[node]['neighbours'].append(node+1+self.cols)
                    if j > 0:           # and the fartest left column
                        self.nodes[node]['neighbours'].append(node-1+self.cols)
                if i > 0:           # all nodes beside the top row
                    self.nodes[node]['neighbours'].append(node-self.cols)
                    if j < self.cols-1: # and the fartest right column
                        self.nodes[node]['neighbours'].append(node+1-self.cols)
                    if j > 0:           # and the fartest left column
                        self.nodes[node]['neighbours'].append(node-1-self.cols) 
                                             
                # Create flow edges based on Aspect:
                if edgeCreation == 'double':    # Allow double Edges:
                    if self.nodes[node]['aspect'] != np.nan:
                        if self.nodes[node]['coords'][2] > 0:
                            
                            aspect = self.nodes[node]['aspect']                                                                    
     
                            if aspect == 0 or aspect == 360:
                                nodeId = node-self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect > 0 and aspect < 45:
                                nodeId = node-self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                            
                                nodeId = node-self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect == 45:
                                nodeId = node-self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])                            
    
                            elif aspect > 45 and aspect < 90:
                                nodeId = node-self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])  
                            
                                nodeId = node+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect == 90:
                                nodeId = node+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])                            
                                    
                            elif aspect > 90 and aspect < 135:
                                nodeId = node+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                                nodeId = node+self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect == 135:
                                nodeId = node+self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])                            
                            
                            elif aspect > 135 and aspect < 180: 
                                nodeId = node+self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                            
                                nodeId = node+self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect == 180:
                                nodeId = node+self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect > 180 and aspect < 225: 
                                nodeId = node+self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                            
                                nodeId = node+self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect == 225:
                                nodeId = node+self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect > 225 and aspect < 270: 
                                nodeId = node+self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId]) 
                            
                                nodeId = node-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])  
                                    
                            elif aspect == 270:
                                nodeId = node-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                
                            elif aspect > 270 and aspect < 315: 
                                nodeId = node-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                            
                                nodeId = node-self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect == 315:
                                nodeId = node-self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])                            
                                
                            elif aspect > 315 and aspect < 360: 
                                nodeId = node-self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])  
                            
                                nodeId = node-self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])  
                                    
                if edgeCreation == 'single':    # only single edges:
                    if self.nodes[node]['aspect'] != np.nan:
                        if self.nodes[node]['coords'][2] > 0:
                            
                            aspect = self.nodes[node]['aspect']                                                                    
     
                            if aspect <= 22.5 or aspect > 360-22.5:
                                nodeId = node-self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                    
                            elif aspect > 22.5+0*45 and aspect <= 22.5+1*45:
                                nodeId = node-self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])

                            elif aspect > 22.5+1*45 and aspect <= 22.5+2*45:
                                nodeId = node+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                        
                            elif aspect > 22.5+2*45 and aspect <= 22.5+3*45:
                                nodeId = node+self.cols+1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])

                            elif aspect > 22.5+3*45 and aspect <= 22.5+4*45:
                                nodeId = node+self.cols
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])

                            elif aspect > 22.5+4*45 and aspect <= 22.5+5*45:
                                nodeId = node+self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])

                            elif aspect > 22.5+5*45 and aspect <= 22.5+6*45:
                                nodeId = node-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])

                            elif aspect > 22.5+6*45 and aspect <= 22.5+7*45:
                                nodeId = node-self.cols-1
                                if nodeId in self.nodes[node]['neighbours']:
                                    self.edgelist.append([node,nodeId])
                                          
                node += 1
                
        # calculating the edges based on the elevation of the neighbouring pixels
        # this needs an additional loop as not all nodes are initialized.
        # Therefore more time consuming!
        if edgeCreation == 'byEle':
            for node in self.nodes:
                nEle        = [(self.nodes[idx]['coords'][2]-node['coords'][2])/(
                               ((self.nodes[idx]['imageCoords'][0]-node['imageCoords'][0])**2+
                                (self.nodes[idx]['imageCoords'][1]-node['imageCoords'][1])**2)**(1/2)) 
                               for idx in node['neighbours']]
                
                if np.nanmin(nEle) <= 0:
                    self.nodes[node['neighbours'][np.nanargmin(nEle)]]['edges'].append(node['id'])
                    self.edgelist.append([node['id'],self.nodes[node['neighbours'][np.nanargmin(nEle)]]['id']])
                    node['edgeSlope']       = abs(np.degrees(np.arctan(np.nanmin(nEle))))
                    node['edgeRatio']      = np.nanmin(nEle)                
                
                

                
    # -------------------       
    def applyMask(self,dataType=None):
        '''
        Parameters
        ----------
        dataType : TYPE, optional
            Options are None,'Asc','Desc', and 'Opt'
            None applies all Masks, and 'Asc', 'Desc' only on the respectiv data. 
            The default is None.

        Returns
        -------
            data of masked nodes are replaced with np.nan.

        '''
        for node in self.nodes:
            if dataType == None:
                if node['MaskA'] > 0:
                    for b in self.bandNamesAsc:
                        node[b] = np.nan
                if node['MaskD'] > 0:
                    for b in self.bandNamesDesc:
                        node[b] = np.nan                    
                if node['MaskO'] > 0:
                    for b in self.bandNamesOpt:
                        node[b] = np.nan
            else:
                if 'Asc' in dataType:
                    if node['MaskA'] > 0:
                        for b in self.bandNamesAsc:
                            node[b] = np.nan                    
                if 'Desc' in dataType:
                    if node['MaskD'] > 0:
                        for b in self.bandNamesDesc:
                            node[b] = np.nan  
                if 'Opt' in dataType:
                    if node['MaskO'] > 0:
                        for b in self.bandNamesOpt:
                            node[b] = np.nan 
                            
    # -------------------
    def add_stream_direction(self):
        '''
        Adds 'upstream' and 'downstream' neighbours to each nodes based on 
        the edgelist which is generated during the initialization of the graph.

        '''
        edgelist = self.edgelist
        el       = np.vstack(edgelist)

        for i in range(len(edgelist)): 
            # depending on the position in the edgelist, the edge is either up 
            # or down to according to the flow direction:
            self.nodes[el[i,0]]['downstream'].append(el[i,1])
            self.nodes[el[i,1]]['upstream'].append(el[i,0])
            
    # -------------------
    def get_endNodes(self,direction='downstream'):
        '''
        Finds nodes which have no connections to either down or upstream nodes.
        These nodes are important during the connected cluster creation process.

        Parameters
        ----------
        direction : string, optional
            either 'upstream' or 'downstream'. The default is 'downstream'.

        Returns
        -------
        None.

        '''
        self.endNodes = []
        for node in self.nodes:
            if len(node[direction]) < 1:
                self.endNodes.append(node['id'])
                
    # -------------------
    def connected_clusters(self,direction='upstream'):
        '''
        Assigning cluster id to each node. Clusters are sets of nodes which are 
        connected to each other along a given flow direction. 

        Parameters
        ----------
        direction : string, optional
            Either 'upstream' or 'downstream'. The default is 'upstream'.

        Returns
        -------
             cluster id added to each node.

        '''
        self.get_endNodes(direction=direction)
        
        uNodes_dict = dict()
        for clusterNr in range(len(self.endNodes)):
            uNodes_new      = self.nodes[self.endNodes[clusterNr]][direction]
            uNodes_total    = []
            uNodes_total.append(self.nodes[self.endNodes[clusterNr]]['id'])
            uNodes_total.extend(uNodes_new)
            while len(uNodes_new) >= 1:
                uNodes = []
                for i in range(len(uNodes_new)):
                    
                    uNodes.extend(self.nodes[uNodes_new[i]][direction])
                uNodes_total.extend(uNodes)
                uNodes_new = uNodes
            uNodes_dict[clusterNr] = uNodes_total

        counter = 0
        for clu in range(len(uNodes_dict)):
            counter += 1
            for idx in range(len(uNodes_dict[clu])):
                self.nodes[uNodes_dict[clu][idx]]['cluster'] = counter
                

    # -------------------
    def dictIdxKeySubset(self,keyList=None,nodesList=None):
        '''
        Function to retrieve values from the graph in a dict object. keylist defines the 
        keys which are selected in each node and nodeslist the nodes id/position.
        This can be fu
        
        Parameters
        ----------
        keyList : list or None, optional
            All keys which are selected. The default is None.
            If None, all information of the node is selected.
            Best used with lists such as bandNames.
        nodesList : list or None, optional
            List of all selected nodes. Used to create a subset for specific 
            nodes. The default is None. If None is selected, all nodes are used.
            Best used with lists such as the 'allUp' or 'allDown' list.

        Returns
        -------
        subdict : dict
            subdict is a dictionnary with of the selected nodes with the selected keys.
            best used in combination with pd.DataFrame.from_dict(subdict,orient='index')
            to create a pd dataframe. Or add .to_numpy() to get a numpy array.

        '''
        def dictKeySubset(node,keyList):
           return {key: node[key] for key in keyList}
       
        if type(nodesList) == int:
            nodesList = [nodesList]
            
        subDict = dict()
        if type(nodesList) == type(None):
            for idx,node in enumerate(self.nodes):
                if type(keyList) == list:
                    subDict[idx] = dictKeySubset(node,keyList)
                elif keyList == None:
                    subDict = self.nodes
        
        elif type(nodesList) == list or type(nodesList) == type(np.array(())):
            if type(keyList) == list:
                for idx,node in enumerate(nodesList):
                    subDict[idx] = dictKeySubset(self.nodes[node],keyList)
            elif keyList == None:
                for idx,node in enumerate(nodesList):
                    subDict[idx] = self.nodes[node]
               
        return subDict
                
    # -------------------
    def kernelNeighbourStats(self,bandNames=None,steps=1):
        '''
        running np.nanmean and np.nanstd for all neighbours and bands within a 
        given square window. 

        Parameters
        ----------
        bandNames : list or None, optional
            Bandnames to calculate stats of, if None, all bandNames are used. 
            The default is None.
        steps : int, optional
            Window size. 1=3x3 window, 2=5x5 etc. 
            The default is 1.

        Returns
        -------
             Stores np.nanmean and np.nanstd for selected band in each nodes and 
             saves the respective bandnames under self.neighboursStatsBandNames.

        '''
        if bandNames == None:
            keys = self.bandNames
        else:
            keys = bandNames        
        
        self.neighboursStatsBandNames     = ['kernelMean'+str(steps)+band for band in keys]+['kernelStd'+str(steps)+band for band in keys]
        
        for node in self.nodes:
            step = 1
            neighbours = node['neighbours'] + [node['id']]
            while step < steps:
                oldneighbours   = list(np.unique(neighbours))
                neighbours      = list(np.unique(np.hstack([[n] + self.nodes[n]['neighbours'] for n in oldneighbours])))
                step += 1
            
            
            data = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=neighbours),orient='index')
            meanData    = np.nanmean(data,axis=0)
            stdData     = np.nanstd(data,axis=0)
            for idx,key in enumerate(keys):
                node['kernelMean'+str(step)+key]  = meanData[idx]
                node['kernelStd'+str(step)+key]   = stdData[idx]

    # -------------------
    def diffRawBands(self,bandNamesPairs,bandNames,diffType='PixelWise',removeRaw=True):
        '''
        calculates de difference of specific bands. 

        Parameters
        ----------
        bandNamesPairs : list
            list of string containing the two bands which are used.
        bandNames : list
            list of string containing the new bandnames.
        diffType : string, optional
            Currently only option. The default is 'PixelWise'.
        removeRaw : bool, optional
            If removeRaw == True, the old bands are removed from the node and
            list of bandnames. The default is True.

        Returns
        -------
             Difference value is stored in each nodes under the chosen name(key).

        '''
        for idn,node in enumerate(self.nodes):
            for idx,[N1,N2] in enumerate(bandNamesPairs):
                if diffType == 'PixelWise':
                    diffVal = node[N1]-node[N2]
                    if removeRaw:
                        del node[N1],node[N2]
                        if idn == 0:
                            self.bandNames.remove(N1)
                            self.bandNames.remove(N2)
                    node[bandNames[idx]] = diffVal
                    if idn == 0:
                        self.bandNames = self.bandNames + [bandNames[idx]]
                        
    # -------------------
    def n_upwards_nodes(self,n_max):
        '''
        adds ids of all nodes upwards of seed to node['allUp']

        Parameters
        ----------
        n_max : int
            amount of "jumps" which should be done at max.

        Returns
        -------
            self.nodes[:]['allUp'] 
            self.nodes[:]['upSize']

        '''
        dist_max = n_max
        for n in range(len(self.nodes)): 
            dist = 0
            uNodes_new   = self.nodes[n]['upstream']
            uNodes_total = []
            uNodes_total.append(self.nodes[n]['id'])
            uNodes_total.extend(uNodes_new)
            while len(uNodes_new) >= 1 and dist <= dist_max:
                uNodes = []
                for i in range(len(uNodes_new)):
                    uNodes.extend(self.nodes[uNodes_new[i]]['upstream'])
                uNodes_total.extend(uNodes)
                uNodes_new = uNodes
                dist += 1
            self.nodes[n]['allUp']  = np.unique(uNodes_total)
            self.nodes[n]['upSize'] = len(np.unique(uNodes_total))  
                
    # -------------------
    def uS_stats(self,bandNames=None):
        '''
        running np.nanmean and np.nanstd for all neighbours and bands within a 
        the previously calculated graph neighbourhood. 

        Parameters
        ----------
        bandNames : list or None, optional
            Bandnames to calculate stats of, if None, all bandNames are used. 
            The default is None.

        Returns
        -------
        self.nodes[:]['uMean...']
        self.nodes['uStd...']

        '''
        if bandNames == None:
            keys = self.bandNamesAsc+self.bandNamesDesc+self.bandNamesOpt
        else:
            keys = bandNames
        self.uBandNames     = ['uMean'+band for band in keys]+['uStd'+band for band in keys]
        for node in self.nodes:
            try:
                data = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=node['allUp']),orient='index')
                meanData = np.nanmean(data,axis=0)
                stdData = np.nanstd(data,axis=0)
                
                for idx,key in enumerate(keys):
                    node['uMean'+key]   = meanData[idx]
                    node['uStd'+key]    = stdData[idx]
            except:
                print(str(node['id'])+' did not work')            
    # -------------------
    def n_downwards_nodes(self,n_max):
        '''
        adds ids of all nodes downwards of seed to node['allDown']

        Parameters
        ----------
        n_max : int
            amount of "jumps" which should be done at max.

        Returns
        -------
            self.nodes[:]['allDown'] 
            self.nodes[:]['downSize']

        '''
        dist_max = n_max
        for n in range(len(self.nodes)): 
            dist = 0
            dNodes_new   = self.nodes[n]['downstream']
            dNodes_total = []
            dNodes_total.append(self.nodes[n]['id'])
            dNodes_total.extend(dNodes_new)
            while len(dNodes_new) >= 1 and dist <= dist_max:
                dNodes = []
                for i in range(len(dNodes_new)):
                    dNodes.extend(self.nodes[dNodes_new[i]]['downstream'])
                dNodes_total.extend(dNodes)
                dNodes_new = dNodes
                dist += 1
            self.nodes[n]['allDown']  = np.unique(dNodes_total)
            self.nodes[n]['downSize'] = len(np.unique(dNodes_total)) 
            
    # -------------------
    def dS_stats(self,bandNames=None):
        '''
        running np.nanmean and np.nanstd for all neighbours and bands within a 
        the previously calculated graph neighbourhood. 

        Parameters
        ----------
        bandNames : list or None, optional
            Bandnames to calculate stats of, if None, all bandNames are used. 
            The default is None.

        Returns
        -------
        self.nodes[:]['dMean...']
        self.nodes['dStd...']

        '''        
        if bandNames == None:
            keys = self.bandNamesAsc+self.bandNamesDesc+self.bandNamesOpt
        else:
            keys = bandNames
        self.dBandNames     = ['dMean'+band for band in keys]+['dStd'+band for band in keys]
        for node in self.nodes:
            try:
                data = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=node['allDown']),orient='index')
                meanData = np.nanmean(data,axis=0)
                stdData = np.nanstd(data,axis=0)
                for idx,key in enumerate(keys):
                    node['dMean'+key]   = meanData[idx]
                    node['dStd'+key]    = stdData[idx]
            except:
                print(str(node['id'])+' did not work')
                
    def neighboursStats(self,bandNames=None,Neighbourhood='both',steps=3):
        '''
        

        Parameters
        ----------
        bandNames : TYPE, optional
            DESCRIPTION. The default is None.
        Neighbourhood : TYPE, optional
            DESCRIPTION. The default is 'both'.
        steps : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        '''
        if bandNames == None:
            keys = self.bandNamesAsc+self.bandNamesDesc+self.bandNamesOpt
        else:
            keys = bandNames

        if Neighbourhood == 'both':
            self.neighboursStatsBandNames   = ['kernelMean'+str(steps)+band for band in keys]+['kernelStd'+str(steps)+band for band in keys]
            self.gBandNames                 = ['gMean'+band for band in keys]+['gStd'+band for band in keys]
            self.dBandNames                 = ['dMean'+band for band in keys]+['dStd'+band for band in keys]
            self.uBandNames                 = ['uMean'+band for band in keys]+['uStd'+band for band in keys]
        
            self.n_downwards_nodes(steps)    
            self.n_upwards_nodes(steps)
            
            for node in self.nodes:
                # NxN window:
                neighbours = node['neighbours'] + [node['id']]
                for step in range(steps+1):
                    oldneighbours   = list(np.unique(neighbours))
                    neighbours      = list(np.unique(np.hstack([[n] + self.nodes[n]['neighbours'] for n in oldneighbours])))

                data = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=neighbours),orient='index')
                meanData    = np.nanmean(data,axis=0)
                stdData     = np.nanstd(data,axis=0)

                # Downstream
                dataDown = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=node['allDown']),orient='index')
                meanDownData = np.nanmean(dataDown,axis=0)
                stdDownData = np.nanstd(dataDown,axis=0)
                   
                # Upstream
                dataUp = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=node['allUp']),orient='index')
                meanUpData = np.nanmean(dataUp,axis=0)
                stdUpData = np.nanstd(dataUp,axis=0)
                
                for idx,key in enumerate(keys):
                    node['kernelMean'+str(step)+key]  = meanData[idx]
                    node['kernelStd'+str(step)+key]   = stdData[idx] 
                    
                    node['dMean'+key]   = meanDownData[idx]
                    node['dStd'+key]    = stdDownData[idx]
                    node['uMean'+key]   = meanUpData[idx]
                    node['uStd'+key]    = stdUpData[idx]
                    
                    node['gMean'+key]   = (meanDownData[idx]+meanUpData[idx])/2
                    node['gStd'+key]    = (stdUpData[idx]+stdDownData[idx])/2

        elif Neighbourhood == 'window':
            self.neighboursStatsBandNames   = ['kernelMean'+str(steps)+band for band in keys]+['kernelStd'+str(steps)+band for band in keys]
            
            for node in self.nodes:
                # NxN window:
                neighbours = node['neighbours'] + [node['id']]
                for step in range(steps+1):
                    oldneighbours   = list(np.unique(neighbours))
                    neighbours      = list(np.unique(np.hstack([[n] + self.nodes[n]['neighbours'] for n in oldneighbours])))

                data = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=neighbours),orient='index')
                meanData    = np.nanmean(data,axis=0)
                stdData     = np.nanstd(data,axis=0)

                
                for idx,key in enumerate(keys):
                    node['kernelMean'+str(step)+key]  = meanData[idx]
                    node['kernelStd'+str(step)+key]   = stdData[idx] 

        elif Neighbourhood == 'graph':
            self.gBandNames                 = ['gMean'+band for band in keys]+['gStd'+band for band in keys]
            self.dBandNames                 = ['dMean'+band for band in keys]+['dStd'+band for band in keys]
            self.uBandNames                 = ['uMean'+band for band in keys]+['uStd'+band for band in keys]
        
            self.n_downwards_nodes(steps)    
            self.n_upwards_nodes(steps)
            
            for node in self.nodes:
                # Downstream
                dataDown = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=node['allDown']),orient='index')
                meanDownData = np.nanmean(dataDown,axis=0)
                stdDownData = np.nanstd(dataDown,axis=0)
                   
                # Upstream
                dataUp = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=keys,nodesList=node['allUp']),orient='index')
                meanUpData = np.nanmean(dataUp,axis=0)
                stdUpData = np.nanstd(dataUp,axis=0)
                
                for idx,key in enumerate(keys):                    
                    node['dMean'+key]   = meanDownData[idx]
                    node['dStd'+key]    = stdDownData[idx]
                    node['uMean'+key]   = meanUpData[idx]
                    node['uStd'+key]    = stdUpData[idx]
                    
                    node['gMean'+key]   = (meanDownData[idx]+meanUpData[idx])/2
                    node['gStd'+key]    = (stdUpData[idx]+stdDownData[idx])/2        
        
    # ------------ ------------ Output Options ------------ ------------ 
    def dataArray(self,bandNames=None):
        '''
        creates numpy.arrays for selected data, groundtruth, training and terrain.
        function used in training and running of RFC

        Parameters
        ----------
        bandNames : list or None, optional
            band to extract as data. The default is None leading to all stored 
            band names.

        Returns
        -------
        list of np.arrays
            [data, terrain, training, groundTruth].

        '''
        if bandNames == None:
            bandNames = self.bandNames + self.uBandNames + self.dBandNames + self.gBandNames + self.neighboursStatsBandNames
            
        data        = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=bandNames,nodesList=None),orient='index')
        groundTruth = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=['GroundTruth'],nodesList=None),orient='index')
        terrain     = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=['terrain'],nodesList=None),orient='index')
        training    = pd.DataFrame.from_dict(self.dictIdxKeySubset(keyList=['training'],nodesList=None),orient='index')

        data        = data.to_numpy()
        terrain     = terrain.to_numpy()
        training    = training.to_numpy()
        groundTruth = groundTruth.to_numpy()
        terrain     = terrain[:,0]
        training    = training[:,0]
        groundTruth = groundTruth[:,0]
        
        return [data, terrain, training, groundTruth]
    
    def dataBandNames(self,bandNameGroups=None,data=['Opt','Asc','Desc']):
        '''
        

        Parameters
        ----------
        bandNameGroups : list of strings or none, optional
            Defines the bandName list(s) which are search (e.g. self.bandNames). 
            The default is None. This results includes all bandNames.
        data : list, optional
            Defines the datatypes which are search for in the given list of bandNames. 
            The default is ['Opt','Asc','Desc'].

        Returns
        -------
        bandNames : list of strings
            A subset of the band names.

        '''        
        bandNames = list()
        if bandNameGroups == None:
            bandNameGroups = self.bandNames+self.uBandNames+self.dBandNames+self.gBandNames+self.neighboursStatsBandNames
            
        for name in bandNameGroups:
            for band in data:
                if band in name:
                    bandNames.append(name)
                    
        return bandNames
    
    def train_RFC(self,n=100,terClasses=[10],loadOld=True,individualDataSets=True,bandNameGroups=None):
        '''
        

        Parameters
        ----------
        n : int, optional
            defines how many of the groundtruth == 0 pixels are used to train the
            unaffected class [0:-1:n]. The default is 100.
        terClasses : list of int, optional
            Selects which landcover class is used. The default is [10].
        loadOld : bool, optional
            Should old classifiers be loaded if they exist. 
            The default is True. If False, new RFCs are trained and overwrite 
            the olds.
        individualDataSets : bool, optional
            whould individual RFCs be trained for all datatypes [Asc,Desc,Opt]. 
            The default is True.
        bandNameGroups : None or list of strings, optional
            Which band groups should be included. The default is None.
            None leads to the inclusion of all bands and derived data.
            e.g., self.bandNames + self.gBandNames for the original bands and the
            graph neighbourhood derived statistics

        Returns
        -------
        self.RFC : dict()
            Dictionary with all trained RFCs stored
        
        external save of RFCs in wd

        '''
        
        if bandNameGroups == None:
            bandNameGroups = self.bandNames+self.uBandNames+self.dBandNames+self.gBandNames+self.neighboursStatsBandNames

        if individualDataSets:
            individualDataSets = ['Opt','Asc','Desc']

        check = True
        self.RFC = dict()
        
        for terClass in terClasses:
            print(' ')
            same_len_print('')
            same_len_print('Class: '+str(terClass))
            same_len_print('')
            print(' ')
            if os.path.isfile('RFC_TerrainClass_'+str(terClass)+'.sav') and loadOld:               
                same_len_print('Loading old classifier for all-Data class '+str(terClass)+'!')
                self.RFC[str(terClass)] = joblib.load('RFC_TerrainClass_'+str(terClass)+'.sav')
            else:
                if loadOld:
                    same_len_print('No Classifier trained for class '+str(terClass)+'!')
                else:
                    same_len_print('Replacing old classifier')
                    
                if check:
                    same_len_print('Extracting nodes data')
                    bandNames = self.dataBandNames(bandNameGroups)
                    [data, terrain, training, groundTruth] = self.dataArray(bandNames)
                    
                    check = False                    

                if terClass in np.unique(terrain[training > 0]).astype('int'):
                    # Find nodes with specific terrain class
                    classData       = data[terrain == terClass,:]
                    classTrain      = training[terrain == terClass]
                    classGT         = groundTruth[terrain == terClass]
                    
                    trainingData    = classData[classTrain > 0,:] 
                    
                    landslideData   = classData[classTrain[:]==1,:]
                    otherData       = classData[classGT[:]==0,:]
                                           
                    trainingData    = np.vstack([landslideData,otherData[0:-1:n,:]])        
                    trainingLabel   = np.vstack([np.ones((len(landslideData),1)),np.zeros((len(otherData[0:-1:n,:]),1))])[:,0]
                    
                    trainingData_noNaN  = trainingData[~np.isnan(np.min(trainingData,axis=1)),:]
                    trainingLabel_noNaN = trainingLabel[~np.isnan(np.min(trainingData,axis=1))]

                    
                    clf = RandomForestClassifier(max_depth=21, random_state=0)
        
                    clf.fit(trainingData_noNaN, trainingLabel_noNaN)
                    
                    self.RFC[str(terClass)] = clf
                    
                    joblib.dump(clf, 'RFC_TerrainClass_'+str(terClass)+'.sav')
                else:
                    same_len_print('Not enough training data in this Terrain Class')
                    
            if type(individualDataSets) == list:
                for dataSet in individualDataSets:
                
                    if os.path.isfile('RFC_TerrainClass_'+dataSet+str(terClass)+'.sav') and loadOld:               
                        same_len_print('Loading old classifier')
                        self.RFC[dataSet+str(terClass)] = joblib.load('RFC_TerrainClass_'+dataSet+str(terClass)+'.sav')
                    else:
                        if loadOld:
                            same_len_print('No Classifier trained for '+dataSet+'-Data class: '+str(terClass)+'!')
                        else:
                            same_len_print('Replacing old classifier')
                            

                        same_len_print('Extracting nodes '+ dataSet +'-data')
                        bandNames = self.dataBandNames(bandNameGroups,data=[dataSet])
                        [dataSub, terrainSub, trainingSub, groundTruthSub] = self.dataArray(bandNames=bandNames)
                 
        
                        if terClass in np.unique(terrainSub[trainingSub > 0]).astype('int'):
                            # Find nodes with specific terrain class
                            print(np.shape(dataSub))
                            classData       = dataSub[terrainSub == terClass,:]
                            classTrain      = trainingSub[terrainSub == terClass]
                            classGT         = groundTruthSub[terrainSub == terClass]
                            
                            trainingData    = classData[classTrain > 0,:] 
                            
                            landslideData   = classData[classTrain[:]==1,:]
                            otherData       = classData[classGT[:]==0,:]
                                                   
                            trainingData    = np.vstack([landslideData,otherData[0:-1:n,:]])        
                            trainingLabel   = np.vstack([np.ones((len(landslideData),1)),np.zeros((len(otherData[0:-1:n,:]),1))])[:,0]
                            
                            trainingData_noNaN  = trainingData[~np.isnan(np.min(trainingData,axis=1)),:]
                            trainingLabel_noNaN = trainingLabel[~np.isnan(np.min(trainingData,axis=1))]
        
                            
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                
                            clf.fit(trainingData_noNaN, trainingLabel_noNaN)
                            
                            self.RFC[dataSet+str(terClass)] = clf
                            
                            joblib.dump(clf, 'RFC_TerrainClass_'+dataSet+str(terClass)+'.sav')
                        else:
                            same_len_print('Not enough training data in this Terrain Class')            
            
    def load_RFC(self,terClasses=[10,20,30,40,50,60,70,80,90,95,100],individualDataSets=['Opt','Asc','Desc']):
        '''
        Loading RFCs from wd.
        
        Parameters
        ----------
        terClasses : list of int, optional
            Specifies the terrain class. The default is [10,20,30,40,50,60,70,80,90,95,100].
        individualDataSets : list of strings, optional
            Specifies the datatype. The default is ['Opt','Asc','Desc'].

        Returns
        -------
        self.RFC : dict
            Stores the RFC under the respectiv key.

        '''
        self.RFC = dict()
        for terClass in terClasses:
            if os.path.isfile('RFC_TerrainClass_'+str(terClass)+'.sav'):
                self.RFC[str(terClass)] = joblib.load('RFC_TerrainClass_'+str(terClass)+'.sav')
            else:
                same_len_print('No Classifier trained for class '+str(terClass)+'!')
            for dataSet in individualDataSets:    
                if os.path.isfile('RFC_TerrainClass_'+dataSet+str(terClass)+'.sav'):
                    self.RFC[dataSet+str(terClass)] = joblib.load('RFC_TerrainClass_'+dataSet+str(terClass)+'.sav')
                else:
                    same_len_print('No Classifier with individual '+dataSet+'-Data trained for class '+str(terClass)+'!') 
                    
    def run_RFC(self,terClasses=[10],individualDataSets='Opt',bandNameGroups=None):
        '''
        

        Parameters
        ----------
        terClasses : list of int, optional
            Select which terrain to classify. The default is [10].

        individualDataSets : string or None, optional
            Which RFC to use. The default is 'Opt'.
        bandNameGroups : None or list of strings, optional
            Which band groups should be included. The default is None.
            None leads to the inclusion of all bands and derived data.
            e.g., self.bandNames + self.gBandNames for the original bands and the
            graph neighbourhood derived statistics

        Returns
        -------
        self.nodes[i]['predicted Label'] : int
            By RFC predicted class.

        '''
        if bandNameGroups == None:
            bandNameGroups = self.bandNames+self.uBandNames+self.dBandNames+self.gBandNames+self.neighboursStatsBandNames
            
        same_len_print('Applying Random Forest Classifier to nodes')        
                
        masks = pd.DataFrame.from_dict(
                self.dictIdxKeySubset(keyList=['MaskO','MaskD','MaskA']),
                orient='index')     
        
        nodeId = pd.DataFrame.from_dict(
                 self.dictIdxKeySubset(keyList=['id']),
                 orient='index').to_numpy()[:,0]  
        
        bandNames       = self.dataBandNames(bandNameGroups)
        [dataCombined, terrain,_,_] = self.dataArray(bandNames=bandNames)      
        
        dataOpt     = dataCombined[:,['Opt' in b for b in bandNames]]
        dataAsc     = dataCombined[:,['Asc' in b for b in bandNames]]
        dataDesc    = dataCombined[:,['Desc' in b for b in bandNames]]
        
        if individualDataSets=='Opt':
            if masks.to_numpy().all()==0:
                for terClass in terClasses:
                    classData  = dataOpt[terrain==terClass,:]
                    classNodes = nodeId[terrain==terClass]
                    
                    classData_noNaN  = classData[~np.isnan(np.min(classData,axis=1)),:]
                    classNodes_noNaN = classNodes[~np.isnan(np.min(classData,axis=1))]
                    
                    if individualDataSets+str(int(terClass)) in self.RFC.keys():
                        classPrediction = self.RFC[individualDataSets+str(int(terClass))].predict(classData_noNaN)
                    else:
                        self.load_RFC(terClasses=[terClass],individualDataSets=[individualDataSets])
                        classPrediction = self.RFC[individualDataSets+str(int(terClass))].predict(classData_noNaN)
                    
                    for node in range(len(classPrediction)):
                        self.nodes[classNodes_noNaN[node]]['predicted Label'] = classPrediction[node]
                        
        elif individualDataSets=='Asc':
            if masks.to_numpy().all()==0:
                for terClass in terClasses:
                    classData  = dataAsc[terrain==terClass,:]
                    classNodes = nodeId[terrain==terClass]
                    
                    classData_noNaN  = classData[~np.isnan(np.min(classData,axis=1)),:]
                    classNodes_noNaN = classNodes[~np.isnan(np.min(classData,axis=1))]
                    
                    if individualDataSets+str(int(terClass)) in self.RFC.keys():
                        classPrediction = self.RFC[individualDataSets+str(int(terClass))].predict(classData_noNaN)
                    else:
                        self.load_RFC(terClasses=[terClass],individualDataSets=[individualDataSets])
                        classPrediction = self.RFC[individualDataSets+str(int(terClass))].predict(classData_noNaN)
                    
                    for node in range(len(classPrediction)):
                        self.nodes[classNodes_noNaN[node]]['predicted Label'] = classPrediction[node] 
                        
        elif individualDataSets=='Desc':
            if masks.to_numpy().all()==0:
                for terClass in terClasses:
                    classData  = dataDesc[terrain==terClass,:]
                    classNodes = nodeId[terrain==terClass]
                    
                    classData_noNaN  = classData[~np.isnan(np.min(classData,axis=1)),:]
                    classNodes_noNaN = classNodes[~np.isnan(np.min(classData,axis=1))]
                    
                    if individualDataSets+str(int(terClass)) in self.RFC.keys():
                        classPrediction = self.RFC[individualDataSets+str(int(terClass))].predict(classData_noNaN)
                    else:
                        self.load_RFC(terClasses=[terClass],individualDataSets=[individualDataSets])
                        classPrediction = self.RFC[individualDataSets+str(int(terClass))].predict(classData_noNaN)
                    
                    for node in range(len(classPrediction)):
                        self.nodes[classNodes_noNaN[node]]['predicted Label'] = classPrediction[node] 
                        
        else:
            if masks.to_numpy().all()==0:
                for terClass in terClasses:
                    classData  = dataCombined[terrain==terClass,:]
                    classNodes = nodeId[terrain==terClass]
                    
                    classData_noNaN  = classData[~np.isnan(np.min(classData,axis=1)),:]
                    classNodes_noNaN = classNodes[~np.isnan(np.min(classData,axis=1))]
                    
                    if str(int(terClass)) in self.RFC.keys():
                        classPrediction = self.RFC[str(int(terClass))].predict(classData_noNaN)
                    else:
                        self.load_RFC(terClasses=[terClass])
                        classPrediction = self.RFC[str(int(terClass))].predict(classData_noNaN)
                    
                    for node in range(len(classPrediction)):
                        self.nodes[classNodes_noNaN[node]]['predicted Label'] = classPrediction[node]
        
    # ------------ ------------ Display Options ------------ ------------                 
    def plotGraph(self,Marker = 'h', Nr=None,reducer=1, zMin=0, zMax=3000, minSlope=0, maxSlope=90, attribute='cluster', bandNr=0, cMap='terrain', mSize=0.1, pTitle='', xLabel='Easting', yLabel='Northing', zLabel='Height', ele=30, az=135):
        
        if pTitle == '':
            pTitle = attribute
        x       = []
        y       = []
        z       = []
        nEdges  = []

        for node in self.nodes:
            if type(node['slope']) == np.float64 and node['slope'] >= minSlope and node['slope'] <= maxSlope:
                if attribute == 'edges':
                    x.append(node['coords'][0])
                    y.append(node['coords'][1])
                    z.append(node['coords'][2])
                    nEdges.append(len(node['edges']))
                
                elif type(node[attribute]) == np.ndarray or type(node[attribute]) == list:
                    x.append(node['coords'][0])
                    y.append(node['coords'][1])
                    z.append(node['coords'][2])
                    nEdges.append(node[attribute][bandNr])
                else:
                    x.append(node['coords'][0])
                    y.append(node['coords'][1])
                    z.append(node['coords'][2])
                    nEdges.append(node[attribute])                    
            
        fig = plt.figure(figsize=(6.4, 4.8),num=Nr,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[0::reducer], y[0::reducer], z[0::reducer], marker=Marker, c=nEdges[0::reducer],cmap= cMap, s=mSize)
        ax.set_title(pTitle)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_zlabel(zLabel)
        ax.view_init(elev=ele, azim=az)
        ax.set_zlim3d(zMin,zMax)
        plt.gca().invert_xaxis()
        #plt.gca().invert_yaxis()

        fig.show()

        if Nr!=None:
            plt.savefig('Figures\\Graphplot_'+str(pTitle)+'_'+str(Nr)+'.png',dpi=150)  

    def plotEdges(self,cProp = 'upSize', cMap ='terrain',mSize = 0.1, pTitle = '', xLabel = 'Easting', yLabel = 'Northing', zLabel = 'Height', ele = 30, az = 135):
        edgelist = self.edgelist
        el       = np.vstack(edgelist)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(edgelist)):
            if cProp == 'upSize':
                upNorm = self.nodes[el[i,0]]['upSize']/self.maxUpSize

                ax.plot([self.nodes[el[i,0]]['coords'][0],self.nodes[el[i,1]]['coords'][0]], 
                        [self.nodes[el[i,0]]['coords'][1],self.nodes[el[i,1]]['coords'][1]], 
                        [self.nodes[el[i,0]]['coords'][2],self.nodes[el[i,1]]['coords'][2]],
                        linewidth=mSize,color=cm.get_cmap(cMap)(upNorm))
            else:
                ax.plot([self.nodes[el[i,0]]['coords'][0],self.nodes[el[i,1]]['coords'][0]], 
                        [self.nodes[el[i,0]]['coords'][1],self.nodes[el[i,1]]['coords'][1]], 
                        [self.nodes[el[i,0]]['coords'][2],self.nodes[el[i,1]]['coords'][2]],
                        linewidth=mSize,color=cMap)
            

        ax.set_title(pTitle)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_zlabel(zLabel)
        ax.view_init(ele,az)
        fig.show()     


# %% gdal Functions
def getExtent(ds):
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height                    = ds.RasterXSize, ds.RasterYSize
    xmax                             = xmin + width * xpixel
    ymin                             = ymax + height * ypixel

    return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)

def get_coord_transform(source_epsg, target_epsg):
    '''
    Creates an OGR-framework coordinate transformation for use in projecting
    coordinates to a new coordinate reference system (CRS). Used as, e.g.:
        transform = get_coord_transform(source_epsg, target_epsg)
        transform.TransformPoint(x, y)
    Arguments:
        source_epsg     The EPSG code for the source CRS
        target_epsg     The EPSG code for the target CRS
    '''
    # Develop a coordinate transformation, if desired
    source_ref = osr.SpatialReference()
    target_ref = osr.SpatialReference()
    source_ref.ImportFromEPSG(source_epsg)
    target_ref.ImportFromEPSG(target_epsg)
    return osr.CoordinateTransformation(source_ref, target_ref) 

# %% Gdal Functions
def preprocess(filePathList,resamplingMethod='NearestNeighbour',tempFile='.//temp//reProj.tif'):
    # Open SAR I
    ds                  = gdal.Open(filePathList[0])
    dsProj              = ds.GetProjection()
    dsTran              = ds.GetGeoTransform()

    dsEPSG              = 'EPSG:'+dsProj.split('"')[-2]
    width, height       = ds.RasterXSize, ds.RasterYSize
    bandNumber          = ds.RasterCount
    
    # Open SAR II
    ds_n                = gdal.Open(filePathList[1])
    dsProj_n            = ds_n.GetProjection()
    
    
    
    dst = gdal.GetDriverByName('GTiff').Create(tempFile, width, height, bandNumber, gdalconst.GDT_Float32)
    dst.SetGeoTransform(dsTran)
    dst.SetProjection(dsProj)
    
    # Reproject Image
    if resamplingMethod == 'Bilinear':
        gdal.ReprojectImage(ds_n, dst, dsProj_n, dsProj, gdalconst.GRA_Bilinear)
    elif resamplingMethod == 'NearestNeighbour':
        gdal.ReprojectImage(ds_n, dst, dsProj_n, dsProj, gdalconst.GRA_NearestNeighbour)
        
    eeTran          = [dsTran[1],dsTran[2],dsTran[0],dsTran[4],dsTran[5],dsTran[3]]
    eeProj          = ee.Projection(dsEPSG,eeTran)
    eeProj.getInfo()
    
    # eeAOI           = ee.Geometry.Polygon(eeAOI)
    for b in range(ds.RasterCount):
        if b == 0:
            dsImg           = ds.GetRasterBand(b+1).ReadAsArray()
        else:
            dsImg   = np.dstack((dsImg,ds.GetRasterBand(b+1).ReadAsArray()))
            
    for b in range(dst.RasterCount):
        if b == 0:
            dstImg           = dst.GetRasterBand(b+1).ReadAsArray()
        else:
            dstImg   = np.dstack((dstImg,dst.GetRasterBand(b+1).ReadAsArray()))
            
    return dsImg,dstImg,eeProj,eeTran,dsTran                

# %% Code:    
# %% Preprocessing of Reference Images:
AscImg,AscImgAfter,eeProj,eeTran,dsTran = preprocess(AscFiles)
_,DescImg,_,_,_                         = preprocess([AscFiles[0],DescFiles[0]])
_,DescImgAfter,_,_,_                    = preprocess([AscFiles[0],DescFiles[1]])
_,Training,_,_,_                        = preprocess([AscFiles[0],TrainingFiles[0]])
_,GroundTruth,_,_,_                     = preprocess([AscFiles[0],GTFiles[0]])

# only necessary for one gT file
Training[Training > 3]  = 0 
Training[Training > 0]  = 1
# Training                = Training[:,:,0]



# %% EE Data collection and processing

testCount       = 1

height,width    = np.shape(AscImg[:,:,0])
            
ulY             = height
ulX             = width 
lrX             = 0
lrY             = 0

dsPixelCount    = width * height

resultsMap      = np.empty((height,width))*np.nan
lCMap           = np.empty((height,width))*np.nan

s2MapBefore     = np.empty((height,width,5))*np.nan
s2MapAfter      = np.empty((height,width,5))*np.nan
terrainMap      = np.empty((height,width,3))*np.nan
coordMap        = np.empty((height,width,3))*np.nan



xArray          = [(i*dsTran[1])+dsTran[0] for i in range(width)]
yArray          = [(i*dsTran[5])+dsTran[3] for i in range(height)]


xMax    = int(np.ceil(width/511))
yMax    = int(np.ceil(height/511))
xMin    = 0
yMin    = 0
    
for x in range(xMin,xMax):
    if testCount <= testCountLimit:      
        xStart = x * 511
        if x < xMax:
            xStop = (x+1)*511
        else:
            xStop = width
        
        for y in range(yMin,yMax):
            terrainTest     = False
            landCoverTest   = False
            S1Test          = False
            S2Test          = False
            # Trail Option:
            if testCount <= testCountLimit:
                print(' ')
                print('Subset: y - '+str(y)+'/'+str(yMax)+'  x - '+str(x)+'/'+str(xMax))
                yStart = y*511
                if y < xMax:
                    yStop = (y+1)*511
                else:
                    yStop = height
                
                dsImgSubset = AscImg[yStart:yStop,xStart:xStop,0]
                s1Test = False
                if not (dsImgSubset[:]==0).all():
                    s1Test = True
                    boxAOI = ee.Geometry.Polygon(coords=
                             [[eeTran[2]+xStart*eeTran[0],eeTran[5]+yStop*eeTran[4]],
                              [eeTran[2]+xStop*eeTran[0],eeTran[5]+yStop*eeTran[4]],
                              [eeTran[2]+xStop*eeTran[0],eeTran[5]+yStart*eeTran[4]],
                              [eeTran[2]+xStart*eeTran[0],eeTran[5]+yStart*eeTran[4]],
                              [eeTran[2]+xStart*eeTran[0],eeTran[5]+yStop*eeTran[4]]])
                    
                    # Adding terrain data subset
                    terrainTest   = False
                    if excludeFlatland:
                        slopeTest = True
                    else:
                        slopeTest = False
                    try:
                        terrain, coords, xy = getTerrainCoords(boxAOI,eeProj)
                        terrainTest         = True
                    except:
                        print('No SRTM Data in this subset.')
                        terrainTest = False
                    
                    if terrainTest:    
                        eeYSize,eeXSize     = np.shape(terrain[:,:,0])
                        eeXStart,eeYStart   = xy[0,0,:]-0.5
                        if eeYStart >= 0:
                            eeYStop         = eeYStart + eeYSize
                            eeYShift        = 0
                            
                        elif eeYStart < 0:
                            eeYStop         = eeYSize + eeYStart
                            eeYShift        = int(abs(eeYStart))
                            eeYStart        = 0
                            
                        if eeXStart >= 0:    
                            eeXStop         = eeXStart + eeXSize
                            eeXShift        = 0
                            
                        elif eeXStart < 0:
                            eeXStop         = eeXSize + eeXStart
                            eeXShift        = int(abs(eeXStart))
                            eeXStart        = 0     
                            
                        if excludeFlatland:
                            slopeTest = (terrain[:,:,1]>=minAngle).any()
                        else:
                            slopeTest == True
                        if slopeTest:
                            if eeYStop <= height and eeXStop <= width:
                                terrainMap[int(eeYStart):int(eeYStop),
                                           int(eeXStart):int(eeXStop),:] = terrain[eeYShift:,eeXShift:,:]                            
                                coordMap[int(eeYStart):int(eeYStop),
                                         int(eeXStart):int(eeXStop),:]   = coords[eeYShift:,eeXShift:,:]
                            elif eeYStop > height and eeXStop > width:
                                terrainMap[int(eeYStart):,int(eeXStart):,:]          = terrain[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:(width-int(eeXStart)),:]
                                coordMap[int(eeYStart):,int(eeXStart):,:]          = coords[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:(width-int(eeXStart)),:] 
                            elif eeYStop > height:
                                terrainMap[int(eeYStart):,
                                           int(eeXStart):int(eeXStop),:]        = terrain[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:,:]
                                coordMap[int(eeYStart):,
                                           int(eeXStart):int(eeXStop),:]        = coords[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:,:] 
                            elif eeXStop > width:
                                terrainMap[int(eeYStart):int(eeYStop),
                                           int(eeXStart):,:]                    = terrain[eeYShift:,
                                                                                          eeXShift:(width-int(eeXStart)),:]                                
                                coordMap[int(eeYStart):int(eeYStop),
                                           int(eeXStart):,:]                    = coords[eeYShift:,
                                                                                          eeXShift:(width-int(eeXStart)),:]                                  
                                
                        else:
                            print('No areas with slope gt '+str(minAngle)+'.')
                            
                    if slopeTest:
                        # Adding landcover data subset
                        landCoverTest = False    
                        try:
                            if landCoverYear >= 2020:
                                landCover,xy    = getWorldCover(boxAOI,eeProj)
                            else:
                                landCover,xy    = getCopernicusLandcover(boxAOI,eeProj,year=landCoverYear)
                            landCoverTest   = True
                        except:
                            print('No WorldCover Data in this subset.')
                            landCoverTest   = False
                            
                        if landCoverTest:
                            eeYSize,eeXSize     = np.shape(landCover[:,:,0])
                            eeXStart,eeYStart   = xy[0,0,:]-0.5
                            if eeYStart >= 0:
                                eeYStop         = eeYStart + eeYSize
                                eeYShift        = 0
                                
                            elif eeYStart < 0:
                                eeYStop         = eeYSize + eeYStart
                                eeYShift        = int(abs(eeYStart))
                                eeYStart        = 0
                                
                            if eeXStart >= 0:    
                                eeXStop         = eeXStart + eeXSize
                                eeXShift        = 0
                                
                            elif eeXStart < 0:
                                eeXStop         = eeXSize + eeXStart
                                eeXShift        = int(abs(eeXStart))
                                eeXStart        = 0                            
                                
                            if eeYStop <= height and eeXStop <= width:
                                lCMap[int(eeYStart):int(eeYStop),
                                           int(eeXStart):int(eeXStop)]        = landCover[eeYShift:,
                                                                                          eeXShift:,0]   
                            elif eeYStop > height and eeXStop > width:
                                lCMap[int(eeYStart):,int(eeXStart):]          = landCover[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:(width-int(eeXStart)),0] 
                            elif eeYStop > height:
                                lCMap[int(eeYStart):,
                                           int(eeXStart):int(eeXStop)]        = landCover[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:,0] 
                            elif eeXStop > width:
                                lCMap[int(eeYStart):int(eeYStop),
                                           int(eeXStart):]                    = landCover[eeYShift:,
                                                                                          eeXShift:(width-int(eeXStart)),0]                              
                            
                            if eeXStop > lrX:
                                lrX = int(eeXStop)
                            if eeYStop > lrY:
                                lrY = int(eeYStop)
                                
                            if eeXStart < ulX:
                                ulX = int(eeXStart)
                            if eeYStart < ulY:
                                ulY = int(eeYStart)
                            
                        # s2 Data
                        s2BeforeTest        = False
                        try:
                            s2Before,xy     = getSentinel2(aoi=boxAOI,startDate=dateSurveyYb,
                                                           endDate=dateBefore,proj=eeProj,
                                                           add_NDVI=True,
                                                           add_NDWI=False,
                                                           SR=False,orbit='DESCENDING',
                                                           maskClouds='light')
                            s2BeforeTest    = True
                        except:
                            print('No S2 before Data in this subset.')
                            s2BeforeTest    = False
                            
                        if s2BeforeTest:
                            eeYSize,eeXSize     = np.shape(s2Before[:,:,0])
                            eeXStart,eeYStart   = xy[0,0,:]-0.5
                            if eeYStart >= 0:
                                eeYStop         = eeYStart + eeYSize
                                eeYShift        = 0
                                
                            elif eeYStart < 0:
                                eeYStop         = eeYSize + eeYStart
                                eeYShift        = int(abs(eeYStart))
                                eeYStart        = 0
                                
                            if eeXStart >= 0:    
                                eeXStop         = eeXStart + eeXSize
                                eeXShift        = 0
                                
                            elif eeXStart < 0:
                                eeXStop         = eeXSize + eeXStart
                                eeXShift        = int(abs(eeXStart))
                                eeXStart        = 0                            
                                
                            if eeYStop <= height and eeXStop <= width:
                                s2MapBefore[int(eeYStart):int(eeYStop),
                                           int(eeXStart):int(eeXStop),:]        = s2Before[eeYShift:,
                                                                                          eeXShift:,:]   
                            elif eeYStop > height and eeXStop > width:
                                s2MapBefore[int(eeYStart):,int(eeXStart):,:]    = s2Before[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:(width-int(eeXStart)),:] 
                            elif eeYStop > height:
                                s2MapBefore[int(eeYStart):,
                                           int(eeXStart):int(eeXStop),:]        = s2Before[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:,:] 
                            elif eeXStop > width:
                                s2MapBefore[int(eeYStart):int(eeYStop),
                                           int(eeXStart):,:]                    = s2Before[eeYShift:,
                                                                                          eeXShift:(width-int(eeXStart)),:]  
                        
                        s2AfterTest         = False        
                        try:
                            s2After,xy     = getSentinel2(aoi=boxAOI,startDate=dateSurvey,
                                                           endDate=dateAfter,proj=eeProj,
                                                           add_NDVI=True,
                                                           add_NDWI=False,
                                                           SR=False,orbit='DESCENDING',
                                                           maskClouds='light')
                            s2AfterTest    = True
                        except:
                            print('No S2 after Data in this subset.')
                            s2AfterTest    = False
                            
                        if s2AfterTest:
                            eeYSize,eeXSize     = np.shape(s2After[:,:,0])
                            eeXStart,eeYStart   = xy[0,0,:]-0.5
                            if eeYStart >= 0:
                                eeYStop         = eeYStart + eeYSize
                                eeYShift        = 0
                                
                            elif eeYStart < 0:
                                eeYStop         = eeYSize + eeYStart
                                eeYShift        = int(abs(eeYStart))
                                eeYStart        = 0
                                
                            if eeXStart >= 0:    
                                eeXStop         = eeXStart + eeXSize
                                eeXShift        = 0
                                
                            elif eeXStart < 0:
                                eeXStop         = eeXSize + eeXStart
                                eeXShift        = int(abs(eeXStart))
                                eeXStart        = 0                            
                                
                            if eeYStop <= height and eeXStop <= width:
                                s2MapAfter[int(eeYStart):int(eeYStop),
                                           int(eeXStart):int(eeXStop),:]        = s2After[eeYShift:,
                                                                                          eeXShift:,:]   
                            elif eeYStop > height and eeXStop > width:
                                s2MapAfter[int(eeYStart):,int(eeXStart):,:]     = s2After[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:(width-int(eeXStart)),:] 
                            elif eeYStop > height:
                                s2MapAfter[int(eeYStart):,
                                           int(eeXStart):int(eeXStop),:]        = s2After[eeYShift:(height-int(eeYStart)),
                                                                                          eeXShift:,:] 
                            elif eeXStop > width:
                                s2MapAfter[int(eeYStart):int(eeYStop),
                                           int(eeXStart):,:]                    = s2After[eeYShift:,
                                                                                          eeXShift:(width-int(eeXStart)),:] 
                                                                           
                        
                        if landCoverTest and terrainTest and s2BeforeTest and s2AfterTest and s1Test:
                            
                            dataSubset              = dict()
                            dataSubset['Landcover'] = lCMap[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop)]
                            dataSubset['Terrain']   = terrainMap[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),:]
                            dataSubset['Coords']    = coordMap[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),:]
                            dataSubset['S1Asc']     = np.log(AscImgAfter[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),1]/AscImg[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),1])
                            dataSubset['S1Desc']    = np.log(DescImgAfter[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),1]/DescImg[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),1])
                            dataSubset['S2']        = np.dstack(((s2MapAfter[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),:-1]-
                                                                    s2MapBefore[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),:-1])/
                                                                   s2MapBefore[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),:-1],
                                                                   s2MapAfter[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),-1]-
                                                                   s2MapBefore[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop),-1]))
                            
                            dataSubset['Training']      = Training[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop)]
                            dataSubset['GroundTruth']   = GroundTruth[int(eeYStart):int(eeYStop),int(eeXStart):int(eeXStop)]                            
                            
                            dataSubset['S1Bands']   = ['Sigma0VV']
                            dataSubset['S2Bands']   = ['OptB2','OptB3','OptB4','OptB8','OptNDVI']
                            
                            

                            
                            print(str(testCount)+' Subset data collection completed')
                          
                            if calculateGraph: # and dataSubset['GroundTruth'].any() == 1:
                                g = graphNet(dataSubset)
                                g.add_stream_direction()
                                g.get_endNodes()
                                g.neighboursStats(steps=3)
                                
                                # Run with graph stats
                                g.train_RFC(bandNameGroups = g.bandNames+g.neighboursStatsBandNames)
                                g.run_RFC(bandNameGroups = g.bandNames+g.neighboursStatsBandNames)   # individualDataSets='Opt',
                                
                                gTGraph         = pd.DataFrame.from_dict(
                                                  g.dictIdxKeySubset(keyList=['GroundTruth']),
                                                  orient='index').to_numpy()[:,0]
                                
                                trainGraph      = pd.DataFrame.from_dict(
                                                  g.dictIdxKeySubset(keyList=['training']),
                                                  orient='index').to_numpy()[:,0]
                                
                                pointCoordsGraph = pd.DataFrame.from_dict(
                                                   g.dictIdxKeySubset(keyList=['coords']),
                                                   orient='index')
                                pointCoordsGraph = np.vstack(pointCoordsGraph['coords'])
                                
                                predictedGraph  = pd.DataFrame.from_dict(
                                                  g.dictIdxKeySubset(keyList=['predicted Label']),
                                                  orient='index')
                                                            
                                
                                
                                resultsMap[int(eeYStart):int(eeYStart+g.rows),
                                           int(eeXStart):int(eeXStart+g.cols)] = np.reshape(predictedGraph.to_numpy(),(g.rows,g.cols))

                             # Count subsets with all data available   
                            testCount += 1
                        else:
                            print('Subset incomplete')
                else:
                    print('No SAR-data')
            else:
                # break y loop if maximal test count is reached
                break

    else:
        # break x loop if maximal test count is reached
        print('Test ran for '+str(testCount)+' subset')
        break  

# %%
if prints:
    #%%
    pN = 10
    fontName = 'Tahoma'
    titleS   = '12'
    labelA   = '10'
    tickS    = '8' 
    
    ele=30
    az=105
    
    yMax = 175
    xMax = 175
    
    [xmin,xmax] = [(xMax-100)*10,xMax*10]#[0,6400]
    [ymin,ymax] = [(yMax-100)*10,yMax*10]#[0,6400]
    [zmin,zmax] = [400,1200]#[0,1500]
    
    norm = Normalize(vmin=-0.2, vmax=0.2)
    
    fig         = plt.figure(figsize=(11.69,8.27))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ele, azim=az)
    
    
    
    for edge in g.edgelist:
        if (g.nodes[edge[0]]['imageCoords'][0] > yMax - 100 and
            g.nodes[edge[0]]['imageCoords'][0] < yMax and 
            g.nodes[edge[0]]['imageCoords'][1] < xMax and
            g.nodes[edge[0]]['imageCoords'][1] > xMax - 100):
            
            ax.plot([g.nodes[edge[0]]['imageCoords'][0]*10,g.nodes[edge[1]]['imageCoords'][0]*10],
                    [g.nodes[edge[0]]['imageCoords'][1]*10,g.nodes[edge[1]]['imageCoords'][1]*10],
                    [g.nodes[edge[0]]['coords'][2],g.nodes[edge[1]]['coords'][2]],
                    linewidth=np.log(g.nodes[edge[0]]['upSize'])/2,
                    color=cm.get_cmap(lsc.from_list(None,['firebrick','grey','lightsteelblue']))((norm(g.nodes[edge[0]]['OptNDVI'])))
                    )
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('black')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='black')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    # ax.xaxis._axinfo['color'] = (0,0,0,1)
    # ax.yaxis._axinfo['color'] = (0,0,0,1)
    # ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    plt.savefig('NDVISmall.png', transparent=True,dpi=600)
    
    # %% UMean
    fig         = plt.figure(figsize=(11.69*2,8.27*2))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ele, azim=az)
    
    for edge in g.edgelist:
        if (g.nodes[edge[0]]['imageCoords'][0] > yMax - 100 and
            g.nodes[edge[0]]['imageCoords'][0] < yMax and 
            g.nodes[edge[0]]['imageCoords'][1] < xMax and
            g.nodes[edge[0]]['imageCoords'][1] > xMax - 100):
            
            ax.plot([g.nodes[edge[0]]['imageCoords'][0]*10,g.nodes[edge[1]]['imageCoords'][0]*10],
                    [g.nodes[edge[0]]['imageCoords'][1]*10,g.nodes[edge[1]]['imageCoords'][1]*10],
                    [g.nodes[edge[0]]['coords'][2],g.nodes[edge[1]]['coords'][2]],
                    linewidth=np.log(g.nodes[edge[0]]['upSize']),
                    color=cm.get_cmap(lsc.from_list(None,['firebrick','grey','lightsteelblue']))((norm(g.nodes[edge[0]]['uMeanOptNDVI'])))
                    )
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('black')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='black')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    # ax.xaxis._axinfo['color'] = (0,0,0,1)
    # ax.yaxis._axinfo['color'] = (0,0,0,1)
    # ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    
    plt.savefig('uMeanNDVISmall.png', transparent=True,dpi=1200)          
    
    # %% u & dMean
    fig         = plt.figure(figsize=(11.69,8.27))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ele, azim=az)
    
    for edge in g.edgelist:
        if (g.nodes[edge[0]]['imageCoords'][0] > yMax - 100 and
            g.nodes[edge[0]]['imageCoords'][0] < yMax and 
            g.nodes[edge[0]]['imageCoords'][1] < xMax and
            g.nodes[edge[0]]['imageCoords'][1] > xMax - 100):
            
            ax.plot([g.nodes[edge[0]]['imageCoords'][0]*10,g.nodes[edge[1]]['imageCoords'][0]*10],
                    [g.nodes[edge[0]]['imageCoords'][1]*10,g.nodes[edge[1]]['imageCoords'][1]*10],
                    [g.nodes[edge[0]]['coords'][2],g.nodes[edge[1]]['coords'][2]],
                    linewidth=np.log(g.nodes[edge[0]]['upSize'])/2,
                    color=cm.get_cmap(lsc.from_list(None,['firebrick','grey','lightsteelblue']))(norm(g.nodes[edge[0]]['gMeanOptNDVI']))
                    )
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('black')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='black')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    # ax.xaxis._axinfo['color'] = (0,0,0,1)
    # ax.yaxis._axinfo['color'] = (0,0,0,1)
    # ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    
    plt.savefig('udMeanNDVISmall.png', transparent=True,dpi=600)  
    
    # %% dMean
    fig         = plt.figure(figsize=(11.69*2,8.27*2))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ele, azim=az)
    
    
    
    for edge in g.edgelist:
        if (g.nodes[edge[0]]['imageCoords'][0] > yMax - 100 and
            g.nodes[edge[0]]['imageCoords'][0] < yMax and 
            g.nodes[edge[0]]['imageCoords'][1] < xMax and
            g.nodes[edge[0]]['imageCoords'][1] > xMax - 100):
            
            ax.plot([g.nodes[edge[0]]['imageCoords'][0]*10,g.nodes[edge[1]]['imageCoords'][0]*10],
                    [g.nodes[edge[0]]['imageCoords'][1]*10,g.nodes[edge[1]]['imageCoords'][1]*10],
                    [g.nodes[edge[0]]['coords'][2],g.nodes[edge[1]]['coords'][2]],
                    linewidth=np.log(g.nodes[edge[0]]['upSize'])/2,
                    color=cm.get_cmap(lsc.from_list(None,['firebrick','ivory','lightsteelblue']))((norm(g.nodes[edge[0]]['dMeanOptNDVI'])))
                    )
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('white')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='white')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    ax.xaxis._axinfo['color'] = (0,0,0,1)
    ax.yaxis._axinfo['color'] = (0,0,0,1)
    ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    plt.savefig('dMeanNDVISmall.png', transparent=True,dpi=1200)        
    
    # %%  7x7 Mean
    fig         = plt.figure(figsize=(11.69,8.27))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ele, azim=az)
    
    
    
    for edge in g.edgelist:
        if (g.nodes[edge[0]]['imageCoords'][0] > yMax - 100 and
            g.nodes[edge[0]]['imageCoords'][0] < yMax and 
            g.nodes[edge[0]]['imageCoords'][1] < xMax and
            g.nodes[edge[0]]['imageCoords'][1] > xMax - 100):
            
            ax.plot([g.nodes[edge[0]]['imageCoords'][0]*10,g.nodes[edge[1]]['imageCoords'][0]*10],
                    [g.nodes[edge[0]]['imageCoords'][1]*10,g.nodes[edge[1]]['imageCoords'][1]*10],
                    [g.nodes[edge[0]]['coords'][2],g.nodes[edge[1]]['coords'][2]],
                    linewidth=np.log(g.nodes[edge[0]]['upSize'])/2,
                    color=cm.get_cmap(lsc.from_list(None,['firebrick','grey','lightsteelblue']))((norm(g.nodes[edge[0]]['kernelMean3OptNDVI'])))
                    )
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('black')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='black')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='black')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    # ax.xaxis._axinfo['color'] = (0,0,0,1)
    # ax.yaxis._axinfo['color'] = (0,0,0,1)
    # ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    
    plt.savefig('mean3NDVISmall.png', transparent=True,dpi=600)        
    
    # %% nodeList
    nodeIdx = list()
    for node in g.nodes:
        if (node['imageCoords'][0] > yMax - 100 and
            node['imageCoords'][0] < yMax and 
            node['imageCoords'][1] < xMax and
            node['imageCoords'][1] > xMax - 100):
            nodeIdx = nodeIdx + [node['id']]
        
    
    dispData = pd.DataFrame.from_dict(
                    g.dictIdxKeySubset(
                        keyList=['imageCoords','coords','OptNDVI',
                                 'mean3OptNDVI','dMeanOptNDVI',
                                 'uMeanOptNDVI','predicted Label',
                                 'terrain'],nodesList=nodeIdx),
                    orient='index')
    
    coords          = np.vstack(dispData['coords'])
    coords[:,0]     = np.vstack(dispData['imageCoords'].to_numpy())[:,0]*10
    coords[:,1]     = np.vstack(dispData['imageCoords'].to_numpy())[:,1]*10
    terrain         = dispData['terrain'].to_numpy()
    
    # %% 
    fig         = plt.figure(figsize=(11.69*2,8.27*2))
    
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=ele, azim=az)
    
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],c=terrain,marker='h',cmap=worldCoverCmap,vmin=10,vmax=100)
            
    
    
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('white')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='white')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    ax.xaxis._axinfo['color'] = (0,0,0,1)
    ax.yaxis._axinfo['color'] = (0,0,0,1)
    ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    plt.savefig('scatterWorldCover.png', transparent=True,dpi=1200)        
    
    # %% 
    fig         = plt.figure(figsize=(11.69*2,8.27*2))
    
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=ele, azim=az)
    
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],
               c=norm(dispData['OptNDVI'].to_numpy()),
               marker='.', s=50,
               cmap=lsc.from_list(None,['firebrick',
                                        'ivory',
                                        'lightsteelblue']))
    
             
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('white')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='white')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    ax.xaxis._axinfo['color'] = (0,0,0,1)
    ax.yaxis._axinfo['color'] = (0,0,0,1)
    ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    plt.savefig('scatterNDVI.png', transparent=True,dpi=1200)        
    # %% 
    fig         = plt.figure(figsize=(11.69*2,8.27*2))
    
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=ele, azim=az)
    
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],
               c=norm(dispData['predicted Label'].to_numpy()),
               marker='.', s=50,
               cmap=lsc.from_list(None,['firebrick',
                                        'ivory',
                                        'lightsteelblue']))
    
             
            
    ax.grid(False)
    
    ax.set_xticks([xmin,(xmin+((xmax-xmin)/2)),xmax])
    ax.set_yticks([ymin,(ymin+((ymax-ymin)/2)),ymax])    
    ax.set_zticks([zmin,(zmin+((zmax-zmin)/2)),zmax])
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontname(fontName)
        label.set_fontsize(tickS)
        label.set_color('white')
    
    
    ax.set_xlabel('Easting [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_ylabel('Northing [m]',fontsize=labelA,fontname=fontName,color='white')
    ax.set_zlabel('Height [m.a.s.l.]',fontsize=labelA,fontname=fontName,color='white')
    
    
    ax.xaxis.pane.set_edgecolor((1,1,1,1))
    ax.yaxis.pane.set_edgecolor((1,1,1,1))
    ax.zaxis.pane.set_edgecolor((1,1,1,1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis._axinfo['juggled'] = (1,0,2)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    ax.zaxis._axinfo['juggled'] = (0,2,1)
    
    ax.xaxis._axinfo['color'] = (0,0,0,1)
    ax.yaxis._axinfo['color'] = (0,0,0,1)
    ax.zaxis._axinfo['color'] = (0,0,0,1)
    plt.show()
    
    plt.savefig('scatterRFC.png', transparent=True,dpi=1200)    
    
    
#%% Terrain
saveDict = dict()
saveDict['lCMap']       = lCMap
saveDict['NDVI']        = s2MapBefore[:,:,-1]
saveDict['NDVI_Change'] = s2MapAfter[:,:,-1]-s2MapBefore[:,:,-1]
saveDict['RFC']         = resultsMap
saveDict['gT']          = GroundTruth

joblib.dump(saveDict, 'ResultsWindow_'+runningExtent+'.sav')

tSteps = 510
fig, axs = plt.subplots(1, 4, sharey=True,sharex=True)
axs[0].imshow(lCMap[ulY:lrY,ulX:lrX],cmap=worldCoverCmap,vmin=10,vmax=100)
axs[0].set_xticks([i * tSteps for i in range(len(xArray[ulX:lrX:tSteps]))])
axs[0].set_xticklabels(np.round(xArray[ulX:lrX:tSteps],2))
axs[0].set_yticks([i * tSteps for i in range(len(yArray[ulY:lrY:tSteps]))])
axs[0].set_yticklabels(np.round(yArray[ulY:lrY:tSteps],2))    
axs[1].imshow((s2MapAfter[ulY:lrY,ulX:lrX,-1]-s2MapBefore[ulY:lrY,ulX:lrX,-1]),vmin=-1,vmax=1,cmap='RdBu')
axs[2].imshow(s2MapAfter[ulY:lrY,ulX:lrX,-1],vmin=-1,vmax=1,cmap='RdBu')
axs[3].imshow(resultsMap[ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)


axs[0].set_title('Landcover')
axs[1].set_title('NDVI Change')
axs[2].set_title('NDVI After')
axs[3].set_title('RFC Result')
axs[0].grid(color='grey', linestyle='-', linewidth=0.5)
axs[1].grid(color='grey', linestyle='-', linewidth=0.5)
axs[2].grid(color='grey', linestyle='-', linewidth=0.5)
axs[3].grid(color='grey', linestyle='-', linewidth=0.5)

#%% Display all results:
if allResults:
    saveDictGraphSmall  = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperGraph/ResultsGraph_small_RFC21.sav')
    saveDictWindowSmall = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperWindow/ResultsWindow_small_RFC21.sav')
    
    saveDictWindow1617  = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperWindow/ResultsWindow_16_17_RFC21.sav')
    saveDictGraph1617   = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperGraph/ResultsGraph_16_17_RFC21.sav')
    
    saveDictGraphLarge  = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperGraph/ResultsGraph_large_RFC21.sav')
    saveDictWindowLarge = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperWindow/ResultsWindow_large_RFC21.sav')
    
    
    
    saveDictGraphMedium = joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperGraph/ResultsGraph_medium_RFC21.sav')
    saveDictWindowMedium= joblib.load('H:/_PhD/_Python/_up2dateVersions/CombineEEandSNAP/PaperWindow/ResultsWindow_medium_RFC21.sav')   
    # %%
    
    # gT          = pd.DataFrame.from_dict(
    #                   g.dictIdxKeySubset(keyList=['GroundTruth']),
    #                   orient='index').to_numpy()[:,0]
      
    # gTGRaph = np.reshape(gT,(g.rows,g.cols))
    
    tSteps = 255
    
    fig, axs = plt.subplots(4, 4, sharey=True,sharex=True,figsize=(8.27,11.69))
    axs[0,0].imshow(saveDictGraph1617['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[0,1].imshow(saveDictWindow1617['NDVI_Change'][ulY:lrY,ulX:lrX],vmin=-1,vmax=1,cmap='RdBu')
    axs[0,2].imshow(saveDictWindow1617['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[0,3].imshow(saveDictGraph1617['gT'][ulY:lrY,ulX:lrX,0],vmin=-1,vmax=1,cmap='RdBu')
    
    axs[0,0].set_xticks([i * tSteps for i in range(len(xArray[ulX:lrX:tSteps]))])
    axs[0,0].set_xticklabels(np.round(xArray[ulX:lrX:tSteps],2))
    axs[0,0].set_yticks([i * tSteps for i in range(len(yArray[ulY:lrY:tSteps]))])
    axs[0,0].set_yticklabels(np.round(yArray[ulY:lrY:tSteps],2)) 
    
    axs[0,0].set_title('RFC Result Graph',fontsize=labelA,fontname=fontName)
    axs[0,1].set_title('NDVI Change',fontsize=labelA,fontname=fontName)
    axs[0,2].set_title('RFC Result Window',fontsize=labelA,fontname=fontName)
    axs[0,3].set_title('Handdrawn Polygons',fontsize=labelA,fontname=fontName)

    axs[0,0].set_ylabel('Detection\n2017-02-08',fontsize=labelA,fontname=fontName)
    axs[1,0].set_ylabel('Largest Extent\n2018-11-18',fontsize=labelA,fontname=fontName)
    axs[2,0].set_ylabel('Medium Extent\n2018-03-08',fontsize=labelA,fontname=fontName)
    axs[3,0].set_ylabel('Smallest Extent\n2018-02-06',fontsize=labelA,fontname=fontName)
    
    axs[1,0].imshow(saveDictGraphLarge['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[1,1].imshow(saveDictWindowLarge['NDVI_Change'][ulY:lrY,ulX:lrX],vmin=-1,vmax=1,cmap='RdBu')
    axs[1,2].imshow(saveDictWindowLarge['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[1,3].imshow(saveDictGraphLarge['gT'][ulY:lrY,ulX:lrX,0],vmin=-1,vmax=1,cmap='RdBu')
  
    axs[2,0].imshow(saveDictGraphMedium['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[2,1].imshow(saveDictWindowMedium['NDVI_Change'][ulY:lrY,ulX:lrX],vmin=-1,vmax=1,cmap='RdBu')
    axs[2,2].imshow(saveDictWindowMedium['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[2,3].imshow(saveDictGraphMedium['gT'][ulY:lrY,ulX:lrX,0],vmin=-1,vmax=1,cmap='RdBu')
  
    axs[3,0].imshow(saveDictGraphSmall['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[3,1].imshow(saveDictWindowSmall['NDVI_Change'][ulY:lrY,ulX:lrX],vmin=-1,vmax=1,cmap='RdBu')
    axs[3,2].imshow(saveDictWindowSmall['RFC'][ulY:lrY,ulX:lrX],cmap='RdBu',vmin=-1,vmax=1)
    axs[3,3].imshow(saveDictGraphSmall['gT'][ulY:lrY,ulX:lrX,0],vmin=-1,vmax=1,cmap='RdBu')
  
    
    for i in range(4):
        for j in range(4):
            axs[i,j].grid(color='grey', linestyle='-', linewidth=0.5)
            for label in (axs[i,j].get_xticklabels() + axs[i,j].get_yticklabels()):
                label.set_fontname(fontName)
                label.set_fontsize(6)
                
    plt.savefig('scatterRFC.png', transparent=True,dpi=1200)       
    
    
    # %% Accuracy Metrics

    def calculateMetrics(results,referenceData):
        
        referenceData   = referenceData[results >= 0][:]
        results         = results[results >= 0][:]
        
        print('Bottom Line of Matrix')
        print('Total amount of Slide Pixels in Reference Data: '+str(len(referenceData[referenceData==1])))
        print('Total amount of noSlide Pixels in Reference Data: '+str(len(referenceData[referenceData==0])))
        print('Total amount of Pixels in Reference Data: '+str(len(referenceData)))
        print('')
        print('Right Side of Matrix')
        print('Total amount of Slide Pixels in Classified Data: '+str(len(results[results==1])))
        print('Total amount of noSlide Pixels in Classified Data: '+str(len(results[results==0])))
        print('Total amount of Pixels in Classified Data: '+str(len(results))) 
        print('')
        print('Slide Values')
        print('Amount of Slide Pixels that are Slides in Reference Data: '+str(sum(results[referenceData==1]==1)))
        print('Amount of noSlide Pixels that are Slides in Reference Data: '+str(sum(results[referenceData==1]==0)))
        print('')
        print('No Slide Values')
        print('Amount of Slide Pixels that are noSlides in Reference Data: '+str(sum(results[referenceData==0]==1)))  
        print('Amount of noSlide Pixels that are noSlides in Reference Data: '+str(sum(results[referenceData==0]==0)))
        print('')
        print('Overall Accuracy: '+str(np.round((sum(results[referenceData==1]==1)+sum(results[referenceData==0]==0))/len(referenceData)*100,2))+'%')
        print('')
        print('Producer Accuracy:')
        print('Slide: '+str(np.round((sum(results[referenceData==1]==1))/sum(referenceData==1)*100,2))+'%')
        print('noSlide: '+str(np.round((sum(results[referenceData==0]==0))/sum(referenceData==0)*100,2))+'%')
        print('')
        print('User Accuracy:')
        print('Slide: '+str(np.round((sum(results[referenceData==1]==1))/sum(results==1)*100,2))+'%')
        print('noSlide: '+str(np.round((sum(results[referenceData==0]==0))/sum(results==0)*100,2))+'%')       
    
    
    print('')
    print('----------------------------------')
    print('')
    print('Graph 16/17')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictGraph1617['RFC'][ulY:lrY,ulX:lrX],saveDictGraph1617['gT'][ulY:lrY,ulX:lrX,0])

    print('')
    print('----------------------------------')
    print('')
    print('Window 16/17')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictWindow1617['RFC'][ulY:lrY,ulX:lrX],saveDictWindow1617['gT'][ulY:lrY,ulX:lrX,0])    

    print('')
    print('----------------------------------')
    print('')
    print('Graph Small')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictGraphSmall['RFC'][ulY:lrY,ulX:lrX],saveDictGraphSmall['gT'][ulY:lrY,ulX:lrX,0])

    print('')
    print('----------------------------------')
    print('')
    print('Window Small')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictWindowSmall['RFC'][ulY:lrY,ulX:lrX],saveDictWindowSmall['gT'][ulY:lrY,ulX:lrX,0])        

    print('')
    print('----------------------------------')
    print('')
    print('Graph Medium')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictGraphMedium['RFC'][ulY:lrY,ulX:lrX],saveDictGraphMedium['gT'][ulY:lrY,ulX:lrX,0])

    print('')
    print('----------------------------------')
    print('')
    print('Window Medium')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictWindowMedium['RFC'][ulY:lrY,ulX:lrX],saveDictWindowMedium['gT'][ulY:lrY,ulX:lrX,0])    

    print('')
    print('----------------------------------')
    print('')
    print('Graph Large')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictGraphLarge['RFC'][ulY:lrY,ulX:lrX],saveDictGraphLarge['gT'][ulY:lrY,ulX:lrX,0])

    print('')
    print('----------------------------------')
    print('')
    print('Window Large')
    print('')
    print('----------------------------------')
    print('')
    calculateMetrics(saveDictWindowLarge['RFC'][ulY:lrY,ulX:lrX],saveDictWindowLarge['gT'][ulY:lrY,ulX:lrX,0])        
    # %%
    dataOrg     = pd.DataFrame.from_dict(
                      g.dictIdxKeySubset(keyList=[n for n in g.bandNames if 'Opt' in n]),
                      orient='index')    
    dataGraph   = pd.DataFrame.from_dict(
                      g.dictIdxKeySubset(keyList=[n for n in g.gBandNames if 'MeanOpt' in n]),
                      orient='index')    
    dataWindow  = pd.DataFrame.from_dict(
                      g.dictIdxKeySubset(keyList=[n for n in g.neighboursStatsBandNames if 'Mean3Opt' in n]),
                      orient='index')    
    
    gT          = pd.DataFrame.from_dict(
                      g.dictIdxKeySubset(keyList=['GroundTruth']),
                      orient='index').to_numpy()[:,0]  
    
    dataGraph.columns = dataOrg.columns
    dataWindow.columns = dataOrg.columns
    
    # %%
    
    pN = 10
    fontName = 'Tahoma'
    titleS   = '12'
    labelA   = '10'
    tickS    = '8' 
    
    
    
    fig, axs = plt.subplots(1, 4, sharey=True,figsize=(11.69,8.27))
    
    axs[0].set_title('Original Optical Data \nof Landslide Area',fontsize=labelA,fontname=fontName)
    dataOrg[gT == 1].boxplot(ax=axs[0],flierprops={'marker': '.', 'markersize': 0.1, 'markerfacecolor': 'grey'})
    axs[0].set_ylabel('Change Value',fontsize=labelA,fontname=fontName)
    
    axs[1].set_title('Graph Neighbourhood Statistics \nof Landslide Area',fontsize=labelA,fontname=fontName)
    dataGraph[gT == 1].boxplot(ax=axs[1],flierprops={'marker': '.', 'markersize': 0.1, 'markerfacecolor': 'grey'})
    
    
    axs[2].set_title('Window Neighbourhood Statistics \nof Landslide Area',fontsize=labelA,fontname=fontName)
    dataWindow[gT == 1].boxplot(ax=axs[2],flierprops={'marker': '.', 'markersize': 0.1, 'markerfacecolor': 'grey'})
    
    axs[3].set_title('General Optical Image Statistics',fontsize=labelA,fontname=fontName)
    dataOrg.boxplot(ax=axs[3],flierprops={'marker': '.', 'markersize': 0.1, 'markerfacecolor': 'grey'})
    axs[0].set_ylim([-0.5,0.5])
    
    
    
    for i in range(4):
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontname(fontName)
            label.set_fontsize(tickS)
            
    
    plt.savefig('boxPlot.png', transparent=True,dpi=1200)  
              
    # %%
    from subprocess import call
    from sklearn.tree import export_graphviz
    
    a = g.RFC['10'].estimators_[5]
    export_graphviz(a, 
                    out_file='tree.dot', 
                    feature_names = g.bandNames+g.neighboursStatsBandNames,
                    class_names = ['slide','noSlide'],
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    call(['C:/Program Files/Graphviz/bin/dot.exe', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
