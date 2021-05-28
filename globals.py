# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:24:14 2021

@author: sarab
"""
import pandas as pd
import ast
import logging
import readData
from typing import List
from featureSets import featureSets

global DEFAULT_FEATURE_SET
DEFAULT_FEATURE_SET = 'balanced'

#-----------
#   Logging
#-----------
isLogging = True
isDebugging = False
dataFileName = ''
vectorFileName = ''

def initializeLogging(log : bool, debug : bool):
    global isLogging
    global isDebugging
    isLogging = log
    isDebugging = debug
    
    if (isLogging):
        logging.basicConfig(filename='D:\sarab\Projects\HCP\HCP.log', level=logging.DEBUG if debug else logging.INFO, force=True)

def info(message : str):
    if isLogging:
        logging.info(message)
    print(message)
 
def debug(message : str):
    if isLogging:
        logging.debug(message)
    if isDebugging:
        print(message)
        
#-----------
#   Initialization Helper Methods
#-----------
        
# reads data from file and initializes associated variables
# returns False if method fails
def initializeData(fileName : str) -> bool:
    global dataFileName
    global number_of_deals
    global features
    global targets
    global scores
    global vulnerabilities
    
    dataFileName = fileName
    deals = readData.readData(fileName)
    if not deals.empty:
        number_of_deals = len(deals)
        
        # extract features from last columns of data frame - 
        #   north features followed by south features
        features = deals.values[:, - number_of_features:]   
        
        # extract target from 'makes_3NT' column
        targets = deals[['makes_3NT']].values
        
        # extract scores if we bid 3NT and they don't - vulnerability is based on board number
        scores = deals[['score']].values
        
        # extract vulnerabilities 
        vulnerabilities = deals[['vulnerable']].values
        return True
    else:
        return False

# reads vector_table from csv or xlsx file 
# vector_table is a DataFrame keyed by vector name
# Columns include
#   Vector
#   Classification_threshold - minimum HCP for classifying deal as "makes game"
#   Vul_threshold - minimum HCP for bidding game if vul
#   Nv_threshold - minimum HCP for bidding game if non-vul
#   Accuracy
#   Score (average imp score per board when imp'd against all other vectors)
#
# returns False if method fails
def initializeVectorTable(fileName : str) -> bool:
    global vectorFileName
    global vector_table
    global number_of_vectors
    
    vector_table = pd.DataFrame()
    vectorFileName = fileName # save file name as default for when we save it later
    if fileName != '':
        vector_table = readData.readFile(fileName, index=True)
        if not vector_table.empty:
            vectors = []
            
            # must convert strings to vectors
            for n in range(len(vector_table)):
                vectors.append(ast.literal_eval(vector_table.iloc[n].Vector))
            vector_table['Vector'] = vectors
            
    # if no file read, load default vectors
    if vector_table.empty:
        vector_table = featureSets[feature_set]['default_vectors']
    number_of_vectors = len(vector_table)   
    if number_of_vectors == 0:
        print ('Vector table is empty')
        return False

    # make sure vector table is consistent with this feature set
    if len(vector_table.Vector[0]) != number_of_features:
        print(f'Error - vector_table is not consistent with feature set {feature_set}')
        return False
   
    return True                                         
       
#-----------
#   Initialization - must be called at start of program
#-----------

# return False if method fails
def initialize(dataFileName : str, featureSet : str = DEFAULT_FEATURE_SET, vectorFileName : str='', log : bool = True, debug : bool = False):
    # dataFileName is required
    # if vectorFileName is blank, we will load a default vector table
    global feature_set
    global feature_names
    global features_ordered_by_high_cards;
    global features_ordered_by_length;
    global number_of_features
    global first_special_feature
    global fixed_features
    
    initializeLogging(log, debug)
    
    feature_set = featureSet
    feature_names = featureSets[feature_set]['feature_names']
    number_of_features = len(feature_names)
    first_special_feature = featureSets[feature_set]['first_special_feature']
    fixed_features = featureSets[feature_set]['fixed_features']
    
    # sets of features constrained to have weights in (not strictly) ascending order
    #    because of high cards
    # example: Ax, AT, AJ, AQ, AK
    features_ordered_by_high_cards = featureSets[feature_set]['features_ordered_by_high_cards']
     
    # sets of features constrained to have weights in (not strictly) ascending order
    #    because of length
    # example: Ax, Axx, Axxx, Axxxx
    features_ordered_by_length = featureSets[feature_set]['features_ordered_by_length']
    
  
    if not initializeData(dataFileName):
        return False
    
    return initializeVectorTable(vectorFileName)
 

#-----------
#   Common methods
#-----------

# takes an index and returns the feature name    
def getFeatureName(feature_number : int) -> str:
    if feature_number >= 0 and feature_number < number_of_features:
        return feature_names[feature_number]
    else:
        print (f'Feature number {feature_number} does not exist')
        return ''       

# takes a feature name and returns the index
def getFeatureNumber(feature_name : str) -> int:
    feature_name = feature_name.upper().replace('X','x')
    if feature_name in feature_names:
        return feature_names.index(feature_name)
    else:
        print (f'Feature name {feature_name} does not exist')
        return -1
    
def getFeatures():
    return features

def getTargets():
    return targets

def getScores():
    return scores

# lists vector indices and associate names
def listVectors():
    for entry in enumerate(vector_table.index):
        print(f'{entry[0]}: {entry[1]}')
        
# returns vector_table columns
def getVectorNames():
    return vector_table.index.to_list()

def getVectors():
    return vector_table.Vector.to_list()
    
def getThresholds():
    return vector_table.Classification_threshold.to_list()

def getThresholdsByVulnerability():
    return vector_table.Vul_threshold.values * vulnerabilities + vector_table.Nv_threshold.values * ~vulnerabilities 

def getVectorTableRow(n : int) -> pd.Series:
    return vector_table.iloc[n]

def getVectorName(n : int) -> str:
    return vector_table.index[n]

def setVectorTableRow(name : str, row : pd.Series):
    global number_of_vectors
    vector_table.loc[name] = row
    number_of_vectors = len(vector_table)

def setVector(index : int, vector : List[int]):
    vector_table.at[getVectorName(index), 'Vector'] = vector.copy()

def setThreshold(index : int, threshold : int):
    vector_table.at[getVectorName(index), 'Classification_threshold'] = threshold
    
def setAccuracy(index : int, accuracy : float):
    vector_table.at[getVectorName(index), 'Accuracy'] = accuracy
    
def deleteVector(index : int):
    vector_table.drop(vector_table.index[[index]], inplace=True)

def saveVectorFile(fileName = ''):
    global vectorFileName
    if fileName == '':
        fileName = vectorFileName
        
    if fileName == '':
        info('No file name specifed for vector table')
        return False
    
    vectorFileName = fileName
    return readData.writeFile(vector_table, fileName, index=True)

def compareVectors(a : List[int], b : List[int]) -> List[str]:
    return [feature_names[index] for index, areEqual in enumerate([a_value == b_value for a_value, b_value in zip(a, b)]) if not areEqual]

if __name__ == "__main__":
    initialize('1000_deals_processed.csv', '', log = False)