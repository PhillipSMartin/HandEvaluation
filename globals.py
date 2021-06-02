# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:24:14 2021

@author: sarab
"""
import numpy as np
import pandas as pd
import ast
import logging
import readData
import engine
from typing import List, Tuple, Callable
from featureSets import featureSets

global DEFAULT_FEATURE_SET
DEFAULT_FEATURE_SET = 'balanced'
DEFAULT_METRIC = 'expectation'
LOG_FILE = 'D:\sarab\Projects\HCP\HCP.log'


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
        logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG if debug else logging.INFO, force=True)

def info(message : str):
    if isLogging:
        logging.info(message)
    print(message)
 
def debug(message : str):
    if isLogging:
        logging.debug(message)
    if isDebugging:
        print(message)
 
def closeLog():
    if isLogging:
        logging.shutdown()
        
#-----------
#   Initialization Helper Methods
#-----------
        
# reads data from csv or xlsx file and initializes associated variables
# returns False if method fails
def initializeData(fileName : str) -> bool:
    global dataFileName
    global number_of_deals
    global features
    global makes_3NT
    global scores
    global vulnerabilities
    
    dataFileName = fileName
    deals = readData.readData(fileName)
    if not deals.empty:
        number_of_deals = len(deals)
        
        # extract features from last columns of data frame - 
        #   north features followed by south features
        features = deals.values[:, - number_of_features:]   
        
        # extract targets - two targets are possible: 'makes_3NT' and 'score'
        makes_3NT = np.expand_dims(deals['makes_3NT'], -1)
        scores =  np.expand_dims(deals['score'], -1)
        
        # extract vulnerabilities 
        vulnerabilities = deals[['vulnerable']].values
        return True
    else:
        return False

# reads vector_table from csv or xlsx file 
# vector_table is a DataFrame keyed by vector name
# Columns include
#   Vector - a list of  weights (point counts) for each feature
#   Vul_threshold - the point count for which we bid a vul game
#   Nv_threshold - the point count for which we bid a non-vul game
#   Excpectation - the net imps per board this vector expects to win
#   Accuracy - the percentage of deals this vector classifies correctly
#   Score - the score we receive on a simulated match against all other vectors
#
# returns False if method fails
def initializeVectorTable(fileName : str) -> bool:
    global vectorFileName
    global vector_table
    global number_of_vectors
    
    vector_table = pd.DataFrame()
    vectorFileName = fileName # save file name as default for when we write vector table  later
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

def initializeMetric(metric : str) -> bool:
    global payoff_metric    
    payoff_metric = metric
    if payoff_metric not in ['expectation', 'accuracy']:
        print('Metric must be "expectation" or "accuracy"') 
        return False
    return True                              
       
#-----------
#   Initialization - must be called at start of program
#-----------

# return False if method fails
def initialize(dataFileName : str, featureSet : str = DEFAULT_FEATURE_SET, vectorFileName : str='', metric = DEFAULT_METRIC, log : bool = True, debug : bool = False):
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
    
    if not initializeMetric(metric):
        return False
    
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
    
    if not initializeVectorTable(vectorFileName):
        return False
    
    return True

def cleanUp():
    closeLog()

#-----------
#   Data access
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
   
def getFeatures() -> np.ndarray:
    return features

# returns targets depending on which payoff metric we are using
def getTargets() -> np.array:
     if payoff_metric == 'expectation':
         return scores
     elif payoff_metric == 'accuracy':
         return makes_3NT
     else:
         print (f'Unsupported metric {payoff_metric}')
         return None   

def getPayoffFunction() -> Callable[[np.ndarray, np.ndarray, np.ndarray], List[int]]:
     if payoff_metric == 'expectation':
         return engine.calculateExpectation
     elif payoff_metric == 'accuracy':
         return engine.calculateAccuracy
     else:
         print (f'Unsupported metric {payoff_metric}')
         return None   

#-----------
#   Vector table Methods
#-----------

# lists vector indices and associate names
def listVectors():
    for entry in enumerate(vector_table.index):
        print(f'{entry[0]}: {entry[1]}')
        
# returns entire vector_table columns
# returns a list of vector names
def getVectorNames() -> List[str]:
    return vector_table.index.to_list()

# returns a list of vectors
def getVectors() -> List[int]:
    return vector_table.Vector.to_list()

# returns an N x V matrix of thresholds for N deals and V vectors    
def getThresholds(metric = 'default') -> Tuple[float]:
    if metric == 'default':
        metric = payoff_metric
        
    if metric == 'expectation':
        return vector_table.Vul_threshold.values * vulnerabilities + vector_table.Nv_threshold.values * ~vulnerabilities 
    elif metric == 'accuracy':
        return np.tile(vector_table.Nv_threshold.values, (number_of_deals, 1))
    else:
        print (f'Unsupported metric {metric}')
        return None
    
# returns info for a givem vector 
def getVectorTableRow(n : int) -> pd.core.series.Series:
    return vector_table.iloc[n]

def getVectorName(n : int) -> str:
    return vector_table.index[n]

# sets info in vector table
def setVectorTableRow(name : str, row : pd.core.series.Series):
    global number_of_vectors
    vector_table.loc[name] = row
    number_of_vectors = len(vector_table)

# set vul and nv thresholds 
# if index is -1 set all thresholds that are not yet set (i.e., where Vul-threshold is 0)
#   but leave all existing thresholds
# thresholds is an array where the first column is the vul theshold and the second is the
#   nv threshold. There must be a row for every vector (even those not being set)
def setThresholds(thresholds : np.ndarray, index : int = -1): 
    if index >= 0:
        if index not in range(len(vector_table)):
            print(f"Error - index {index} is out of range")
        else:
            vector_table.at[getVectorName(index), ['Vul_threshold', 'Nv_threshold']] = thresholds
    else:
        if any(vector_table.Vul_threshold == 0):
            # we set only the vul thresholds when setting them in bulk - nv threshold must be set explicitly by specifying the index
            vector_table.at[vector_table[vector_table.Vul_threshold == 0].index, ['Vul_threshold']] = \
                np.minimum(vector_table.Nv_threshold.values, thresholds[:,0])[vector_table.Vul_threshold == 0]

# if index is -1, set all expectations
# expectation can be a scalar or a list of floats   
def setExpectation(expectation, index : int = -1): 
    if index > 0:
        if index not in range(len(vector_table)):
            print(f"Error - index {index} is out of range")
        else:
            vector_table.at[getVectorName(index), 'Expectation'] = expectation
    else:
        vector_table['Expectation'] = expectation
    
# if index is -1, set all accuracies
# accuracy can be a scalar or a list of floats   
def setAccuracy(accuracy : float, index : int = -1): 
    if index > 0:
        if index not in range(len(vector_table)):
            print(f"Error - index {index} is out of range")
        else:
            vector_table.at[getVectorName(index), 'Accuracy'] = accuracy
    else:
        vector_table['Accuracy'] = accuracy
     
def setPayoff(index : int, payoff : float):
     if payoff_metric == 'expectation':
         setExpectation(payoff, index = index)
     elif payoff_metric == 'accuracy':
         setAccuracy(payoff, index = index)
     else:
         print (f'Unsupported metric {payoff_metric}')

# cannot set scores for individual vectors - must set all scores
def setScores(scores : List[float]):
    vector_table['Score'] = scores

# manages table 
def setVector(index : int, vector : List[int]):
    vector_table.at[getVectorName(index), 'Vector'] = vector.copy()

def deleteVector(index : int):
    vector_table.drop(vector_table.index[[index]], inplace=True)

# returns False if write failed
def saveVectorFile(fileName = ''):
    global vectorFileName
    if fileName == '':
        fileName = vectorFileName # this is file name we read from or last wrote to
        
    if fileName == '':
        info('No file name specifed for vector table')
        return False
    
    vectorFileName = fileName
    return readData.writeFile(vector_table, fileName, index=True)

# returns a list of columns that differ between two vectors
def compareVectors(a : List[int], b : List[int]) -> List[str]:
    return [feature_names[index] for index, areEqual in enumerate([a_value == b_value for a_value, b_value in zip(a, b)]) if not areEqual]

if __name__ == "__main__":
    initialize('1000_deals_balanced.csv', log = False)