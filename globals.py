# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:24:14 2021

@author: sarab
"""
from featureSets import featureSets
from typing import List, Tuple, Dict, Any

import ast
import log
import numpy as np
import pandas as pd
import readData

global features
global deals
global vectors
global payoff_metric 

VALID_METRICS = ['expectation', 'accuracy']
DEFAULT_METRIC = 'expectation'
LOG_FILE = 'HCP.log'


def initializeMetric(metric : str) :
    global payoff_metric    
    payoff_metric = metric
    assert payoff_metric in VALID_METRICS, f'Metric must be in {VALID_METRICS}'
        
#-----------
#   Initialization - must be called at start of program
#-----------

# return False if method fails
def initialize(dataFileName : str, featureSet : str, vectorFileName : str='', metric = DEFAULT_METRIC, logFlag : bool = True, debugFlag : bool = False):
    # dataFileName is required
    # if vectorFileName is blank, we will load a default vector table
    global features
    global deals
    global vectors
     
    log.initializeLogging(LOG_FILE, logFlag, debugFlag)
       
    try:
        initializeMetric(metric)
        features = Features(featureSet, featureSets[featureSet])
        deals = Deals(dataFileName)
        vectors = Vectors(vectorFileName or features.getDefaultVectors())
    except AssertionError as e:
        print(e)
        return False
    
    return True

def cleanUp():
    log.closeLog()

#-----------
#   Vector table Methods
#-----------

class Features:
    _featureSetName = ''
    _featureSet = None
     
    number_of_features = 0
    
    def __init__(self, feature_set_name : str, feature_set : Dict[str, Any]):
        self._featureSetName = feature_set_name
        self._featureSet = feature_set
        self.number_of_features = len(self.getFeatureNames())
        
    def getFeatureSetName(self) -> str:
        return self._featureSetName
        
    def getFeatureNames(self) -> List[str]:
        return self._featureSet['feature_names']
        
    # takes an index and returns the feature name    
    def getFeatureName(self, feature_number : int) -> str:
        if feature_number >= 0 and feature_number < self.number_of_features:
            return self.getFeatureNames()[feature_number]
        else:
            print (f'Feature number {feature_number} does not exist')
            return ''       
    
    # takes a feature name and returns the index
    def getFeatureNumber(self, feature_name : str) -> int:
        feature_name = feature_name.upper().replace('X','x')
        if feature_name in self.getFeatureNames():
            return self.getFeatureNames().index(feature_name)
        else:
            print (f'Feature name {feature_name} does not exist')
            return -1
        
    def getDefaultVectors(self) -> pd.core.frame.DataFrame:
        return self._featureSet['default_vectors']
        
    def getFirstSpecialFeature(self) -> int:
        return self._featureSet['first_special_feature']
    
    def getFixedFeatures(self) -> List[int]:
        return self._featureSet['fixed_features']

    # sets of features constrained to have weights in (not strictly) ascending order
    #    because of high cards
    # example: Ax, AT, AJ, AQ, AK
    def getFeaturesOrderedByHighCards(self) -> List[int]:
        return self._featureSet['features_ordered_by_high_cards']
     
    # sets of features constrained to have weights in (not strictly) ascending order
    #    because of length
    # example: Ax, Axx, Axxx, Axxxx
    def getFeaturesOrderedByLength(self) -> List[int]:
        return self._featureSet['features_ordered_by_length']

class Deals:
    _deals = None
    
    number_of_deals = 0
    
    def __init__(self, fileName : str):
        self._deals = readData.readData(fileName)
        assert not self._deals.empty, f'Unable to read file {fileName}'
        self.number_of_deals = len(self._deals)
       
    def getFeatures(self) -> np.ndarray:
        return self._deals.values[:, - features.number_of_features:]   
    
    def getMakes3NT(self) -> np.ndarray:
        return np.expand_dims(self._deals['makes_3NT'], -1)
    
    def getScores(self) -> np.ndarray:
        return np.expand_dims(self._deals['score'], -1)
    
    def getVulnerabilities(self) -> List[int]:
        return self._deals[['vulnerable']].values
    
    # returns targets depending on which payoff metric we are using
    def getTargets(self) -> np.array:
         if payoff_metric == 'expectation':
             return self.getScores()
         elif payoff_metric == 'accuracy':
             return self.getMakes3NT()
         else:
             assert False, f'Unsupported metric {payoff_metric}'
 
class Vectors:
    _vectorTable = None
    _vectorFileName = ''
   
    number_of_vectors = 0
    
    def __init__(self, source): # source an be a fileName or a DataFrame
        if type(source) == pd.core.frame.DataFrame:
            self._vectorTable = source
        elif type(source) == str:
            self.readFile(source)
        else:
            assert False, 'Vectors class contructor must be passed a DataFrame or a fileName'
        self.number_of_vectors = len(self._vectorTable)
        assert self.number_of_vectors > 0, 'Vector table is empty'
         
    def readFile(self, fileName : str):
        self._vectorTable = readData.readFile(fileName, index=True)
        self._vectorFileName = fileName
        
        # must convert strings to vectors
        if not self._vectorTable.empty:
            vectors = []
            for n in range(len(self._vectorTable)):
                vectors.append(ast.literal_eval(self._vectorTable.iloc[n].Vector))
            self._vectorTable['Vector'] = vectors

    # lists vector indices and associate names
    def listVectors(self):
        for entry in enumerate(self._vectorTable.index):
            print(f'{entry[0]}: {entry[1]}')
            
    # returns a list of vector names
    def getVectorNames(self) -> List[str]:
        return self._vectorTable.index.to_list()
    
    # returns a list of vectors
    def getVectors(self) -> List[int]:
        return self._vectorTable.Vector.to_list()
    
    # returns an N x V matrix of thresholds for N deals and V vectors    
    def getThresholds(self, metric = 'default') -> Tuple[float]:
        if metric == 'default':
            metric = payoff_metric
            
        if metric == 'expectation':
            return self._vectorTable.Vul_threshold.values * deals.getVulnerabilities() + self._vectorTable.Nv_threshold.values * ~deals.getVulnerabilities() 
        elif metric == 'accuracy':
            return np.tile(self._vectorTable.Nv_threshold.values, (deals.number_of_deals, 1))
        else:
            print (f'Unsupported metric {metric}')
            return None
        
    # returns info for a givem vector 
    def getVectorTableRow(self, n : int) -> pd.core.series.Series:
        return self._vectorTable.iloc[n]
    
    def getVectorName(self, n : int) -> str:
        return self._vectorTable.index[n]
    
    # sets info in vector table
    def setVectorTableRow(self, name : str, row : pd.core.series.Series):
        global number_of_vectors
        self._vectorTable.loc[name] = row
        self.number_of_vectors = len(self._vectorTable)
    
    # set vul and nv thresholds 
    # if index is -1 set all thresholds that are not yet set (i.e., where Vul-threshold is 0)
    #   but leave all existing thresholds
    # thresholds is an array where the first column is the vul theshold and the second is the
    #   nv threshold. There must be a row for every vector (even those not being set)
    def setThresholds(self, thresholds : np.ndarray, index : int = -1): 
        if index >= 0:
            if index not in range(len(self._vectorTable)):
                print(f"Error - index {index} is out of range")
            else:
                self._vectorTable.at[self.getVectorName(index), ['Vul_threshold', 'Nv_threshold']] = thresholds
        else:
            if any(self._vectorTable.Vul_threshold == 0):
                # we set only the vul thresholds when setting them in bulk - nv threshold must be set explicitly by specifying the index
                self._vectorTable.at[self._vectorTable[self._vectorTable.Vul_threshold == 0].index, ['Vul_threshold']] = \
                    np.minimum(self._vectorTable.Nv_threshold.values, thresholds[:,0])[self._vectorTable.Vul_threshold == 0]
    
    # if index is -1, set all expectations
    # expectation can be a scalar or a list of floats   
    def setExpectation(self, expectation, index : int = -1): 
        if index > 0:
            if index not in range(len(self._vectorTable)):
                print(f"Error - index {index} is out of range")
            else:
                self._vectorTable.at[self.getVectorName(index), 'Expectation'] = expectation
        else:
            self._vectorTable['Expectation'] = expectation
        
    # if index is -1, set all accuracies
    # accuracy can be a scalar or a list of floats   
    def setAccuracy(self, accuracy : float, index : int = -1): 
        if index > 0:
            if index not in range(len(self._vectorTable)):
                print(f"Error - index {index} is out of range")
            else:
                self._vectorTable.at[self.getVectorName(index), 'Accuracy'] = accuracy
        else:
            self._vectorTable['Accuracy'] = accuracy
         
    def setPayoff(self, index : int, payoff : float):
         if payoff_metric == 'expectation':
             self.setExpectation(payoff, index = index)
         elif payoff_metric == 'accuracy':
             self.setAccuracy(payoff, index = index)
         else:
             print (f'Unsupported metric {payoff_metric}')
    
    # cannot set scores for individual vectors - must set all scores
    def setScores(self, scores : List[float]):
        self._vectorTable['Score'] = scores
    
    # manages table 
    def setVector(self, index : int, vector : List[int]):
        self._vectorTable.at[self.getVectorName(index), 'Vector'] = vector.copy()
    
    def deleteVector(self, index : int):
        self._vectorTable.drop([self._vectorTable.index[index]], inplace=True)
    
    # returns False if write failed
    def saveVectorFile(self, fileName : str = '') -> bool:
        if fileName == '':
            fileName = self._vectorFileName # this is file name we read from or last wrote to
            
        if fileName == '':
            log.info('No file name specifed for vector table')
            return False
        
        self._vectorFileName = fileName
        return readData.writeFile(self._vectorTable, fileName, index=True)
    
    # returns a list of columns that differ between two vectors
    def compareVectors(self, a : List[int], b : List[int]) -> List[str]:
        return [features.getFeatureNames()[index] for index, areEqual in enumerate([a_value == b_value for a_value, b_value in zip(a, b)]) if not areEqual]
              

if __name__ == "__main__":
    initialize('1000_deals_balanced.csv', log = False)    
