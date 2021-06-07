# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:25:16 2021

@author: sarab
"""
from featureSets import featureSets
from readConsole import readConsole
from typing import List
from traceback import print_tb

import globals
import log
import os
import sys

DEFAULT_FILE_NAME = '1000_deals_balanced.csv'
DEFAULT_FEATURE_SET = 'balanced'

# arguments are:
#   fileName of data file (required)
#   options:
#       -v:<fileName >- vector file (csv of xlsx)
#       -f:<featureSet> (defaults to balanced)
#       -m:<payoff metric>('expectation' or 'accuracy' - defaults to 'expecation')
#       -d - debug mode
#       -q - quiet mode - no logging
#       -l:<text> - text to add to log to identify this run
#       -c:<text> - comma-delimited list of commands to execute
def go(*args : List[str]):
    fileName = ''
    featureSet = 'balanced'
    metric = 'expectation'
    vectorFile = ''
    logFlag = True
    debugFlag = False
    
    commands = ''
    comment = ''
    
    if len(args) > 0:
        for arg in args:
            if arg[0] != '-' and fileName == '':
                fileName = arg
            elif '-f:' in arg:
                featureSet = arg[3:].lstrip()
            elif '-m:' in arg:
                metric = arg[3:].lstrip()
            elif '-v:' in arg:
                vectorFile = arg[3:].lstrip()
            elif '-d' in arg:
                debugFlag = True
            elif '-q' in arg:
                logFlag = False
            elif '-l:' in arg and comment == '':
                comment = arg[3:].lstrip()
            elif '-c:' in arg and commands == '':
                commands = arg[3:]
            else:
                print(f'Unknown parameter: {arg}')
                return
        
    if fileName == '':
        print('Error - no data file specified')
        return
    
    if featureSet not in featureSets.keys():
        print(f'Error - feature set {featureSet} not found')
        return
 
    try:
        # initialize global variables
        if not globals.initialize(fileName, featureSet = featureSet or DEFAULT_FEATURE_SET, vectorFileName = vectorFile, metric = metric, logFlag = logFlag, debugFlag = debugFlag):
            print('Initialization failed - aborting')
        else:
             readConsole(commands)
             
    except Exception as e:
        log.info(e)
        print_tb(e.__traceback__)
        
    finally:
        globals.cleanUp()

def buildArgs() -> List[str]:
    fileNameOK = False
    while not fileNameOK:
        fileName = input(f'Enter name of data file {DEFAULT_FILE_NAME}: ')
        if fileName == '':
            fileName = DEFAULT_FILE_NAME
        fileNameOK = os.path.isfile(fileName)
        if not fileNameOK:
            print (f'File {fileName} does not exists')
    
    featureSet = ''
    for key in featureSets.keys():
        if f'_{key}.' in fileName:
            featureSet = key
            break;
 
    while featureSet == '':
        featureSet = input(f'Enter name of feature set ({DEFAULT_FEATURE_SET}): ')
        if featureSet not in featureSets.keys():
            print(f'{featureSet} is not a valid feature set')
            print(f'valid feature sets are {featureSets.keys()}')
            featureSet = ''
       
    metric = ''
    while metric == '':
        metric = input('Enter name of metric ("a" for accuracy, "E" for expectation): ')
        if metric.upper() == 'A':
            metric = 'accuracy'
        elif metric.upper() == 'E' or metric == '':
            metric = 'expectation'
        else:
            print('Invalid metric')
            metric = ''
 
    vectorFileOK = False
    while not vectorFileOK:      
        vectorFile = input('Enter name of vector file (or press enter for defaults): ')
        if vectorFile == '':
            vectorFileOK = True
        else:
            vectorFileOK = os.path.isfile(vectorFile)
            if not vectorFileOK:
                print (f'File {vectorFile} does not exists')
              
    debug = 'x'
    while debug not in 'YN':
        debug = input('Debug? (y or N): ').upper()
        if debug == '':
            debug = 'N'
            
    log = 'x'
    while log not in 'YN':
        log = input('Log? (y or N): ').upper()
        if log == '':
            log = 'N'
            
    args = [fileName]
    if featureSet != '':
        args.append(f'-f:{featureSet}')
    if metric != '':
        args.append(f'-m:{metric}')
    if vectorFile != '':
        args.append(f'-v:{vectorFile}')
    if debug == 'Y':
        args.append('-d')
    if log == 'N':
        args.append('-q')
    return args
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        go(*sys.argv[1:])
    
    # if no arguments were passed, get console input
    else:
        go(*buildArgs())
