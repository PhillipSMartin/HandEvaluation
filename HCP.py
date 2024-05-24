# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:25:16 2021

@author: sarab
"""
from featureSets import featureSets
from readConsole import readConsole
from typing import List
from traceback import print_tb

import argparse
import globals
import log
import os
import sys

DEFAULT_FILE_NAME = '1000_deals_balanced.csv'
DEFAULT_FEATURE_SET = 'balanced'
DEFAULT_METRIC = 'expectation'

# arguments are:
#   fileName of data file (required)
#   options:
#       -v:<fileName >- vector file (csv of xlsx)
#       -f:<featureSet> (defaults to balanced)
#       -m:<payoff metric>('expectation' or 'accuracy' - defaults to 'expectation')
#       -d - debug mode
#       -q - quiet mode - no logging
#       -c:<text> - comma-delimited list of commands to execute
def go(args):
   
    try:
        # initialize global variables
        if not globals.initialize(args.fileName, 
                featureSet = args.featureSet,
                vectorFileName = args.vectors, 
                metric = args.metric, 
                logFlag = not args.quiet,
                debugFlag = args.debug,
                ignoreTens = args.ignoreTens):
            print('Initialization failed - aborting')
        else:
             readConsole(args.commands)
             
    except Exception as e:
        log.info(e)
        print_tb(e.__traceback__)
        
    finally:
        globals.cleanUp()

def buildArgs() -> List[str]:
    fileNameOK = False
    while not fileNameOK:
        fileName = input(f'Enter name of data file {DEFAULT_FILE_NAME}: ') or DEFAULT_FILE_NAME
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
            
    ignoreTens = 'x'
    while ignoreTens not in 'YN':
        ignoreTens = input('Ignore tens? (y or N): ').upper() or 'N'
       
    metric = ''
    while metric == '':
        metric = input('Enter name of metric ("a" for accuracy, "E" for expectation): ') or 'E'
        if metric.upper() == 'A':
            metric = 'accuracy'
        elif metric.upper() == 'E':
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
              
    debugFlag = 'x'
    while debugFlag not in 'YN':
        debugFlag = input('Debug? (y or N): ').upper() or 'N'
            
    logFlag = 'x'
    while logFlag not in 'YN':
        logFlag = input('Log? (y or N): ').upper() or 'N'
            
    args = [fileName]
    if featureSet != '':
        args.extend(['-f', featureSet])
    if metric != '':
        args.extend(['-m', metric])
    if vectorFile != '':
        args.extend(['-v', vectorFile])
    if ignoreTens == 'Y':
        args.append('-t')
    if debugFlag == 'Y':
        args.append('-d')
    if logFlag == 'N':
        args.append('-q')
    return args
   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'HCP evaluator')
    parser.add_argument('fileName', 
                        help='name of file containsing deals for learning or testing')
    parser.add_argument('-v', '--vectors', 
                        help='name of file with vectors to test - will use default vectors if omitted')
    parser.add_argument('-f', '--featureSet', 
                        choices=('flat', 'balanced', 'semi-balanced'),
                        default=DEFAULT_FEATURE_SET, 
                        help='name of feature set (flat (4432 o4 4433), balanced (default, flat + 5332 or 5422), semi-balanced (balanced + 6332))')
    parser.add_argument('-t', '--ignoreTens', 
                        action='store_true', 
                        help='Force tens to have a value of zero')
    parser.add_argument('-m', '--metric', 
                        choices=('accuracy', 'expectation'),
                        default=DEFAULT_METRIC, 
                        help='Payoff metric (expectation is default)')
    parser.add_argument('-c', '--commands', 
                        help='comma-delimited list of commands to execute (interactive mode if omitted)')
    parser.add_argument('-d', '--debug', 
                        action='store_true', 
                        help='debug mode')
    parser.add_argument('-q', '--quiet', 
                        action='store_true', 
                        help='quiet mode (no log file created)')
        
    if len(sys.argv) > 1:
        go(parser.parse_args())
    
    # if no arguments were passed, get console input
    else:
        go(parser.parse_args(buildArgs()))
