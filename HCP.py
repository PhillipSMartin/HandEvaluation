# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:25:16 2021

@author: sarab
"""
import sys

import globals
from featureSets import featureSets
from readConsole import readConsole


# arguments are:
#   fileName of data file (required)
#   options:
#       -v:<fileName >- vector file (csv of xlsx)
#       -f:<featureSet> (defaults to balanced)
#       -d - debug mode
#       -q - quiet mode - no logging
#       -l:<text> - text to add to log to identify this run
#       -c:<text> - comma-delimited list of commands to execute
def main(*args):
    fileName = ''
    featureSet = 'balanced'
    vectorFile = ''
    log = True
    debug = False
    
    commands = ''
    comment = ''
    
    if len(args) > 0:
        for arg in args:
            if arg[0] != '-' and fileName == '':
                fileName = arg
            elif '-f:' in arg:
                featureSet = arg[3:].lstrip()
            elif '-v:' in arg:
                vectorFile = arg[3:].lstrip()
            elif '-d' in arg:
                debug = True
            elif '-q' in arg:
                log = False
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
    
    # initialize global variables
    if not globals.initialize(fileName, featureSet = featureSet, vectorFileName = vectorFile, log = log, debug = debug):
        print('Initialization failed - aborting')
        return
     
    readConsole(commands)
    

# if no arguments were passed, get console input

if len(sys.argv) > 1:
    main(*sys.argv[1:])

else:
    defaultFileName = '1000_deals.csv'
    fileName = input(f'Enter name of data file {defaultFileName}: ')
    if fileName == '':
        fileName = defaultFileName
    
    featureSet = ''
    for key in globals.featureSets.keys():
        if f'_{key}.' in fileName:
            featureSet = key
            break;
    
    if featureSet == '':
        featureSet = input(f'Enter name of feature set ({globals.DEFAULT_FEATURE_SET}): ')
        
    vectorFile = input('Enter name of vector file (or press enter for defaults): ' )
    
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
    if vectorFile != '':
        args.append(f'-v:{vectorFile}')
    if debug == 'Y':
        args.append('-d')
    if log == 'N':
        args.append('-q')
    main(*args)
    
