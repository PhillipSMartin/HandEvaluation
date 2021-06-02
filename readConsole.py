# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:33:55 2021

@author: sarab
"""
import numpy as np
import globals
import engine
from typing import List

currentVectorIndex = -1
vectorFileIsDirty = False

def getCurrentVector() -> List[int]:
    if currentVectorIndex >= 0:
        return globals.getVectorTableRow(currentVectorIndex).Vector.copy()
    else:
        print('Error: current vector has not been set')
        return None

# returns False if not set
def setCurrentVector(index : int) -> bool:
    global currentVectorIndex
    try:
        index = int(index)
        if index not in range(globals.number_of_vectors):
            print(f'index {index} is out of range')
            return False
        else:
            currentVectorIndex = index
            globals.info(f'Setting current vector to {globals.getVectorName(index)}')
            return True
    except ValueError:
        print('Vector index must be an integer')
        return False
     
# select current vector or, if '*' specified for second argument, all vectors
# return vector indices, thresholds and pointCount
def selectVectors(*args):
    vectors = []
    vectorIndices = []
    
    # select threshold based on use
    metric = 'default' 
    if args[0].upper() in ['A']:
        metric = 'accuracy'
    if args[0].upper() in ['X', 'R']:
        metric = 'expectation'
    
    # current vector only
    if len(args) < 2:
        if currentVectorIndex < 0:
            print('Error: current vector has not been set') 
        else:
            vectors = [getCurrentVector()]
            vectorIndices = [currentVectorIndex]
            thresholds = globals.getThresholds(metric)[:,currentVectorIndex].reshape(-1,1)
            print(f'Running against {globals.getVectorNames()[currentVectorIndex]}')
            
    # all vectors
    else:
        if args[1] == '*':
            vectors = globals.getVectors()
            vectorIndices = range(globals.number_of_vectors)
            thresholds = globals.getThresholds(metric)
            print('Running against all vectors')
            
    if len(vectors) == 0:
        print('Invalid parameter')
        return list(), None, None
    else:
        return vectorIndices, thresholds, engine.calculatePointCount(globals.getFeatures(), vectors)

#-----------
#   Console commands
#-----------
       
# I - display info    
def processI(*args):
    vectorNumbers = []
    if currentVectorIndex >= 0:
        vectorNumbers = [currentVectorIndex]
        
    if len(args) > 1:
        if (args[1] == '*'):
            vectorNumbers = range(0, globals.number_of_vectors)
                 
    for n in vectorNumbers:
        row = globals.vector_table.iloc[n]
        globals.info(f'*** Info for {globals.getVectorNames()[n]} ***\n{row}\n')

# V - list all vectors or set current vector
def processV(*args):
    # if no argument specified, list all vectors
    if len(args) == 1:
        globals.listVectors()
    # else change current vector
    else:
        if setCurrentVector(args[1]):
            # display info for new curent vector
            processI('I')

# W - write vector file
def processW(*args):
    global vectorFileIsDirty
    fileName = ''
    if len(args) > 1:
        fileName = args[1]
    vectorFileIsDirty =  not globals.saveVectorFile(fileName)

# F - display or edit feature
def processF(*args):
    global vectorFileIsDirty
    feature = -1
    increment = 0
    
    if currentVectorIndex < 0:
        print('Must set current vector before editing feature')
        return
    
    try:
        if len(args) > 1:
            feature = globals.getFeatureNumber(args[1])
        if (len(args) > 2) and (feature >= 0):
            if feature in globals.fixed_features:
                print('cannot modify fixed feature')
            else:
                increment = int(args[2]) - getCurrentVector()[feature]
    except ValueError:
        print('Feature value must be an integer')
    
    if feature >= 0:
        # if no increment specified, display feature
        if increment == 0:
            print(f'{args[1]} = {getCurrentVector()[feature]}')
        # else change it
        else:
            vectorFileIsDirty = True
            oldVector = getCurrentVector()
            globals.setVector(currentVectorIndex, engine.adjustFeature(getCurrentVector(), feature, increment))
            globals.info(f'updated {args[1]} to {args[2]} ')
            globals.info(f'columns changed = {globals.compareVectors(oldVector, getCurrentVector())}')

# S - save
def processS(*args):
    global vectorFileIsDirty
    if currentVectorIndex < 0:
        print('Must use "v n" to set current vector to vector n before saving to a new name')
        return
    if len(args) < 2:
        print('Must specify a new name')
        return
    if args[1] in globals.getVectorNames():
        print(f'Vector {args[1]} already exists')
        return
    
    vectorFileIsDirty = True
    globals.setVectorTableRow(args[1], globals.getVectorTableRow(currentVectorIndex).copy(deep=False))
    globals.info(f'Current vector saved as {args[1]}')
    setCurrentVector(globals.number_of_vectors - 1)
    
# D - delete vector
def processD(*args):
    global currentVectorIndex
    if currentVectorIndex < 0:
        print('Must set current vector before deleting')
        return
         
    if input(f'Delete vector {globals.getVectorName(currentVectorIndex)}? (y or n): ').upper() == 'Y':
        globals.deleteVector(currentVectorIndex)
        globals.info(f'Vector {globals.getVectorName(currentVectorIndex)} deleted')
        currentVectorIndex = -1
    else:
        print('Aborting delete')
        
# M - change metric
def processM(*args):
    if len(args) > 1:
        newMetric = args[1].upper();
        if newMetric == 'A':
            return globals.initializeMetric('accuracy')
        elif newMetric == 'E':
            return globals.initializeMetric('expectation')
    print('must specify "a" or "e" for new metric')
    return False
            
# A - accuracy
def processA(*args):
    global vectorFileIsDirty
    vectorIndices, thresholds, pointCounts = selectVectors(*args)
    
    if any(globals.vector_table.Vul_threshold[vectorIndices] == 0):
        print('Must run T command to update thresholds before calculating accuracy')
        return
    
    if len(vectorIndices) > 0:
        accuracy = engine.calculateAccuracy(pointCounts, thresholds, globals.makes_3NT)
        vectorFileIsDirty = True
        if len(vectorIndices) == 1:
            globals.setAccuracy(accuracy[0], index = currentVectorIndex)
            globals.info(f"{globals.vector_table.loc[globals.getVectorNames()[vectorIndices[0]], 'Accuracy']}")
        else:
            globals.setAccuracy(accuracy)
            globals.info(f"{globals.vector_table[['Accuracy']]}")

# X - expectations
def processX(*args):
    global vectorFileIsDirty
    vectorIndices, thresholds, pointCounts = selectVectors(*args)
    
    if any(globals.vector_table.Vul_threshold[vectorIndices] == 0):
        print('Must run T command to update thresholds before calculating expectations')
        return
    
    if len(vectorIndices) > 0:
        expectation = engine.calculateExpectation(pointCounts, thresholds, globals.scores)
        vectorFileIsDirty = True
        if len(vectorIndices) == 1:
            globals.setExpectation(expectation[0], index = currentVectorIndex)
            globals.info(f"{globals.vector_table.loc[globals.getVectorNames()[vectorIndices[0]], 'Expectation']}")
        else:
            globals.setExpectation(expectation)
            globals.info(f"{globals.vector_table[['Expectation']]}")
        
# T - update thresholds
#   if two numbers are specified, update thresholds for current vector
#   if * is specified, calculate thesholds for all vectors who currently have thresholds of 0
#   if X is specified, zero out all vul thresholds for all vectors
def processT(*args):
    global vectorFileIsDirty
    vectorFileIsDirty = True
    
    # T *: calculate and save all thesholds
    if len(args) > 1:
        if (args[1] == '*'):
            _, _, pointCounts = selectVectors(*args)
            globals.setThresholds(engine.calculateAllThresholdsByVulnerability(
                globals.makes_3NT, pointCounts))
            globals.info(globals.vector_table[['Vul_threshold', 'Nv_threshold']])
            return
    
        # T X: zero out all thresholds
        if (args[1].upper() == 'X'):
            if 'Y' == input('Zero out vul thresholds for all vectors (y or N): ').upper():
                globals.vector_table.Vul_threshold = 0
                globals.info(globals.vector_table[['Vul_threshold', 'Nv_threshold']])
            else:
                print('Aborting T command')
            return
        
    # T <float> <float>: save these thresholds for current vector
    if currentVectorIndex < 0:
        print('Must set current vector before updating threshold')
        return
    
    try:
        if len(args) > 2:
             globals.setThresholds([int(x) for x in args[1:3]], currentVectorIndex)
             # display info for new curent vector
             processI('I')
        else:
            print('Must specify both a vul and a non-vul threshold')
    except ValueError:
        print('Threshold must be an integer')
        return
 
# P - point count
def processP(*args):
    if currentVectorIndex < 0:
        print('Must set current vector before running this command')
        return
    
    pointCount = engine.calculatePointCount(globals.getFeatures(), [getCurrentVector()])
    globals.info(engine.calculateSuccessesByPointCount(
        globals.makes_3NT, pointCount))

# L - learn
def processL(*args):
    if currentVectorIndex < 0:
        print('Must set current vector before running learning algorithm')
        return
     
    if globals.vector_table.Vul_threshold[currentVectorIndex] == 0:
        print('Must run T command to update thresholds before calculating expectations')
        return
    
    global vectorFileIsDirty
    vectorFileIsDirty = True
    _, thresholds, pointCounts = selectVectors(*args)
    
    globals.info(f'Using payoff_metric={globals.payoff_metric}')
    newVector, newPayoff = engine.learn(getCurrentVector(), pointCounts, thresholds,
          globals.getTargets(), globals.getPayoffFunction())
    globals.setVector(currentVectorIndex, newVector)
    globals.setPayoff(currentVectorIndex, newPayoff)

# R - run
def processR(*args):
    global vectorFileIsDirty
    vectorFileIsDirty = True
    _, thresholds, pointCounts = selectVectors('R', '*')
    percentDifferences, scores = engine.runSimulatedMatch(pointCounts, thresholds, globals.scores)                                                         
    globals.setScores(scores)
    globals.info(f'{percentDifferences * 100}% of boards were not flat')
    globals.info(globals.vector_table.Score)
        
def processH():
    print('Commands:')
    print('   v - list all Vectors')
    print('   v <n> - select Vector (by index) for editing or learning')
    print('   w [<filename>] - Write vector table as csv or xlsx file (depending on extension)\n')
    print('   i [*]- display Info for current vector or all vectors (*)')
    print('   f <text> [<weight>]- display weight for specified Feature in current vector')       
    print('      or update it to specified value')
    print('   s <name> - Save current vector as a new vector with specified name')
    print('   d - Delete current vector\n')
    print('   m <a|e>- Change metric')
    print('   a [*] - calculate Accuracy for current vector (or for all vectors if * is specified)')
    print('      and save results')
    print('   x [*] - calculate Expectation for current vector (or for all vectors if * is specified)')
    print('      and save results')
    print('   t <vul threshold> <nv threshold> - update thresholds for current vector')
    print('   t [*] - calculate thresholds for all vectors whose thresholds are currently zero')
    print('        and save results')
    print('   t x - zero out all thresholds so we can recalculate them')
    print('   p - display pointCount success rate table for curent vector\n')
    print('   l - Learn (run learning algorithm against current vector)')
    print('   r - Run simulated match and store results\n')
    print('   c <text> - Comment - copy this comment to the log')
    print('   q - Quit')

def processQ():
    if vectorFileIsDirty:
        if 'N' != input('Save vector file before quitting? (Y or n): ').upper():
            print('Use "w" command to save file')
            return True
    return False
    
# return False if we are to quit, else True
def processCommand(command) -> bool:  
    
    tokens = command.split(' ')
    operator = tokens[0].upper()
    
    # Q - quit
    if operator == 'Q':
        return processQ()
    
    # V - list all vectors or set current vector
    elif operator == 'V':
        processV(*tokens)
   
    # W - write vector file
    elif operator == 'W':
        processW(*tokens)
    
    # I - display info
    elif operator == 'I':
        processI(*tokens)
    
    # F - display or edit feature
    elif operator == 'F':
        processF(*tokens)               
     
    # S - save current vector
    elif operator == 'S':
        processS(*tokens)
    
    # D - delete
    elif operator == 'D':
        processD(*tokens)
   
    # M - metric
    elif operator == 'M':
        processM(*tokens)
        
    # A - accuracy
    elif operator == 'A':
        processA(*tokens)
    
    # T - update threshold
    elif operator == 'T':
        processT(*tokens)               
       
    # R - run
    elif operator == 'P':
        processP()
        
    # L - learn
    elif operator == 'L':
        processL(*tokens)
       
    # R - run
    elif operator == 'R':
        processR(*tokens)
    
    # C - comment
    elif operator == 'C':
        globals.info(command[2:])
        
    # %X - calculate expections 
    elif operator == 'X':
        processX(*tokens)
        
    # H - list available commands
    else:
        processH()
        
    return True

# executes specified commands if passed - otherwise waits for input from console
def readConsole(commands : str):
    # if a command list is specified, execute commands in that list
    if commands != '':
        command_list = commands.split(',')
        for command in command_list:
            # return of False signals we are done
            if not processCommand(command):
               return
           
    # no command list, get console input
    else:
         command = input('Enter h for help, q to quit, or any command:\n')
         while (processCommand(command)):
             command = input('Enter h for help, q to quit, or any command:\n')
            
'''
globals.initialize('1000_deals_balanced.csv','balanced')
currentVectorIndex = 0
processT('T', '*')
'''
