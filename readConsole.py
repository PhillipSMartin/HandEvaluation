# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:33:55 2021

@author: sarab
"""
import globals
import engine
from typing import List

currentVectorIndex = -1

def getCurrentVector() -> List[int]:
    if currentVectorIndex >= 0:
        return globals.getVectorTableRow(currentVectorIndex).Vector.copy()
    else:
        print('Error: current vector has not been set')
        return None

def setCurrentVector(index : int):
    global currentVectorIndex
    try:
        index = int(index)
        if index not in range(globals.number_of_vectors):
            print(f'index {index} is out of range')
        else:
            currentVectorIndex = index
            globals.info(f'Setting current vector to {globals.getVectorName(index)}')
    except ValueError:
        print('Vector index must be an integer')
        
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
        setCurrentVector(args[1])
        # display info for new curent vector
        processI('I')

# W - write vector file
def processW(*args):
    fileName = ''
    if len(args) > 1:
        fileName = args[1]
    globals.saveVectorFile(fileName)

# F - display or edit feature
def processF(*args):
    feature = -1
    increment = 0
    
    if currentVectorIndex < 0:
        print('Must set current vector before editing feature')
        return
    
    try:
        if len(args) > 1:
            feature = globals.getFeatureNumber(args[1])
        if (len(args) > 2) and (feature >= 0):
            increment = int(args[2]) - getCurrentVector()[feature]
    except ValueError:
        print('Feature value must be an integer')
    
    if feature >= 0:
        # if no increment specified, display feature
        if increment == 0:
            print(f'{args[1]} = {getCurrentVector()[feature]}')
        # else change it
        else:
            oldVector = getCurrentVector()
            globals.setVector(currentVectorIndex, engine.adjustFeature(getCurrentVector(), feature, increment))
            globals.info(f'updated {args[1]} to {args[2]} ')
            globals.info(f'columns changed = {globals.compareVectors(oldVector, getCurrentVector())}')

# T - update threshold
def processT(*args):
    if currentVectorIndex < 0:
        print('Must set current vector before updating threshold')
        return
    
    try:
        if len(args) > 1:
             newThreshold = int(args[1])
    except ValueError:
        print('Threshold must be an integer')
        return
     
    globals.setThreshold(currentVectorIndex, newThreshold)
    # display info for new curent vector
    processI('I')
    
# D - delete vector
def processD(*args):
    global currentVectorIndex
    if currentVectorIndex < 0:
        print('Must set current vector before deleting')
        return
         
    if input(f'Delete vector {globals.getVectorName(currentVectorIndex)}? (y or n)').upper() == 'Y':
        globals.deleteVector(currentVectorIndex)
        globals.info(f'Vector {globals.getVectorName(currentVectorIndex)} deleted')
        currentVectorIndex = -1
    else:
        print('Aborting delete')

# S - save
def processS(*args):
    if currentVectorIndex < 0:
        print('Must use "v n" to set current vector to vector n before saving to a new name')
        return
    if len(args) < 2:
        print('Must specify a new name')
        return
    if args[1] in globals.getVectorNames():
        print(f'Vector {args[1]} already exists')
        return
    
    globals.setVectorTableRow(args[1], globals.getVectorTableRow(currentVectorIndex).copy(deep=False))
    globals.info(f'Current vector saved as {args[1]}')
    setCurrentVector(globals.number_of_vectors - 1)

# L - learn
def processL(*args):
    if currentVectorIndex < 0:
        print('Must use "v n" to set current vector to vector n before running learning algorithm')
        return
    newVector, newAccuracy = engine.learn(getCurrentVector(), 
        globals.getThresholds()[currentVectorIndex], globals.getTargets())
    globals.setVector(currentVectorIndex, newVector)
    globals.setAccuracy(currentVectorIndex, newAccuracy)
    
def selectVectors(*args):
    # select current vector or, if '*' specified, all vectors
    # return vector indices, thresholds and pointCount
    vectors = []
    vectorIndices = []
    if len(args) < 2:
        if currentVectorIndex in range(globals.number_of_vectors):
            vectors = [getCurrentVector()]
            vectorIndices = [currentVectorIndex]
            thresholds = [globals.getThresholds()[currentVectorIndex]]
            print(f'Running against {globals.getVectorNames()[currentVectorIndex]}')
    else:
        if args[1] == '*':
            vectors = globals.getVectors()
            vectorIndices = range(globals.number_of_vectors)
            thresholds = globals.getThresholds()
            print('Running against all vectors')
            
    if len(vectors) == 0:
        print('Invalid parameter')
        return list(), None, None
    else:
        return vectorIndices, thresholds, engine.calculatePointCount(globals.getFeatures(), vectors)
   
# A - accuracy
def processA(*args):
    vectorIndices, thresholds, pointCount = selectVectors(*args)
    if len(vectorIndices) > 0:
        accuracy = engine.calculateAccuracy(pointCount, thresholds, globals.getTargets())
              
        if len(vectorIndices) == 1:
            globals.setAccuracy(currentVectorIndex, accuracy[0])
            globals.info(f"{globals.vector_table.loc[globals.getVectorNames()[vectorIndices[0]], 'Accuracy']}")
        else:
            globals.vector_table['Accuracy'] = accuracy
            globals.info(f"{globals.vector_table[['Accuracy']]}")
  
# P - point count
def processP(*args):
    if currentVectorIndex < 0:
        print('Must set current vector before running this command')
        return
    
    pointCounts = engine.calculatePointCount(globals.getFeatures(), [getCurrentVector()])
    globals.info(engine.calculateSuccessesByPointCount(globals.targets, pointCounts))
    
 
# R - run
def processR(*args):
    vectorIndices, _, pointCount = selectVectors(*args)
    if len(vectorIndices) > 0:
        for n, v in enumerate(vectorIndices):
            name = globals.getVectorNames()[v]
            globals.vector_table.at[name, 
                ['Vul_threshold', 'Nv_threshold']] = engine.calculateThresholdsByVulnerability(globals.targets, pointCount[:, n])
 
        globals.info(globals.vector_table.iloc[vectorIndices][['Vul_threshold', 'Nv_threshold']])

# E - expectations
def processE():
    percentDifferences, expectations, analysis = engine.calculateExpectation(engine.calculatePointCount(globals.getFeatures(), globals.getVectors()), 
       globals.getThresholdsByVulnerability(), globals.getScores())
    globals.vector_table['Expectation'] = expectations
    globals.info(f'{percentDifferences * 100}% of boards were not flat')
    globals.info(globals.vector_table['Expectation'])
    globals.info("Analysis of lost boards:")
    globals.info(analysis.to_string())
        
def processH():
    print('Commands:')
    print('   v - list all Vectors')
    print('   v <n> - select Vector (by index) for editing or learning')
    print('   w [<filename>] - Write vector table as csv or xlsx file (depending on extension)\n')
    print('   i [*]- display Info for current vector or all vectors (*)')
    print('   f <text> [<weight>]- display weight for specified Feature in current vector')       
    print('      or update it to specified value')
    print('   t <threshold> - update classification Threshold for current vector')
    print('   s <name> - Save current vector as a new vector with specified name')
    print('   d - Delete current vector\n')
    print('   l - Learn (run learning algorithm against current vector)')
    print('   a [*] - calculate Accuracy for current vector (or for all vectors if * is specified)')
    print('      and save results')
    print('   r [*] - Run evaluation for current vector (or for all vectors if * is specified)')
    print('      to calculate vul_threshold, nv_threshold and save results')
    print('   p - display pointCount success rate table for curent vector')
    print('   e - calculate and save Expectations of all vectors by running simulated tournament\n')
    print('   c <text> - Comment - copy this comment to the log')
    print('   q - Quit')

# return False if we are to quit, else True
def processCommand(command) -> bool:  
    
    tokens = command.split(' ')
    operator = tokens[0].upper()
    
    # Q - quit
    if operator == 'Q':
        return False
    
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
    
    # T - update threshold
    elif operator == 'T':
        processT(*tokens)               
     
    # S - save current vector
    elif operator == 'S':
        processS(*tokens)
    
    # D - delete
    elif operator == 'D':
        processD(*tokens)
        
    # L - learn
    elif operator == 'L':
        processL(*tokens)
        
    # A - accuracy
    elif operator == 'A':
        processA(*tokens)
       
    # R - run
    elif operator == 'R':
        processR(*tokens)
       
    # R - run
    elif operator == 'P':
        processP()
        
    # E - calculate expectations
    elif operator == 'E':
        processE()
    
    # C - comment
    elif operator == 'C':
        globals.info(command[2:])
        
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
            
    

