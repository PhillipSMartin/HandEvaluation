# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:49:51 2021

@author: sarab
"""
import logging

isLogging = True
isDebugging = False

def initializeLogging(fileName : str, logFlag : bool, debugFlag : bool):
    global isLogging
    global isDebugging
    isLogging = logFlag
    isDebugging = debugFlag
    
    if (isLogging):
        logging.basicConfig(filename=fileName, level=logging.DEBUG if debug else logging.INFO, force=True)

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
