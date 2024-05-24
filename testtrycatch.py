# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:16:37 2021

@author: sarab
"""

'''
while True:
    try:
        print('In try block')
        x = input('> ')[0]
        if x[0] == 'q':
            break
        elif x[0] == 'e':
            y = 1 / 0
    except Exception as e:
        print(e)
        break

print ('exited loop')
'''
import os
CWD = os.path.dirname(os.path.realpath(__file__))
print( os.path.join(CWD, 'HCP.py'))
        