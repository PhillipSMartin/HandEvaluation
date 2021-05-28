# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:44:02 2021

@author: sarab
"""
import pandas as pd
import globals

# append .csv to filename if not already there and if not an xlsx file
def fixFileName(fileName : str) -> str:
    if len(fileName) > 4:
       if fileName[-4:] == '.csv':
           return fileName
       elif fileName[-5:] == '.xlsx':
           return fileName
    return fileName + '.csv'
    

# save DataFrame csv or xlsx file
# returns True if write worked
def writeFile(dataFrame : pd.core.frame.DataFrame, fileName : str, index=False) -> bool:
    fileName = fixFileName(fileName)
    
    try:
        globals.info(f'Writing {fileName}')
        if fileName[-4:] == '.csv':
           dataFrame.to_csv(fileName, index=index)
           return True
           
        elif fileName[-5:] == '.xlsx':
           dataFrame.to_excel(fileName, index=index)
           return True
       
    except IOError as e:
        globals.info(e)
        
    return False
    
# read csv or xlsx file into DataFrame
# returns an empty DataFrame if read fails
def readFile(fileName : str, index=False) ->  pd.core.frame.DataFrame:
    dataFrame = pd.DataFrame()
    fileName = fixFileName(fileName)
    
    try:
        if fileName[-4:] == '.csv':
            globals.info(f'Reading {fileName}')
            if index:
                dataFrame = pd.read_csv(fileName, index_col=0)
            else:
                dataFrame = pd.read_csv(fileName)
            
        elif fileName[-5:] == '.xlsx':
            globals.info(f'Reading {fileName}')
            if index:
                dataFrame = pd.read_excel(fileName)
            else:
                dataFrame = pd.read_excel(fileName, index_col=0)
         
    except IOError as e:
        globals.info(e)
        
    return dataFrame

# Reads data from file created by DealMaster Pro
#
# The file must be generated to compare 3NT by North with 2NT by North
# The north and south hands are in the columns 'north' and 'south'
# The column a_makedown is positive if 3NT by North makes, negative if it does not.
# In addition, the fields nv_imps and vul_imps will contain the results of the 3NT/2NT
# comparison, that is, the imps won or lost by playing 3NT at your table
# when 2NT is played at the other table
#
# Returns a DataFrame with additional columns needed for analysis.
# Returns an empty DataFrame if method fails
# 
def readData(fileName : str) -> pd.core.frame.DataFrame:
    # read file
    deals = readFile(fileName)
    if deals.empty:
        return deals
  
    # if we have a column named makes_3NT, we have processed the data and saved it
    # if not, we must process it now - we can save the file so we won't need to process it again
    if 'makes_3NT' not in deals:
        # drop unneeded columns and rows
        deals.drop(['anal_num', 'a_cont_by', 'a_nv_scor', 'a_vul_scor', 'b_cont_by',
            'b_nv_scor', 'b_vul_scor', 'nv_netscor', 'v_netscor', 'east', 'west', 'n_hcp',
            'e_hcp', 's_hcp', 'w_hcp'], 
            axis = 1, inplace = True)
        deals.drop(deals[pd.isnull(deals['a_makedown'])].index, inplace = True)
        
        # translate spots to x's - add blank to end of hand to give EOF for clubs
        spots_to_x = {50: 'x', 51: 'x', 52: 'x', 53: 'x', 54: 'x', 55: 'x', 56: 'x', 57: 'x'}
        deals['north'] = [x.translate(spots_to_x) + ' ' for x in deals['north']]
        deals['south'] = [x.translate(spots_to_x) + ' ' for x in deals['south']]
        
        # add columns for analysis
        # True if vulnerable
        deals['vulnerable'] = deals.apply(lambda row: row.bd_number % 16 in [2, 4, 5, 7, 10, 12, 13, 15] , axis = 1)
        # True if 3NT makes
        deals['makes_3NT'] = deals.apply(lambda row: row.a_makedown > 0, axis = 1)
        # score if we bid 3NT and they don't
        deals['score'] = deals.apply(lambda row: row.vul_imps if row.vulnerable else row.nv_imps, axis = 1) 
        
        # North working variables
        deals['n_S'] = deals.apply(lambda row: row.north.index('H') - 3, axis = 1)
        deals['n_H'] = deals.apply(lambda row: row.north.index('D') - row.north.index('H') - 3, axis = 1)
        deals['n_D'] = deals.apply(lambda row: row.north.index('C') - row.north.index('D') - 3, axis = 1)
        deals['n_C'] = deals.apply(lambda row: len(row.north) - row.north.index('C') - 2, axis = 1)
                  
        # South working variables
        deals['s_S'] = deals.apply(lambda row: row.south.index('H') - 3, axis = 1)
        deals['s_H'] = deals.apply(lambda row: row.south.index('D') - row.south.index('H') - 3, axis = 1)
        deals['s_D'] = deals.apply(lambda row: row.south.index('C') - row.south.index('D') - 3, axis = 1)
        deals['s_C'] = deals.apply(lambda row: len(row.south) - row.south.index('C') - 2, axis = 1)
     
        # Features for both North and South   
        for feature in globals.feature_names[:globals.first_special_feature]:  # special features are not honor holding
            deals[feature] = deals.apply(lambda row: row.north.count(f':{feature} ') + row.south.count(f':{feature} '), axis = 1)
        
        if '5422' in globals.feature_names:
            deals['5422'] = deals.apply(lambda row: int(row.n_S != 3 and row.n_H != 3 and 
                    row.n_D != 3 and row.n_C != 3) + int(row.s_S != 3 and row.s_H != 3 and 
                    row.s_D != 3 and row.s_C != 3), axis = 1)
        
        if '4333' in globals.feature_names:
            deals['4333'] = deals.apply(lambda row: int(row.n_S > 2 and row.n_H > 2 and 
                    row.n_D > 2 and row.n_C > 2) + int(row.s_S > 2 and row.s_H > 2 and 
                    row.s_D > 2 and row.s_C > 2), axis = 1)
                                       
         
        # save processed data in a new file
        writeFile(deals, fileName.replace('.csv', f'_{globals.feature_set}.csv').replace('.xslx', f'_{globals.feature_set}.xslx'))
        
    return deals


