#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 23:07:05 2020

@author: mariliapc
"""

import csv
import numpy as np
from collections import OrderedDict

from dataset import Dataset
from utils import isfloat




def loadDataset(fname='DS1/ionosphere.data', header=False, delimiter=','):
    
    print('**** Now reading '+fname+' ****')
    
    # Read the row converting string element into ID (a number) and saving the
    # collected data in X and y. Each string and their associated ID are saved
    # in Xvalues dictionnary
    def process_row():
        nonlocal row, counter
        nonlocal Xvalues, X, y
        
        if(check_line()==False):
            return # Ignore this line
        
        Xrow = []
        for (i, value) in enumerate(row):
            
            if(isfloat(value)):
                Xrow.append(float(value))
            else:
                # If this value is read for the first time, add it in the list
                if(not value in Xvalues[i]):
                    Xvalues[i] = np.append(Xvalues[i], value)
                
                ID = np.flatnonzero(Xvalues[i]==value)[0]
                Xrow.append(ID)
        
        Xrow = np.array(Xrow)
        Xrow = Xrow[None, :]
        X = np.concatenate((X, Xrow[:, :-1]))
        y = np.append(y, Xrow[0, -1]) # Assume the last column correspond to the label

        
    
    # Check if the line contains incomplete data or if the line is empty
    def check_line():
        nonlocal row, counter
        
        row_raw = ''.join(row)
        if('?' in row_raw or len(row_raw)==0):
            #print('Line '+str(counter)+' contains incomplete data')
            return False
        return True
    
    
    Xnames = []
    Xvalues = OrderedDict()
    yvalues = []
    
    with open(fname, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        
        # --- First line ---
        row = next(csvreader)
        nfeat = len(row)-1 # Ignore label
        X = np.empty((0, nfeat))
        y = np.empty((0,))
        
        if(header):
            # Read header
            for (i, _name) in enumerate(row):
                Xnames.append(_name)
                Xvalues[i] = np.array([])
        else:
            # Generate feat name
            for i in range(len(row)):
                Xnames.append('Feat'+str(i))
                Xvalues[i] = np.array([])
            process_row()
        
        
        # --- Process the remaining of the file ---
        for (counter, row) in enumerate(csvreader):
            process_row()
            
        # The last feature name correspond to the label
        yvalues = Xvalues.pop(list(Xvalues.keys())[-1])
        Xnames = Xnames[:-1] # The label name doesn't matter so much
        
        
    
    ds = Dataset(X, y, Xnames=Xnames,
                         Xvalues=Xvalues,
                         yvalues=yvalues, name=fname)
    
    print('Nb samples read: '+str(ds.getNbSamples()))
    
    return ds


results_names = ['avg_acc','std_acc','avg_pre','std_pre','avg_rcl','std_rcl','time']
def write_header(filename):
    with open(filename, 'w') as fout:
        fout.write('Condition;')
        for name in results_names:
            fout.write(name+';')
        fout.write('\n')

def write_results(filename, condition, results):
    with open(filename, 'a') as fout:
        fout.write(condition+';')
        for name in results_names:
            fout.write('%.3f;' % results[name])
        fout.write('\n')
    
    
    
    