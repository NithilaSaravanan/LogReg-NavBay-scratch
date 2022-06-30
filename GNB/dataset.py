#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:25:44 2020

@author: mariliapc
"""

import numpy as np
# Since Python 3.6, dictionnaries keep the keys in the same order as they are declared
# OrderedDict is used to maintain compatibility with previous python version
from collections import OrderedDict


class Dataset:
    def __init__(self, X, y, Xnames=[], Xvalues=OrderedDict(), yvalues=[], name=''):
        self.X = X  # Features, shape: [nsamples, nfeatures]
        self.y = y  # Label, shape: (nsamples,)
        
        self.Xnames = Xnames # Array of features names (in order of apparition)
        self.Xvalues = Xvalues # [feature ID] -> array of string values for features
        self.yvalues = yvalues # Array of string values for label vector
        
        self.name = name # Dataset name
        
    def compute_correlation(self, i, j):
        Xi = self.X[:, i]
        Xj = self.X[:, j]
        
        Xi = Xi / np.linalg.norm(Xi)
        Xj = Xj / np.linalg.norm(Xj)
        
        #input(np.multiply(Xi, Xj).shape)
        #input(np.sum(np.multiply(Xi, Xj)))
        
        return np.sum(np.multiply(Xi, Xj))
    
    # ---- Preprocessing ----
    ''' If label_only is set to False, this method convert all the categorical
    text features into binary features '''
    def binarize(self, inc_thr=6, label_only=True):
        class1 = (self.y >= inc_thr)
        self.y[class1] = 1
        self.y[~class1] = 0
        
        if(label_only):
            return
        
        nsamples = self.getNbSamples()
        nfeat = self.getNbFeatures()
        
        X = np.empty((nsamples, 0))
        Xvalues = OrderedDict()
        Xnames = []
        
        for i in range(nfeat):
            (set_type, var_type) = self.getFeatType(i)
            if(set_type == 'categorical'):
                feat = self.X[:, i]
                nvalues = np.unique(feat)
                Xbin = np.zeros((nsamples, nvalues), dtype=int)
                
                for j in range(nvalues):
                    Xbin[:, j] = np.array(feat == j)
                    Xnames.append(self.Xvalues[i][j])
                    Xvalues[i] = []
                X = np.concatenate((X, Xbin), axis=1)
            else:
                X = np.concatenate((X, self.X[:, i]), axis=1)
                Xnames.append(self.Xnames[i])
                Xvalues[i] = self.Xvalues[i]
                
        self.Xnames = Xnames
        self.Xvalues = Xvalues
        self.X = X
        
    ''' Getters '''
    def getNbFeatures(self):
        return self.X.shape[1]
        
    def getNbSamples(self):
        return self.X.shape[0]
    
    def getSamples(self, classe=0):
        if(not classe in np.unique(self.y)):
            print('Warning: class '+str(classe)+' not present in labels')
        selected_sample = (self.y==classe).ravel()
        return self.X[selected_sample, :]
    
    def getFeatName(self, index):
        return self.Xnames[index]
    
    # Return the number of classes
    def getNbClass(self):
        return int(np.unique(self.y).size)
    
    def getFeatType(self, index):
        _values = self.Xvalues[index]# Set of values of this features (empty = infinite)
        
        if(np.unique(self.X[:, index]).size > 2):
            if(_values.size > 0):
                # The list of values for this features is not empty -> categorical
                set_type = 'categorical'
                var_type = 'text'
            else:
                # Multiple element and not finite set of values -> continuous
                set_type = 'continuous'
                var_type = 'number'
        else:
            # The feature is necessarely binary
            set_type = 'binary'
            
            # The list of values is available for text features only
            var_type = 'text' if (_values.size > 0) else 'number'
            
        return (set_type, var_type)
    
    def normalize(self):
        _min = np.amin(self.X, axis=0)
        _max = np.amax(self.X, axis=0)
        self.X = self.X-_min
        self.X = np.divide(self.X, (_max-_min))
    
    ''' Miscelleanous '''
    def info(self, verbose=1):
        nfeat = self.getNbFeatures()
        
        print('--- Dataset '+self.name+' info ---')
        print('nb samples : '+str(self.getNbSamples()))
        print('nb features: '+str(nfeat))
        
        classID = np.unique(self.y)
        
        if(classID.size==2): # If label are binary
            nsamples_c0 = (self.y==0).sum()
            nsamples_c1 = (self.y==1).sum()
            nsamples = self.y.size
            ratio0 = nsamples_c0/nsamples
            ratio1 = nsamples_c1/nsamples
            print('Chance accuracy: '+str(max(ratio0, ratio1)))
        else:
            # Plot distribution of samples from each class
            for i in classID:
                n = (self.y==i).sum()
                print('class '+str(i)+' n= '+str(n))
        
        assert(self.X.shape[1]==len(self.Xvalues))
        
        if(verbose>0):
            for i in range(nfeat):
                _name = self.getFeatName(i)
                _type = self.getFeatType(i)
                print('Feature '+_name+' type: '+str(_type))
            print('labels: '+str(np.unique(self.y)))
        
    def addFeatures(self, Type='Quadratic'):
        nsample = self.X.shape[0]
        nfeat = self.X.shape[1]
        
        if(Type=='Quadratic'):
            Xextra = np.square(self.X)
            Xextra_names = ['quad_'+name for name in self.Xnames]
            
        elif(Type=='Log'):
            Xextra = np.empty((nsample, 0))
            Xextra_names = []
            for i in range(nfeat):
                (set_type, var_type) = self.getFeatType(i)
                if(set_type == 'continuous'): #if feat type is continous
                    logFeat = np.log(self.X[:, i]) #then apply log on feature and concatenate on Xextra and Xextra_names
                    
                    logFeat[np.isnan(logFeat)] = 0 
                    logFeat[np.isneginf(logFeat)] = 0
                    
                    Xextra = np.concatenate((logFeat[:, None], Xextra), axis=1)
                    Xextra_names.append('log_'+self.Xnames[i])
                    
        elif(Type=='Interactive'):
            Xextra = np.empty((nsample, 0))
            Xextra_names = []
            for i in range(nfeat):                
                    for j in range(nfeat):
                        if(i>j):
                            newsig = np.multiply(self.X[:, i], self.X[:, j])
                            if(any(np.isnan(newsig))):
                                print('Nan:' + str(newsig[i]))
                            if(any(np.isneginf(newsig))):
                                print('inf:' + str(newsig[i]))
                            if(any(newsig != 0)):
                                newsig = newsig[:, None]
                                Xextra = np.concatenate((Xextra, newsig), axis=1)
                                Xextra_names.append(self.Xnames[i]+'x'+self.Xnames[j])
            
        self.X = np.concatenate((self.X, Xextra), axis=1)
        self.Xnames = np.concatenate((self.Xnames, Xextra_names), axis=0)
        for i in range(len(Xextra_names)):
            self.Xvalues[nfeat+i] = np.array([])
             
    def addFeaturesNames(self,  Xnames=[]):
        if(len(Xnames)==self.getNbFeatures()):
            self.Xnames = Xnames
        else:
            print('Xnames given has '+str(len(Xnames))+' but there is only '+str(self.getNbFeatures())+' features')
    
    def removeFeature(self, arr):
        
        nfeat = self.getNbFeatures()
        
        # Make sure index is a list (even for one element)
        if(not isinstance(arr, list)):
            if(isinstance(arr, np.ndarray)):
                arr = arr.tolist()
            else:
                arr = [arr]
        
        # Convert the names of features to their corresponding index (if necessary)
        if(isinstance(arr[0], str)):
            index = []
            for _idx in arr:
                index.append(self.Xnames.index(_idx))
        else:
            index = arr
        
        self.X = np.delete(self.X, index, axis=1)
        self.Xnames = np.delete(self.Xnames, index)
        
        Xvalues = OrderedDict()
        cnt=0
        for i in range(nfeat):
            # Keep feature values if its index is not in the list
            if(not i in index):
                Xvalues[cnt] = self.Xvalues[i]
                cnt+=1
        self.Xvalues = Xvalues
    
    
    
    
    