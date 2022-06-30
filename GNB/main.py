# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:23:11 2020

@author: mariliapc
"""

import os

from classification import GaussianNaiveBayes
from visualization import plot_corr_matrix, plot_histogram
from IO import loadDataset, write_header, write_results
from classification import cross_validation, corr_fs


''' Load datasets '''
Dataset = {}
Dataset['Ion']    = ionDS    = loadDataset('DS1/ionosphere.data')
Dataset['Adult']  = adultDS  = loadDataset('DS2/adult.data')
Dataset['Wine']   = wineDS   = loadDataset('DS3/winequality-red.csv', header=True, delimiter=';')
Dataset['Cancer'] = cancerDS = loadDataset('DS4/breast-cancer-wisconsin.data')

''' Preprocessing '''
# Binarize labels and features
wineDS.binarize()
cancerDS.binarize(inc_thr=3)


# Add features names
cancerDS.addFeaturesNames(['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
              'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
              'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'])
adultDS.addFeaturesNames(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

# Remove redundant features
adultDS.removeFeature('education-num')
ionDS.removeFeature(1)

ionDS.normalize()
adultDS.normalize()
wineDS.normalize()
cancerDS.normalize()



# Data augmentation
'''for name in Dataset:
    Dataset[name].addFeatures(Type='Interactive')
    Dataset[name].addFeatures(Type='Log')
    Dataset[name].addFeatures(Type='Quadratic')
'''

''' --- Explore datasets --- '''
# Correlation matrix
#plot_corr_matrix(cancerDS)
#plot_corr_matrix(wineDS)
#plot_corr_matrix(IonDS)
#plot_corr_matrix(adultDS)

#plot_histogram(wineDS)
#input('stop')
#plot_histogram(ionDS, feat=4, labels=['Bad', 'Good'])
#input('stop')
plot_histogram(adultDS, feat=[0], labels=['<=50K', '>50K'])
input('stop')
#plot_histogram(cancerDS, feat='Uniformity of Cell Size', labels=['Benign', 'Malignant'])
#input('stop')

# Basic Informations
#wineDS.info()
#cancerDS.info()
#IonDS.info()
#adultDS.info()

''' Classification '''
model = GaussianNaiveBayes()
pipeline = 'original'



# Prepare results file
fname = 'result_LR.csv'
if(not os.path.exists(fname)):
    write_header(fname)

# Cross-validation with specific condition
print('--- pipeline %s ---' % pipeline)
for name in Dataset:
    if(pipeline=='original'):
        results = cross_validation(model, Dataset[name], cv=5)
        write_results(fname, name+'_'+pipeline, results)
        
    elif(pipeline=='original_fs3'):
        Dataset[name] = corr_fs(Dataset[name], top=3)
        results = cross_validation(model, Dataset[name], cv=5)
        write_results(fname, name+'_'+pipeline, results)
        
    elif(pipeline=='augmented'):
        Dataset[name].addFeatures(Type='Interactive')
        Dataset[name].addFeatures(Type='Log')
        Dataset[name].addFeatures(Type='Quadratic')
        
        results = cross_validation(model, Dataset[name], cv=5)
        write_results(fname, name+'_'+pipeline, results)

    elif(pipeline=='augmented_fs3'):
        Dataset[name].addFeatures(Type='Interactive')
        Dataset[name].addFeatures(Type='Log')
        Dataset[name].addFeatures(Type='Quadratic')
        
        Dataset[name] = corr_fs(Dataset[name], top=3)
        
        results = cross_validation(model, Dataset[name], cv=5)
        write_results(fname, name+'_'+pipeline, results)




