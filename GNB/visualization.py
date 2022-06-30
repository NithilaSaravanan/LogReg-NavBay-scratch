# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:22:37 2020

@author: mariliapc
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Plot distribution of samples from each class according to a specific feature
def plot_histogram(data, feat='alcohol', labels=['Bad quality', 'Good quality'], invColor=False):
    X0 = data.getSamples(classe=0)
    X1 = data.getSamples(classe=1)
    
    # Select specific feature
    feat_idx = data.Xnames.index(feat)
    X0 = X0[:, feat_idx]
    X1 = X1[:, feat_idx]
    
    sns.set_style("white")

    # Plot
    if(invColor):
        color0 = "blue"
        color1 = "orange"
    else:
        color0 = "orange"
        color1 = "blue"
    
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

    plt.title('Samples distribution according to '+feat)
    sns.distplot(X0, color=color0, label=labels[0], **kwargs)
    sns.distplot(X1, color=color1, label=labels[1], **kwargs)
    
    #plt.xlim(0,1)
    plt.legend()
    plt.show()
    
    
    
def plot_corr_matrix(data, className='target', title=None, useAbs=True):
    nfeat = data.getNbFeatures()
    
    corrMatrix = np.zeros((nfeat, nfeat))
    
    for i in range(nfeat):
        for j in range(nfeat):
            corrMatrix[j][i] = data.compute_correlation(i, j)
    
    Xnames = np.concatenate((data.Xnames, [className])) 
    X = np.concatenate((data.X, data.y[:, None]), axis=1)
    corrMatrix = np.corrcoef(X, rowvar=False)
    
    if(useAbs):
        corrMatrix = np.abs(corrMatrix)
    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(corrMatrix, interpolation='nearest', cmap=plt.cm.jet)
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.colorbar()
    #plt.xticks(np.arange(len(Xnames)), Xnames, rotation=45)
    plt.xticks(np.arange(len(Xnames)), Xnames, rotation=90)
    plt.yticks(np.arange(len(Xnames)), Xnames)
    if(title is None):
        title = 'Features cross-correlation'
        plt.title(title+' for '+data.name)
    else:
        plt.title(title)
    plt.savefig('corrMat.png', bbox_inches='tight')
    plt.show()
    
    
    


    
    
    