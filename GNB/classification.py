# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:18:53 2020

@author: mariliapc
"""

# Dataset
from dataset import Dataset
import numpy as np
from utils import sparse_var, LossFunc, sigmoid_func
import matplotlib.pyplot as plt
import time


''' Super class for all the implemented classifier '''
class Classifier:
    
    def evaluate_acc(self, y_true, y_pred):
        (TP, TN, FP, FN) = self.evaluate_confusion_matrix(y_true, y_pred)
        return (TP+TN)/(TP+TN+FP+FN)
    
    
    def evaluate_confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_true==y_pred) & (y_true==True))
        TN = np.sum((y_true==y_pred) & (y_true==False))
        FP = np.sum((y_true!=y_pred) & (y_pred==True))
        FN = np.sum((y_true!=y_pred) & (y_pred==False))
        
        return (TP, TN, FP, FN)
    
    
    def evaluate_all_metrics(self, y_true, y_pred):
        (TP, TN, FP, FN) = self.evaluate_confusion_matrix(y_true, y_pred)
        
        # (acc, pre, rcl)
        return (self.evaluate_acc(y_true, y_pred), TP/(TP+FP), TP/(TP+FN))
    
class GaussianNaiveBayes(Classifier):
    def __init__(self):
        pass
    
    def fit(self, ds, verbose=True):
        self.ds = ds
        nbclass = ds.getNbClass()
        nbsamples = ds.getNbSamples()
        nbfeatures = ds.getNbFeatures()
        y = ds.y
        X = ds.X
        
        if(verbose):
            print('Fitting dataset with')
            print('\tnclass    : '+str(nbclass))
            print('\tnsamples  : '+str(nbsamples))
            print('\tnfeatures : '+str(nbfeatures))
        
        ''' Note that only the variance of each feature is computed and not the
        covariance between features because the Naive Bayes assumption is that 
        features are considered independant within any class y '''
        
        # Compute variance and mean
        # nbclass Vectors with length of number of features (one vector per class)
        Mean = np.zeros((nbclass, nbfeatures))
        Var = np.zeros((nbclass, nbfeatures))
        log_den = np.zeros((nbclass, nbfeatures))
        Py = np.zeros((nbclass,))
        
        for k in range(nbclass):
            X = ds.getSamples(classe=k) # (nbsamples, nbfeatures)
            
            Mean[k] = X.mean(axis=0) # compute mean for each features (along nbsamples)
            Var[k] = sparse_var(X, axis=0) # compute variance for each features (along nbsamples)
            Py[k] = np.log( (y==k).sum() / nbsamples)
            #log_den[k] = np.zeros((nbfeatures,))
            np.log(np.sqrt(2*np.pi*Var[k])) # compute gaussian distribution coefficient
            
            if(verbose):
                print('P(y='+str(k)+')   = '+str(Py[k]))
                print('log( (2piVar)^2 ) = '+str(log_den[k]))
                print('avg mean 20-class = '+str(np.mean(Mean, axis=1)))
                print('avg var  20-class = '+str(np.mean(Var,  axis=1)))
                print('class '+str(k))
        
        self.log_den = log_den
        #input('log_den: ' + str(log_den))
        self.Py = Py
        #input('Py: ' + str(Py))
        self.Mean = Mean
        #input('Mean: ' + str(Mean))
        self.Var = Var
        #input('Var: ' + str(Var))
   
    def predict(self, ds_test, verbose=False):
        X_test = ds_test.X
        nbclass = self.ds.getNbClass()
        nbsamples = X_test.shape[0] # Number of samples to test
        
        Mean = self.Mean       # (nbclass, nbfeatures)
        Var = self.Var         # (nbclass, nbfeatures)
        log_den = self.log_den # (nbclass, nbfeatures)
        Py = self.Py           # (nbclass)
        # X_test                 (nbsamples, nbfeatures)
        
        y_pred = []
        
        class_prob = np.zeros((nbclass,))
        for i in range(nbsamples):
            
            # Prediction for sample i
            for k in range(nbclass):
                # Log-robability of x knowing y (for each features)
                c = np.square(X_test[i]-Mean[k])
                #input('c: ' + str(c))
                d = Var[k]
                #input('d: ' + str(d))
                div = np.divide(c, d, out=np.zeros_like(c), where=d!=0)
                #input('div: ' + str(div))
                Px_y = -0.5*div - log_den[k] # Log of gaussian distribution
                #input('Px_y: ' + str(Px_y))
                Px_y = np.sum(Px_y) # Sum log probability for all features -> Scalar
                #input('sum Px_y: ' + str(Px_y))
                class_prob[k] = Py[k]+Px_y # Compute final class probability
                #input('class_prob['+str(k)+']: ' + str(class_prob[k]))
            
            #print('class_prob' + str(class_prob))
            y_pred.append(np.argmax(class_prob))
            #input('y_pred: ' + str(y_pred))

            if(verbose):
                print('Prediction '+str(i)+': class '+str(np.argmax(class_prob)))
        
        return np.array(y_pred)

def cross_validation(model, data, cv=5, verbose=False):
    
    start_time = time.time()
    
    X0 = data.getSamples(classe=0)
    X1 = data.getSamples(classe=1)
    n0 = X0.shape[0]
    n1 = X1.shape[0]
    nfeat = data.getNbFeatures()
    
    
    nSamplePerSplit0 = int(n0/cv)
    nSamplePerSplit1 = int(n1/cv)
    nSamplePerSplitTot = nSamplePerSplit0+nSamplePerSplit1
    print('Splitting dataset in '+str(cv)+' stratified fold of '+str(nSamplePerSplitTot)+' samples')

    # Split dataset
    Xsplit = {}
    ysplit = {}
    for icv in range(cv):
        begin0 = int(icv*nSamplePerSplit0)
        end0 = int((icv+1)*nSamplePerSplit0)
        begin1 = int(icv*nSamplePerSplit1)
        end1 = int((icv+1)*nSamplePerSplit1)
        Xsplit[icv] = np.concatenate((X0[begin0:end0, :], X1[begin1:end1, :]), axis=0)
        ysplit[icv] = np.concatenate((np.zeros((nSamplePerSplit0,)), np.ones((nSamplePerSplit1,))), axis=0)
        
        '''print(nSamplePerSplit0)
        print(nSamplePerSplit1)
        print(begin0)
        print(end0)
        print(begin1)
        print(end1)
        input(str(np.unique(ysplit[icv])))'''

    # CV Evaluation metrics
    list_accuracy = []
    list_precision = []
    list_recall = []
    for icv in range(cv):
        # Extract train and test set
        Xtrain = np.empty((0, nfeat))
        ytrain = np.array([])
        Xtest = np.empty((0, nfeat))
        ytest = np.array([])
        for jcv in range(cv):
            if(jcv != icv):
                Xtrain = np.concatenate((Xtrain, Xsplit[jcv]), axis=0)
                ytrain = np.concatenate((ytrain, ysplit[jcv]), axis=0)
            else:
                Xtest = np.concatenate((Xtest, Xsplit[jcv]), axis=0)
                ytest = np.concatenate((ytest, ysplit[jcv]), axis=0)
        
        trainSet = Dataset(Xtrain, ytrain)
        #print('Training set information')
        #trainSet.info()
        # Train model with training set
        model.fit(trainSet, verbose=verbose)
        #input('fit stop')
        # Test model with testing set
        testSet = Dataset(Xtest, ytest)
        y_pred = model.predict(testSet, verbose=verbose)
        
        
        # Evaluate metrics
        (accuracy, precision, recall) = model.evaluate_all_metrics(ytest, y_pred)
        list_accuracy.append(accuracy)
        list_precision.append(precision)
        list_recall.append(recall)
        
        if(verbose):
            print('----- fold '+str(icv)+' -----')
            print('-> accuracy:  '+str(accuracy))
            print('-> precision: '+str(precision))
            print('-> recall:    '+str(recall))
    
    result = {}
    result['avg_acc'], result['std_acc'] = (np.mean(list_accuracy) , np.std(list_accuracy))
    result['avg_pre'], result['std_pre'] = (np.mean(list_precision), np.std(list_precision))
    result['avg_rcl'], result['std_rcl'] = (np.mean(list_recall)   , np.std(list_recall))
    result['time'] = (time.time() - start_time)
    
    print('\n----- Global results -----')
    print('accuracy : '+str(result['avg_acc'])+' +/- '+str(result['std_acc']))
    print('precision: '+str(result['avg_pre'])+' +/- '+str(result['std_pre']))
    print('recall   : '+str(result['avg_rcl'])+' +/- '+str(result['std_rcl']))    
    print("  runtime =  --- %s seconds ---," % result['time'])
    
    return result
    

def corr_fs(data, top=5):
    
    nbfeat = data.getNbFeatures()
    if(top<0):
        top = nbfeat+top
    
    X = np.concatenate((data.X, data.y[:, None]), axis=1)
    corrMatrix = np.abs(np.corrcoef(X, rowvar=False))
    label_corr = corrMatrix[:, -1].squeeze()
    label_corr = label_corr[:-1] # Remove correlation of label with himself
    worst_feat_idx = np.argsort(label_corr)[:-top] # Take only the less correlated features
    data.removeFeature(worst_feat_idx)
    
    return data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        