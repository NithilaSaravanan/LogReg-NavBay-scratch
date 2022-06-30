#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 00:52:30 2020

@author: nithila
"""


#Class for logistic regression

import numpy as np
import matplotlib.pyplot as plt
import os


class LogistRegr:
    
    # Logistic sigmoid function
    def sigmoid_func(self, a):    
        return 1 / (1 + np.exp(-a))
    
    
    
    # Defining the cost function for Logistic Regression
    def LossFunc(self, Inp, Tar, omg, lam):   
        y = self.sigmoid_func(np.dot(Inp, omg))
        a = np.multiply((1 - Tar), np.log(1 - y))
        b = np.multiply(Tar, np.log(y))
        return np.mean((-1/len(Inp)) * np.sum(a + b))
    
    
    
    # Function to calcualte gradient descent that gives us the optimised weights for the model   
    def Fit(self, Inp, Tar, alpha, iters): 
        a = Inp.shape[0] #nSample size
        theta = np.zeros(Inp.shape[1])
        Inp_transpose = Inp.transpose()
        cost1 = []
        print("Calculating...")
        
        for iter in range(0,iters):
            
            y_h = self.sigmoid_func(np.dot(Inp, theta))
            #print("y_h", y_h.shape[0])
            loss = y_h - Tar
            #print(loss.shape[1])

            J1 = self.LossFunc(Inp, Tar, theta, 0)  # Loss Function evaluated on the validation set
            #J2 = self.LossFunc(cv_x, cv_y, theta, 0)
            #print (iter)  
            #diff_val = J1 - old_val
            #old_val = J1
            gradient = np.dot(Inp_transpose, loss) / a  
            #print("Gradient", gradient.shape[0])
            theta = theta - alpha * gradient 
            #print("Theta", theta.shape[0])
            cost1.append(J1)
            #cost2.append(J2)
            #if (diff_val<=0.000000000001): #(and iter>10000)
             #  break
        plt.plot(cost1)
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function value')
        plt.title('Ionosphere - Cost trend')
        #plt.plot(cost2)
        plt.show()
        return theta
    
    
    
    # Function to predict the target using our model
    def predict(self, w, input):
       
        tar_eval = np.full((input.shape[0], 1), -1)
        #tar_eval = []
        for i in range(input.shape[0]):
            Z = np.dot(np.transpose(w), input[i])
            tar_eval[i] = 0 if (1 / (1 + np.exp(-1*Z))) <= 0.5 else 1
        return tar_eval
    
    
    
    # For KFold CV
    def cv(self, Inp, Tar, alpha, loops, k):
        
        samp_x = np.array (Inp)
        samp_y = np.array (Tar)
        size = int(np.ceil(samp_x.shape[0]/k))
        
        nsamples = samp_x.shape[0]
        newfeat = np.arange(nsamples)
        samp_x = np.concatenate((samp_x, newfeat[:, None]), axis=1)
        
        tot_acc = []
        for iter in range(k):
            inp_train_x = list(samp_x)
            inp_train_y = list(samp_y)
            
            inp_test_x = samp_x[iter * size : (iter + 1) * size]
            inp_test_y = samp_y[iter * size : (iter + 1) * size]
            
            inp_train_x[iter * size : (iter +1) * size] = []
            inp_train_y[iter * size : (iter +1) * size] = []
            
            inp_train_x = np.array(inp_train_x)
            inp_train_y = np.array(inp_train_y)
            
            best_theta = self.Fit(inp_train_x, inp_train_y, alpha, loops)
            predict = self.predict(best_theta, inp_test_x)
            
            correct = [1 if a == b else 0 for (a, b) in zip(predict, inp_test_y.tolist())]
            accuracy = (sum(map(int, correct)) / len(correct)*100)  
            print("Fold #", iter + 1)
            print('The Classification Error is {0:.2f}'.format(correct.count(0)/len(correct)))
            print('The accuracy is {0:.2f}%'.format(accuracy),'\n\n')
            tot_acc.append(accuracy)
        avg_acc = np.mean(tot_acc)
        print('Average accuracy is {0:.2f}%'.format(avg_acc))
        print('Std accuracy is {0:.2f}%'.format(np.std(tot_acc)))

    def FitExp(self, Inp, Tar, alpha, iters): 
        a = Inp.shape[0] #nSample size
        theta = np.zeros(Inp.shape[1])
        Inp_transpose = Inp.transpose()
        cost1 = []
        print("Calculating...")
        diff_val=1
        old_val=999999999
        iter_ = 0
        
        while (diff_val>0.00001):
        #for iter_ in range(0,iters):
            
            y_h = self.sigmoid_func(np.dot(Inp, theta))
            #print("y_h", y_h.shape[0])
            loss = y_h - Tar
            #print(loss.shape[1])

            J1 = self.LossFunc(Inp, Tar, theta, 0)  # Loss Function evaluated on the validation set
            #J2 = self.LossFunc(cv_x, cv_y, theta, 0)
            print (iter_,J1)  
            diff_val = old_val - J1
            old_val = J1
            gradient = np.dot(Inp_transpose, loss) / a  
            #print("Gradient", gradient.shape[0])
            theta = theta - alpha * gradient 
            #print("Theta", theta.shape[0])
            cost1.append(J1)
            #cost2.append(J2)
            #if (diff_val<=0.000000000001): #(and iter>10000)
            #   break
            iter_ = iter_ + 1
        plt.plot(cost1)
        #plt.plot(cost2)
        plt.show()
        return theta
    
    def cvexp(self, Inp, Tar, alpha, loops, k):
        samp_x = np.array (Inp)
        samp_y = np.array (Tar)
        size = int(np.ceil(samp_x.shape[0]/k))
        
        tot_acc = []
        for iter in range(k):
            inp_train_x = list(samp_x)
            inp_train_y = list(samp_y)
            
            inp_test_x = samp_x[iter * size : (iter + 1) * size]
            inp_test_y = samp_y[iter * size : (iter + 1) * size]
            
            inp_train_x[iter * size : (iter +1) * size] = []
            inp_train_y[iter * size : (iter +1) * size] = []
            
            inp_train_x = np.array(inp_train_x)
            inp_train_y = np.array(inp_train_y)
            
            best_theta = self.FitExp(inp_train_x, inp_train_y, alpha, loops)
            predict = self.predict(best_theta, inp_test_x)
            
            correct = [1 if a == b else 0 for (a, b) in zip(predict, inp_test_y.tolist())]
            accuracy = (sum(map(int, correct)) / len(correct)*100)  
            print("Fold #", iter + 1)
            print('The Classification Error is {0:.2f}'.format(correct.count(0)/len(correct)))
            print('The accuracy is {0:.2f}%'.format(accuracy),'\n\n')
            tot_acc.append(accuracy)
        avg_acc = np.mean(tot_acc)
        print('Average accuracy is {0:.2f}%'.format(avg_acc))

    