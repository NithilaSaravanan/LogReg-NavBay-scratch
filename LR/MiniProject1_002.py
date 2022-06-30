#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:43:10 2020

@author: nithila
"""
# Changelog: ---
# To Do: Implement K Fold


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # for shuffling datasets for train set size experiment

from LogRegrClass import LogistRegr as lgrg

obj = lgrg()


#Loading Datasets into dataframes

# Dataset 1
df1=pd.read_csv('DS1/ionosphere.data', delimiter=',' , names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','target'])

# Dataset 2
df21=pd.read_csv ('DS2/adult.data',  delimiter = ',', header=None, names=['age',
'workclass',
'fnlwgt',
'education',
'education_num',
'marital_status',
'occupation',
'relationship',
'race',
'sex',
'capital_gain',
'capital_loss',
'hours_per_week',
'native_country',
'target'])
df22=pd.read_csv ('DS2/adult.test', delimiter = ',', header=None, names=['age',
'workclass',
'fnlwgt',
'education',
'education_num',
'marital_status',
'occupation',
'relationship',
'race',
'sex',
'capital_gain',
'capital_loss',
'hours_per_week',
'native_country',
'target'])
df22=df22.iloc[1:,:] #removing the first row - gibberish
df2=pd.concat([df21,df22]).drop_duplicates().reset_index(drop=True)#Combining the two datasets and removing duplicates    
df2=df2.drop('education', axis = 1)

# Dataset 3

df3 = pd.read_csv('DS4/breast-cancer-wisconsin.data', delimiter = ',', names = ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','frac_dim','target'])
print(df3.head(3))

# Dataset 4
df4 = pd.read_csv('DS3/winequality-red.csv', delimiter = ';')
print(df4.head(3))

#***********


#df1['count']=np.arange(len(df1))
#df2['count']=np.arange(len(df2))

#print(df.head())
'''
colsum_1=(df1.sum(axis=0, skipna=True))
print (colsum_1)

rowsum_1=df1.sum(axis=1, skipna=True)
print (rowsum_1)


colsum_2=(df2.sum(axis=0, skipna=True))
print (colsum_2)

rowsum_2=df2.sum(axis=1, skipna=True)
print (rowsum_2)
'''


# df.drop('2', axis=1) #Drop later, after creating basic stats
#df1.groupby(['target']) ['count'].count() #Class splits
(df1.isnull().any().any()) #Check nulls
(df1.isnull().values.sum()) #Check nulls

df2 = df2.replace(" ?", 'XXX')

(df2.isnull().any().any())
(df1.isnull().values.sum()) #Check nulls



tobedel = df2[ (df2['workclass'] == 'XXX') | (df2['occupation'] == 'XXX') | (df2['native_country'] == 'XXX')].index
df2.drop(tobedel , inplace=True)

df2['target']=df2['target'].map({' >50K' : 1, ' >50K.' : 1, ' <=50K' : 0, ' <=50K.' : 0})
df1['target']=df1['target'].map({'g' : 1, 'b' : 0})



df2['workclass'] = df2['workclass'].str.replace(" ",'')
#df2['education'] = df2['education'].str.replace(" ",'')
df2['marital_status'] = df2['marital_status'].str.replace(" ",'')
df2['relationship'] = df2['relationship'].str.replace(" ",'')
df2['race'] = df2['race'].str.replace(" ",'')
df2['sex'] = df2['sex'].str.replace(" ",'')
df2['native_country'] = df2['native_country'].str.replace(" ",'')

list(df2)
tobenormalized2 = ['fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
df2[tobenormalized2] = df2[tobenormalized2].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
q = df2.groupby(['target','native_country']) ['age'].count()
df2=pd.get_dummies(df2) #Final DF2 dataset
df2.shape[1]



df3 = df3.replace('?', float('Nan'))
df3['concavity'].fillna(df3['concavity'].mode()[0], inplace=True)
df3['target'] = df3['target'].map({2 : 0, 4 : 1})
tobenormalized3 = ['radius']
df3[tobenormalized3] = df3[tobenormalized3].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df3 = df3.astype(dtype=np.float64)

df4 = df4.rename(columns={'quality':'target'})
tobenormalized4 = ['fixed acidity','volatile acidity','residual sugar','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
df4[tobenormalized4] = df4[tobenormalized4].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df4['target'] = df4['target'].map({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 9:1, 10:1})


#a = df3['concavity'].mode() [0][0].tolist()
corr = df4.corr()
print(corr)

#print(df4.target.unique())
    
'''
print (df1.target.unique())
    
print (df2.workclass.unique())
print (df2.education.unique())
print (df2.education_num.unique())
print (df2.marital_status.unique())
print (df2.occupation.unique())
print (df2.relationship.unique())
print (df2.race.unique())
print (df2.sex.unique())
print (df2.capital_gain.unique())
print (df2.capital_loss.unique())
print (df2.hours_per_week.unique())
print (df2.native_country.unique())
print (df2.target.unique())






pd.set_option('display.width', 50)
pd.set_option('precision',2)

dscrp_df1=df1.describe()
print(dscrp_df1)

dscrp_d2=df2.describe()
print(dscrp_d2)
'''

# Plotting class splits with the help of a pie chart
print(df4.groupby(['target'])['target'].count())
labels1 = 'Good','Bad'
size1 = [225, 126]
labels2 = '>50K','<=50K'
size2 = [11206, 33988]
labels3 = 'Benign','Malignant'
size3 = [458, 241]
labels4 = 'Good','Bad'
size4 = [855, 744]

colors = ('#009999','#ff9933')
fig1, ax1 = plt.subplots()
#ax1.pie(size1, labels=labels1, autopct='%.0f%%', shadow=True,
#	startangle=90, colors=colors)
#ax1.pie(size2, labels=labels2, autopct='%.0f%%', shadow=True,
#	startangle=90, colors=colors)
ax1.pie(size3, labels=labels3, autopct='%.0f%%', shadow=True,
	startangle=90, colors=colors)
#ax1.pie(size4, labels=labels4, autopct='%.0f%%', shadow=True,
#	startangle=90, colors=colors)
plt.axis('equal')
plt.tight_layout()
plt.title('Wine quality Dataset class splits')
plt.show()




#Splitting the input and output for the two datasets
df1_x = (df1.iloc[:,0:34])
df1_y = (df1['target'])
    
df2_x=df2.drop('target', axis=1)    
df2_y = df2['target']

df3_x=df3.drop('target', axis=1)    
df3_y = df3['target']


df4_x=df4.drop('target', axis=1)    
df4_y = df4['target']



#Creating test train splits for the two datasets 75 - 10 - 15
df1_train_x_k, df1_test_x, df1_train_y_k, df1_test_y = tts(df1_x, df1_y, test_size=0.15, random_state = 10)
df2_train_x_k, df2_test_x, df2_train_y_k, df2_test_y = tts(df2_x, df2_y, test_size=0.15, random_state = 10)
df3_train_x_k, df3_test_x, df3_train_y_k, df3_test_y = tts(df3_x, df3_y, test_size=0.15, random_state = 10)
df4_train_x_k, df4_test_x, df4_train_y_k, df4_test_y = tts(df4_x, df4_y, test_size=0.15, random_state = 10)
#For experiment------
df4_train_x_exp, df4_test_x_exp, df4_train_y_exp, df4_test_y_exp = tts(df4_x, df4_y, test_size=0.10, random_state = 7)
df4_train_x_exp1 = np.array(df4_train_x_exp.iloc[0:int((0.9)*len(df4_train_x_exp))])
df4_train_y_exp1 = df4_train_y_exp.iloc[0:int((0.9)*len(df4_train_x_exp))]
df4_train_x_exp, df4_test_x_exp, df4_train_y_exp, df4_test_y_exp = tts(df4_x, df4_y, test_size=0.10, random_state = 9)
df4_train_x_exp2 = df4_train_x_exp.iloc[0:int((0.75)*len(df4_train_x_exp))]
df4_train_y_exp2 = df4_train_y_exp.iloc[0:int((0.75)*len(df4_train_x_exp))]
df4_train_x_exp, df4_test_x_exp, df4_train_y_exp, df4_test_y_exp = tts(df4_x, df4_y, test_size=0.10, random_state = 11)
df4_train_x_exp3 = df4_train_x_exp.iloc[0:int((0.60)*len(df4_train_x_exp))]
df4_train_y_exp3 = df4_train_y_exp.iloc[0:int((0.60)*len(df4_train_x_exp))]
df4_train_x_exp, df4_test_x_exp, df4_train_y_exp, df4_test_y_exp = tts(df4_x, df4_y, test_size=0.10, random_state = 14)
df4_train_x_exp4 = df4_train_x_exp.iloc[0:int((0.50)*len(df4_train_x_exp))]
df4_train_y_exp4 = df4_train_y_exp.iloc[0:int((0.50)*len(df4_train_x_exp))]
df4_train_x_exp, df4_test_x_exp, df4_train_y_exp, df4_test_y_exp = tts(df4_x, df4_y, test_size=0.10, random_state = 18)
df4_train_x_exp5 = df4_train_x_exp.iloc[0:int((0.3)*len(df4_train_x_exp))]
df4_train_y_exp5 = df4_train_y_exp.iloc[0:int((0.3)*len(df4_train_x_exp))]

#-------------

#Splitting training set into training and validation set
df1_train_x, df1_cv_x, df1_train_y, df1_cv_y = tts(df1_train_x_k, df1_train_y_k, test_size=0.1, random_state = 10)
df2_train_x, df2_cv_x, df2_train_y, df2_cv_y = tts(df2_train_x_k, df2_train_y_k, test_size=0.1, random_state = 10)
df3_train_x, df3_cv_x, df3_train_y, df3_cv_y = tts(df3_train_x_k, df3_train_y_k, test_size=0.1, random_state = 10)
df4_train_x, df4_cv_x, df4_train_y, df4_cv_y = tts(df4_train_x_k, df4_train_y_k, test_size=0.1, random_state = 10)

#Adding 1s in the first column
df1_train_x = np.hstack((np.ones((df1_train_x.shape[0], 1)), df1_train_x))
df1_test_x = np.hstack((np.ones((df1_test_x.shape[0], 1)), df1_test_x))
df1_cv_x = np.hstack((np.ones((df1_cv_x.shape[0], 1)), df1_cv_x))

df2_train_x = np.hstack((np.ones((df2_train_x.shape[0], 1)), df2_train_x))
df2_test_x = np.hstack((np.ones((df2_test_x.shape[0], 1)), df2_test_x))
df2_cv_x = np.hstack((np.ones((df2_cv_x.shape[0], 1)), df2_cv_x))

df3_train_x = np.hstack((np.ones((df3_train_x.shape[0], 1)), df3_train_x))
df3_test_x = np.hstack((np.ones((df3_test_x.shape[0], 1)), df3_test_x))
df3_cv_x = np.hstack((np.ones((df3_cv_x.shape[0], 1)), df3_cv_x))

df4_train_x = np.hstack((np.ones((df4_train_x.shape[0], 1)), df4_train_x))
df4_test_x = np.hstack((np.ones((df4_test_x.shape[0], 1)), df4_test_x))
df4_cv_x = np.hstack((np.ones((df4_cv_x.shape[0], 1)), df4_cv_x))


'''
Run all the code above before trying to classify the different datasets
'''
'''
 #******************************  DATASET 1 **************************
 
# Classification for Dataset 1
alpha = 0.1


#best_theta1 = obj.Fit(df1_train_x, df1_train_y, alpha, 10000)
#best_theta1 = best_theta1.tolist()
#print(best_theta1)

#print(best_theta1.shape())
#print(df1_train_x.shape[1])
#print("Theta for training data:\n",theta_sg1)
#print(obj.Fit.y_h.shape())

#print(np.shape(np.ones(df1_train_x.shape[1])))

predict_1 = obj.predict(best_theta1, df1_test_x) 
#print(predict_1) 
correct1 = [1 if a == b else 0 for (a, b) in zip(predict_1, df1_test_y.tolist())]  

accuracy1 = (sum(map(int, correct1)) / len(correct1)*100)  
print('The Classification Error is {0:.2f}'.format(correct1.count(0)/len(correct1)))
#print('No of mis-classified Training data points: {0}'.format(correct1.count(0)))
print('The accuracy is {0:.2f}%'.format(accuracy1))



 #******************************  DATASET 2 **************************
 
alpha = 1


#best_theta2 = obj.Fit(df2_train_x, df2_train_y, alpha, 12000)
#print(best_theta2)
#print("Theta for training data:\n",theta_sg1)


#print(np.shape(np.ones(df1_train_x.shape[1])))

#predict_2 = obj.predict(best_theta2, df2_test_x) 
#print(predict_2) 
print(alpha)
#correct2 = [1 if a == b else 0 for (a, b) in zip(predict_2, df2_test_y.tolist())]  
#accuracy2 = (sum(map(int, correct2)) / len(correct2)*100) 
print('The Classification Error is {0:.2f}'.format(correct2.count(0)/len(correct2)))
#print('No of mis-classified Training data points: {0}'.format(correct2.count(0)))
print('The accuracy is {0:.2f}%'.format(accuracy2))


 #******************************  DATASET 3  **************************
 
alpha = 0.05 # put value here

#best_theta3 = obj.Fit(df3_train_x, df3_train_y, alpha, 5000)
print(best_theta3)
#print("Theta for training data:\n",theta_sg1)


#print(np.shape(np.ones(df1_train_x.shape[1])))

predict_3 = obj.predict(best_theta3, df3_test_x) #Change here
#print(predict_2) 
print(alpha)
correct3 = [1 if a == b else 0 for (a, b) in zip(predict_3, df3_test_y.tolist())]  # Change here
accuracy3 = (sum(map(int, correct3)) / len(correct3)*100)  
print('The Classification Error is {0:.2f}'.format(correct3.count(0)/len(correct3)))
#print('No of mis-classified Training data points: {0}'.format(correct2.count(0)))
print('The accuracy is {0:.2f}%'.format(accuracy3))



 #******************************  DATASET 4  **************************
 
alpha = 0.6 # put value here


best_theta4 = obj.Fit(df4_train_x, df4_train_y, alpha, 5000)
#print(best_theta2)
#print("Theta for training data:\n",theta_sg1)


#print(np.shape(np.ones(df1_train_x.shape[1])))

predict_4 = obj.predict(best_theta4, df4_test_x) #Change here
#print(predict_2) 
print(alpha)
correct4 = [1 if a == b else 0 for (a, b) in zip(predict_4, df4_test_y.tolist())]  # Change here
accuracy4 = (sum(map(int, correct4)) / len(correct4)*100)  
print('The Classification Error is {0:.2f}'.format(correct4.count(0)/len(correct4)))
#print('No of mis-classified Training data points: {0}'.format(correct2.count(0)))
print('The accuracy is {0:.2f}%'.format(accuracy4))

'''

 #****************************** KFold CV for DATASET 1 **************************

obj.cv(df1_x, df1_y, 0.1, 10000, 5)
input('stop')
#obj.cv(df1_train_x_k, df1_train_y_k, 0.1, 10000, 5)

 #****************************** KFold CV for DATASET 2 **************************
 
#obj.cv(df2_train_x_k, df2_train_y_k, 1, 30000, 5)

#****************************** KFold CV for DATASET 3 **************************
 
#obj.cv(df3_train_x_k, df3_train_y_k, 0.05, 15000, 5)

#****************************** KFold CV for DATASET 4 **************************
 
#obj.cv(df4_train_x_k, df4_train_y_k, 0.0006, 2000, 5)


#****************************** Experiment on Dataset 3 - stopping criteria using error **************************
 

alpha = 0.05 # put value here

best_thetaz = obj.FitExp(df3_train_x, df3_train_y, alpha, 5000)
print(best_thetaz)
#print("Theta for training data:\n",theta_sg1)


#print(np.shape(np.ones(df1_train_x.shape[1])))

predict_z = obj.predict(best_thetaz, df3_test_x) #Change here
#print(predict_2) 
print(alpha)
correctz = [1 if a == b else 0 for (a, b) in zip(predict_z, df3_test_y.tolist())]  # Change here
accuracyz = (sum(map(int, correctz)) / len(correctz)*100)  
print('The Classification Error is {0:.2f}'.format(correctz.count(0)/len(correctz)))
#print('No of mis-classified Training data points: {0}'.format(correct2.count(0)))
print('The accuracy is {0:.2f}%'.format(accuracyz))


# KFold implementation of the same
obj.cvexp(df3_train_x_k, df3_train_y_k, 0.05, 15000, 5)



#****************************** Experiment on Dataset 3 - train set size **************************
 

# KFold implementation 
obj.cv(df4_train_x_exp5, df4_train_y_exp5, 0.05, 500, 5)



alpha = 0.05 # put value here

best_theta = obj.Fit(df4_train_x_exp1, df4_train_y_exp1, alpha, 500)
print(best_theta)
#print("Theta for training data:\n",theta_sg1)


#print(np.shape(np.ones(df1_train_x.shape[1])))
df4_test_x_exp = np.array(df4_test_x_exp)
predict_3 = obj.predict(best_theta, df4_test_x_exp) #Change here
#print(predict_2) 
print(alpha)
correct3 = [1 if a == b else 0 for (a, b) in zip(predict_3, df4_test_y_exp.tolist())]  # Change here
accuracy3 = (sum(map(int, correct3)) / len(correct3)*100)  
print('The Classification Error is {0:.2f}'.format(correct3.count(0)/len(correct3)))
#print('No of mis-classified Training data points: {0}'.format(correct2.count(0)))
print('The accuracy is {0:.2f}%'.format(accuracy3))


