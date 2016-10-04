# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:00:49 2016

@author: Mike Crabtree
"""

#Import pandas and numpy for data manipulation and computation
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

#When combined, this dataset becomes quite large, grabbing the compressed sparse row
    #matrix function from scipy
from scipy.sparse import csr_matrix, hstack

#Import the scikit-learn library for modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

#set the directory for the data, since this notebook is in the root for the conda env, I stashed all
    #the data into a directory called Data
datadir = 'Data/'

#Now read in all the data
#First, sniggity snag the data gender_age_train
    #and set the index column
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')

#Do the same for gender_age_test
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')

#And again for phone_brand_device_model
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

#Get rid of duplicate device ids in phone, keep the first one that occurs
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

#Same for events, and correctly set the datatype for the timestamp feature
    #to be a date-time YYYY-MM-DD 00:HH:MM
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')

#Same deal for app_events but not the is_installed feature
    #the idea is that if the app is active, it is installed
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})

#Same for label_categories
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

#Revisit this later
#labelcats = pd.read_csv(os.path.join(datadir,'label_categories.csv'))

#Look at the first 5 tuples for the gender age group training set
gatrain.head(n=5)

#Look at the first 5 for the devices and their associated brands and models
phone.head(n=5)

#Look at the first 5 for the phone events data set
events.head(n=5)

#Look at the first 5 for the table that ties the application id to the mobile event
appevents.head(n=5)

#Look at the table that associates mobile applications
applabels.head(n=5)

#Using one-hot encoding, create two columns that show which train or test set row a particular device_id belongs to
    #one-hot encoding is for some models that cannot encode a n way choice properly
#http://stackoverflow.com/questions/17469835/one-hot-encoding-for-machine-learning
gatrain['trainrow'] = np.arange(gatrain.shape[0])

gatest['testrow'] = np.arange(gatest.shape[0])

#Because the dataset information is from Chinese users, many of the phone brands and models are in Hanzi, not english
    #UTF-8 Unicode
#Start some preprocessing from scikit-learn to transform the labels
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])

#Propogate the associated phone brands to both the train and test sets given
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']

#Create compressed sparse row matrices for both train and test
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))

Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))

#Double check the dimensions of the two csr matrices
print('feature matrix shapes: train {}, test {}'.format(Xtr_brand.shape, Xte_brand.shape))

#Have a look at the top 5 tuples for the gatrain dataset to look at the encoded brands
gatrain.head()

#Many of the cellphone models are also Hanzi, not English characters (pinyin)
#Use the same type of preprocessing to transform the models into categorical levels represented numerically
modelencoder = LabelEncoder().fit(phone.phone_brand.str.cat(phone.device_model))
phone['model'] = modelencoder.transform(phone.phone_brand.str.cat(phone.device_model))

#Propogate the phone models the same way as the brand
gatrain['model'] = phone['model']
gatest['model'] = phone['model']

#Same thing with the compressed sparse row matrices as done for the brands
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))

#And again with double checking the dimensions
print('feature matrix shapes: train {}, test {}'.format(Xtr_model.shape, Xte_model.shape))

#Take a look at gatrain again to see the new numerical levels for models
gatrain.head()

#Do the same kind of transformation to the app_ids
applabelenc = LabelEncoder().fit(appevents.app_id)
appevents['app'] = applabelenc.transform(appevents.app_id)

#Find the number of apps/levels
napps = len(applabelenc.classes_)

#Merging all the tables together becomes massive
#Merge and group by device ids and the number of times the app was used such that
    #The id, app, and size is what makes each tuple unique
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
                       
deviceapps.head(n=10)

#Drop the tuples where trainrow has an NA
data = deviceapps.dropna(subset=['trainrow'])

#Same thing with the compressed sparse row matrices as before
Xtr_app = csr_matrix((np.ones(data.shape[0]), (data.trainrow, data.app)), 
                      shape=(gatrain.shape[0],napps))

#Drop the tuples where testrow has an NA
data = deviceapps.dropna(subset=['testrow'])

Xte_app = csr_matrix((np.ones(data.shape[0]), (data.testrow, data.app)), 
                      shape=(gatest.shape[0],napps))

print('feature matrix shapes: train {}, test {}'.format(Xtr_app.shape, Xte_app.shape))

data.head()

#From the appevents dataset grab the location feature vector from each unique id
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = applabelenc.transform(applabels.app_id)


labelencoder = LabelEncoder().fit(applabels.label_id)

applabels['label'] = labelencoder.transform(applabels.label_id)

nlabels = len(labelencoder.classes_)
print('Number of classes/nlabels: {}'.format(nlabels))

#Group and aggregate the devices and
devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                #Make sure to use left joins here
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()

data = devicelabels.dropna(subset=['trainrow'])

Xtr_label = csr_matrix((np.ones(data.shape[0]), (data.trainrow, data.label)), 
                      shape=(gatrain.shape[0],nlabels))

data = devicelabels.dropna(subset=['testrow'])

Xte_label = csr_matrix((np.ones(data.shape[0]), (data.testrow, data.label)), 
                      shape=(gatest.shape[0],nlabels))

print('feature matrix shapes: train {}, test {}'.format(Xtr_label.shape, Xte_label.shape))

#Stack the csr's created earlier into new test and train sets
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')

print('Final matrix shapes: train {}, test {}'.format(Xtrain.shape, Xtest.shape))

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)

nclasses = len(targetencoder.classes_)
print('Number of features being used for modelling: {}'.format(nclasses))

def score(clf, random_state = 0):
    
    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    
    pred = np.zeros((y.shape[0],nclasses))
    
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest,:] = clf.predict_proba(Xte)
        
        # Downsize to one fold only for kernels
        return log_loss(yte, pred[itest, :])
        print("{:.5f}".format(log_loss(yte, pred[itest,:])), end=' ')
        
    print('')
    return log_loss(y, pred)

#Look at different c values to determine effects on predictions in logistic regression
ranges = np.logspace(-4,0,4)
res = []
for C in ranges:
    
    res.append(score(LogisticRegression(C = C)))
    
plt.semilogx(ranges, res,'-o');
print(res)

#Looks like the best value is somewhere around 0.02
cvals = [0.01, 0.013, 0.019, 0.02, 0.023, 0.29, 0.03]
scores=[]
for c in cvals:
    scores.append(score(LogisticRegression(C = c)))
print(scores)

#Looks like .023 is narrowed down sufficient to generate a solution file for the first run
score(LogisticRegression(C=0.023, multi_class='multinomial',solver='lbfgs'))

clf = LogisticRegression(C=0.023, multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain, y)

prediction = pd.DataFrame(clf.predict_proba(Xtest), index = gatest.index, columns=targetencoder.classes_)
prediction.head(n=10)

prediction.to_csv('my_submission(logi).csv',index=True)                       