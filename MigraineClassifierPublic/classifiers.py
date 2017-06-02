# -*- coding: utf-8 -*-
"""

Includes some supporting methods for the experiments.

Created on Thu Feb 23 14:38:35 2017

@author: TaTuLec
"""
import pandas as pd
import numpy as np
from scipy import signal

import migraine_processing as mig
from imblearn.metrics import make_index_balanced_accuracy as iba
from imblearn.metrics import geometric_mean_score as geomsc


from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn import preprocessing, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
 
import tsfresh as tsf


def calcLabelPosition(start, end, label_by):
    """ Returns the label position """
    # infer a label for this window (based on 1st element label in original)
    #if middle of win is labelled we use that label as target
    #label_idx = df.flag[start+(end-start)/2]
    # if 1st (start) or last (end) element of sequence is label
    res = start
    if label_by[:4]=='cent': 
        res = start+(end-start)/2
    elif label_by=='end':            
        res = end
    
    return res
            
def featureNightQuality(df, sleep):
    """ Last night sleep quality measured as mean of whole sleep period
    Returns a series containing last nights score. This score is propagated
    until the next update, i.e. next night-to-day transition."""
    MIN_REQ_PERIOD = 120
    
    sleep_status = sleep
    sleep_score = pd.Series(np.full(len(df), -1.0))    
    last = sleep_status[0]
    start = 0
    score = 0.0
    for i in range (0, len(df.value)):      
        # night just changed to day        
        if last - sleep_status[i] == -1:
            if i-start > MIN_REQ_PERIOD:
                # based on mean value        
                #score = df.value.iloc[start:i].mean() 
                #score = df.value.iloc[start:i].sum()
                # based on frequency of low-values
                score = df.value.iloc[start:i][df.value>500].count()
        # day changed to night
        if last - sleep_status[i] == 1:
            start = i
            
        last = sleep_status[i]
        sleep_score[i] = score
    
    return sleep_score

def featureTimeOfDay(df, sleep):
    """ Encode time of day: night, morning, afternoon """
    MORNING = ('05:00:00','13:00:00')
    
    result = pd.Series(data=np.zeros(len(df)), index=df.index, dtype='int32')
    result[:] = 2
    result[sleep==0] = 0
    morning = df.index.indexer_between_time(MORNING[0], MORNING[1])
    result[morning] = 1

    return result    
    
    
  
def makeSerialFeatureSet(df, size, step, aggstep=2, label_by='start'):
    """ Creates features from datapoints in windowed tseries.
    OBS: Doesnt seem to work at all as normal features..."""
    
    temp = mig.getAggregate(df, aggstep)
    feats, lab= [], []
    idx = 0 # OBS id column removed in makeFeature() only required for tsfresh 
    for (start, end) in mig.getWindows(temp.value, size, step):
        tmp = list(temp.value[start:end])
        # maybe a questionable way of inserting an id into each list header!
        #tmp.insert(0,idx)                
        feats.append(tmp)    
        lab.append(int(df.flag[calcLabelPosition(start, end, label_by)]))
        #idx += 1

    # could also go directly through  np.array(feats)
    X = np.matrix(np.nan_to_num(feats))
    y = pd.Series(lab)    
    return X, y


def makeSpecialFeatureSet(df, size, step, label_by='start'):
    """ Create special features only """
    feats, lab= [], []
    sleep = mig.sleepTime(df)
    sleep_scores = featureNightQuality(df, sleep)
    #time of day features
    tod = featureTimeOfDay(df, sleep)
    
    for (start, end) in mig.getWindows(df.value, size, step):
        label_pos = calcLabelPosition(start, end, label_by)
        
        # get the mode (most frequent value) of this window in tod 
        tod_value = tod[start:end].mode()[0]
        
        tmp = [sleep_scores[label_pos], tod_value]
        feats.append(tmp)
        lab.append(int(df.flag[label_pos]))
        
    X = np.matrix(feats)
    y = pd.Series(lab)
    return X, y
 

def makeLaggedFeatureSet(X, lags=2):
    """ Create lagged features out of existing feature(s). Takes a numpy
    matrix as input and lags all of the columns in it by 'lags'. Careful
    not to send in the index column as it will be shifted/modified like
    the rest. This should be used as an additional step after creating
    a feature matrix to supplement it with additional lagged features.
    Returns a new matrix with lagged columns."""
    res = np.roll(X, 1, 0)
    res[0,:] = 0.0
    rolls = 1
    
    # rolls all columns down 'lag' times and fill the newly incoming
    # rows with zeros.
    while lags > rolls:
        rolls += 1
        tmp = np.roll(X, rolls, 0)
        tmp[0:rolls,:] = 0.0
        res = np.append(res, tmp, axis=1)
    
    return res

       
def makeBasicFeatureSet(df, size, step, label_by='start'):
    """ Create an array (list of lists) of features.""" 
    feats, lab= [], []
    #idx = 0
    for (start, end) in mig.getWindows(df.value, size, step): 
        
        # *** add filtering ***
#        #scaled_value = (df.value.values/np.linalg.norm(df.value.values))
#        filt_win = signal.tukey(end-start, 0.6)        
#        currentseq = df.value[start:end] * filt_win
#        tmp = mig.getFeature(currentseq, 0, -1)
#        
        tmp = mig.getFeature(df.value, start, end)
        
        #tmp.insert(0,idx)
        #idx += 1    

        feats.append(tmp)
        label_pos = calcLabelPosition(start, end, label_by)
        lab.append(int(df.flag[label_pos]))
        
#        if df.flag[label_pos]:
#            print start, end, label_pos, df.flag[label_pos]
        
        
    # could also go directly through  np.array(feats)
    X = np.matrix(np.nan_to_num(feats))
    y = pd.Series(lab)
    
    return X, y

def makeTSFeatureSet(df, size=45, step=15, label_by='start'):
    """ creates features using TSF library 
    Returns: numpy feature matrix, pd.Series target labels"""
    fsX, fsy = createFreshData(df, size, step, label_by)
    
    fsettings = tsf.feature_extraction.FeatureExtractionSettings()
    fsettings.IMPUTE = tsf.utilities.dataframe_functions.impute
    feats = tsf.feature_extraction.extract_features(fsX, \
                feature_extraction_settings=fsettings, \
                column_id='id', column_value='value')
    
    X = feats.as_matrix()
    y = fsy
    
    return X, y


def createFreshData(df, size, step, label_by='start'):
    """ prepare dataframe for tsfresh processing . Care must be taken with
    the stepping window params as the computational complexity may increase
    drastically.
    Returns: tsfresh-ready dataframe, target pd.Series with corresp. id-col"""
    
    # for tsfresh feature extraction: 
    # each feature window must be indexed same as the target-array
    # create a new dataframe with each feature window in sequence 
    # this leads to (possibly) substantial overlap of data points and
    # therefore increased computational time - depends on overlap size
    # 50/25 win in 5000 length data doubles the no.of datapoints
    # 50/10 would 5x increase number of processed datapoints!
    
    # OBS: seems pd.append and list.append behave differently (in-place/non ip)
    ind = pd.Series()
    res = pd.Series()
    lab = []
    idx = 0
    for (start, end) in mig.getWindows(df.value, size, step):
        
        # create common index for each row in current window
        #tmp = np.empty(size); tmp.fill(idx)  #this method created floats..
        tmpid = np.full((size), idx)
        ind = ind.append(pd.Series(tmpid), ignore_index=True)
        
        
        # tukey filter
        filt_win = signal.tukey(end-start, 0.6)        
        currentseq = df.value[start:end] * filt_win
        tmp = currentseq.values
        
        #tmp = df.value[start:end].values
        
        # fill in the values from original dataframe
        res = res.append(pd.Series(tmp), ignore_index=True)
        
        # set label appropriately to label_by
        if label_by=='start':
            lab.append(int(df.flag[start]))
        else:
            lab.append(int(df.flag[end]))
        idx += 1
        
#        lab.append(int(df.flag[calcLabelPosition(start, end, label_by)]))

    return pd.DataFrame(data={'id': ind, 'value': res}), pd.Series(lab)

    
def performScaling(train, test):
    """ Scaling in one method, return transformed train and test, according
    to fit_transform on train..."""
    #scaler = preprocessing.MinMaxScaler().fit(train)
    scaler = preprocessing.StandardScaler().fit(train)

    X_train = scaler.transform(train)
    X_test = scaler.transform(test)
    return X_train, X_test

