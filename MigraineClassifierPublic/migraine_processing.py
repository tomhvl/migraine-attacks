# -*- coding: utf-8 -*-
"""
Various methods for processing the migraine files.



"""

import pandas as pd
import numpy as np
#from scipy import stats


# filenames
#FOLDER_NAME = "..\..\PREPPED_DATAPREPPED_DATA"
CASE_NAMES = ["stm_aasane16_modified.csv", "stm_aasane18_modified.csv", \
              "stm_sb_modified.csv", "stm_aasane01_03_modified.csv",
              "stm_kontroll02 (edited).csv", "stm_fusa04 (edited).csv"]


def getIndexSequence(df, index, pre, post):
    """ Grab a chunk from the dataframe at index position -pre steps to
    +post steps, i.e. slice, returning the indexes """

    # create timedelta for pre and post periods, return the slice
#    predt = pd.to_timedelta(pre, unit='m')
#    postdt = pd.to_timedelta(post, unit='m')
#    return df.loc[(index-predt):(index+postdt)].index

    # 'pre' gets reduced by 1 because of the way slicing works in pandas
    # and the last index in a slice is not included. This caused problems
    # with 0 valued post values and label count was off by one.
    # update: it causes inconsistent behaviour on pre = 1 vs pre = 0 
    if pre > 0:
        pre -= 1

    idx = df.index.get_loc(index)
    
    #print pre,post,idx, index
    return df.iloc[max(0,idx-pre):idx+max(post,1)].index


def labelSequence(df, pre, post):
    """ Create a series based on flag column where the flags are spread over
    specified length of a sequence surrounding the original flagged index.
    Propagates the flag pre-steps before and post-steps after original label.
    Return a new series with expanded flags according to pre/post given."""
        
    maskidx = df.index[df.flag] # series of (DT) indexes with true flags
    result = df.flag.copy()
    for idx in maskidx:
        mask = getIndexSequence(df, idx, pre, post)
        result[mask] = True

    return result


def getCaseData(filename):
    """ Read a converted .AWD to csv file as dataframe, set 'M' to True, rest
    to False, set date as index, and fix column names. Return dataframe. """
    filename = filename
    casedata = pd.read_table(filename, sep=',', skiprows=3, parse_dates=True, \
    infer_datetime_format=True, index_col=0, na_filter=False, \
    names=["time", "value", "flag"], true_values=['M'], false_values=[''])
    return casedata


def getFeature(df, start, end):
    """ Create basic feature set based on window size and step.
    Returns list with windows stats as calculated feature values"""

    return [df[start:end].mean(),
            df[start:end].std(),
            df[start:end].skew(),
            df[start:end].kurt(),
            df[start:end].quantile(0.25),
            df[start:end].quantile(0.75),
            df[start:end].quantile(0.90),
            df[start:end].quantile(0.15),
            df[start:end].median(),
            df[start:end].mad(),
            df[start:end].sem(),
            df[start:end].var(),
            df[start:end].autocorr(1),
            df[start:end].autocorr(2),
            df[start:end].autocorr(3),
            df[start:end].autocorr(4),
            df[start:end].autocorr(5),
            np.append(df[start:end].mode(), -1)[0]
           ]


def getWindows(df, size=75, step=15):
    """ Slices of simple windows of df.size, with step slide option"""
    start = 0
    while start+size < df.count():
        yield start, start + size  #pd.to_timedelta(size, unit='m'))
        start += step

def shift(xs, n):
    """ Shift a numpy array n-steps. Ripped from StackOverflow by chrisaycock.
    Slight modification replaces nan's with 0's"""
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0.0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0.0
        e[:n] = xs[-n:]
    return e


def getAggregate(df, step):
    """ Create a by-step aggregate series/dataframe of df.
    Returns: a dataframe with [index, value, flag] ."""
    #df = mig.getAggregate(df, 2)
    #df = df.resample('2t').mean()   :alternate resampling method?

    idx, res, flag = [], [], []

    for (start, end) in getWindows(df.value, step, step):
        idx.append(df.index[end])
        res.append(df.value.iloc[start:end].mean())
        flag.append(df.flag.iloc[start] & df.flag.iloc[end])

    return pd.DataFrame.from_records({'value':res, 'flag':flag}, index=idx,
                                     columns=['value', 'flag'])


def getMorningActivity(df, start=4, end=4, period=64):
    """ Find morning activity period as described in bipolar activity
    study, 64 mins of activity in the  morning i.e. a period of consecutive
    datapoints with no more than zero_boundary x 0-activity levels.
    This does not take into account other attributes than 'value'.
    Returns: a list of integer indexes where such sequences start.
    """

    # find valid intervals of length >= period
    count = end    
    # starting index of current unfinished sequence
    start = 0    
    # flags true when sequence is of at least length == period
    ready = False    
    # sequence element increment, if 0 then sequence not started yet
    active = 1
    # day info (keeps track of current day)
    day = -1
    
    result = []
    for i in range(end, len(df.value)):
        #  set ready flag if period requirement has been met
        ready = (count == period+start) or ready

        # time to append start of just ended sequence
        if ready and count == 0:
            ready = False
            if day != df.index.dayofyear[start]:
                day = df.index.dayofyear[start]
                result.append(start)

        # only when end boundary is reached can we reset count & active
        if df.value[i-end:i].sum() > 0:
            count += active
        else:
            count = 0
            active = 0
            
        if df.value[i-start:i].sum() > 0 and active == 0:
            # possible start of a new sequence here and activate
            start = i-1
            active = 1

    # make sure last found sequence is appended if flagged ready
    if ready and day != df.index.dayofyear[start]:
        result.append(start)

    return result


# night detector
def sleepTime(df, lev1=20, lev2=120, lev3=30, init_state=True):
    """ Defines the sleeping period as based on starting number of zero
    activity steps and broken by a sequence of higher activity steps as
    described in (Nova et. al. 2014)
    init_state defines starting state (sleep=0, wake=1)
    Returns a binary Series object. 
    """
    REQ_PASSIVE_RATE = 0.75   # at least 75% of values
    
    sleep = pd.Series(np.full(len(df), -1, dtype=int), index=df.index)

    dfmed = df.value.rolling(lev1, center=True).median()
    pstart = []
    
    lev1_pre = lev1/2
    lev1_pos = lev1-lev1_pre
    
    for i in range (lev1, len(dfmed)-lev1_pos):
        # if lev1 size window is 0, add to possible start of sleep list
        # and step over to next possible sleep start
        if dfmed[i-lev1_pre:i].sum() == 0 and dfmed[i+1:i+lev1_pos].sum() == 0: 
            pstart.append(i-lev1)
            i += lev1_pos
        
    # alternative:
    # dfmed.rolling(lev1).sum()
    #pend = dfmed.index[(dfmed.rolling(lev3).min() > 0)]
    
    laststate = -lev2
    for i in pstart:
        # ignore possible start state earlier flagged as active sleep states
        if i - laststate < lev2:
            continue
        # if given percentage is 0, we have active sleep state
        if dfmed[i:i+lev2].quantile(REQ_PASSIVE_RATE) == 0:
            laststate = i
            sleep[i] = 0   
        
    for i in range (lev1_pre, len(dfmed)-lev3):
        # if lev3 size window is non 0, then end of active asleep
        if dfmed[i:i+lev3].min() > 0: 
            i += lev3
            sleep[i] = 1            
    
    laststate = init_state
    for i in range(len(sleep)):
        if sleep[i] != -1:
            laststate = sleep[i]
        else:
            sleep[i] = laststate
   
    return sleep
              


         
def getFlaggedSeqList(df, pre, post):
    """ Creates a list of sequence coordinates (slices) relative to set flags.
    Should be used on original dataframe with single point labels. i.e. single
    flags, not on dataframe earlier modified by labelSequence().
    Returns a list of tuple pairs, (start:end)
    """
    maskidx = df.index[df.flag] # series of (DT) indexes with true flags
    result = []
    for idx in maskidx:
        maskseq = getIndexSequence(df, idx, pre, post)
        result.append((maskseq[0], maskseq[-1]))

    return result
            
           
    
        
            
            
    


