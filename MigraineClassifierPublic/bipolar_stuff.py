# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 22:03:30 2017

@author: TaTuLec
"""
import os
import pandas as pd
import numpy as np
import migraine_processing as mig
import classifiers as css

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def findDistances(df, seqpairs):
    """ Compare distances between flagged sequences supplied as pairs (beg, end)
    Returns a dict with 'beg' as key followed by a list of distances to each of
    the pairs in 'seqpairs'
    """

    result = {}   
    for (beg, end) in seqpairs:      
        sqa = df.value.loc[beg:end]
        tmp = []
        for (beg_b, end_b) in seqpairs:
            if True:#beg != beg_b:
                sqb = df.value.loc[beg_b:end_b]
                distance, path = fastdtw(sqa.values, sqb.values, dist=euclidean)
                tmp.append(distance)
        
        result.update({beg: tmp})
        
    return result
    
def findMatches(df, query, size=60, step=30, noflag=True):
    """ Find matches of query in df, using step-size stride and size window
    size. noflag=True will ignore all flagged patches. Takes a pd.Series as
    query.
    Returns Series of distances indexed by their original timestamps"""
    
    rindex = []
    result = []
    for (start, end) in mig.getWindows(df.value, size, step):
        seq = df.value.iloc[start:end]

        distance, path = fastdtw(seq.values, query.values, dist=euclidean)
        result.append(distance)
        rindex.append(df.index[start])
    
    result = pd.Series(result, index=rindex)
    return result
    


    
def compareNightQualityWithMigraineAttacks(fileid):
    """ Experiment that shows migraine attacks alongside the night quality
    values as they are being calculated atm. Remember that a value for any
    day is the calculation of night quality from the night preceding it.
    This is a complete all-in-one experiment. The values should be ideally
    low for night preceding migraine attacks."""
    
    df = mig.getCaseData(mig.CASE_NAMES[fileid])
    
    if fileid==2:
        df = df.loc[:'2002-06-24 00:00:00']  # for the sb_dataset
        
    nq = css.featureNightQuality(df)
    df['nq'] = nq.values
    flag = df.index[df.flag==True]
    evry = df.index[df.index.indexer_at_time('12:00:00')]
    
    # 1. check for existence of each (date w/o time) in flagged days
    # 2. assign timestamp extracted from flag into migraine_ts
    # 3. extract iloc index from migraine_ts (timestamp)
    result = pd.DataFrame(columns=['nq', 'attack'])
    for day in evry:
        if day.date() in flag.date:
            migraine_ts = flag[flag.date == day.date()][0]
            index = df.index.get_loc(migraine_ts)
            print 'attack ', migraine_ts, df.nq[index]
            result = result.append({'nq':df.nq[index], 'attack': 1}, ignore_index=True)
        else:
            index = df.index.get_loc(day)
            print 'normal ', day, df.nq[index]
            result = result.append({'nq':df.nq[index], 'attack': 0}, ignore_index=True)

    
    # set colors according to results
    colors = [(i,0,0.2) for i in result.attack]
    result.nq.plot.bar(title='sleep quality vs attacks via count(500) method', colors=colors)
    
    

if __name__=="__main__":
    fileid = 0
    
    
    df = mig.getCaseData(mig.CASE_NAMES[fileid])
    
#    if fileid==2:
#        df = df.loc[:'2002-06-24 00:00:00']  # for the sb_dataset
#        
    #
    #
    path = 'C:\Users\Tomba\Dropbox\UIB\SENSORDATA' 
    files = os.listdir(path)
    #
    files_txt = [i for i in files if i.endswith('.AWD')]
    #print sorted(files_txt), len(files_txt)
    # for each file read
    # find the morning activity of every day
    # add it to a dictionary {'filename': [(fra, til), (fra1, til1),...]
    # OR calculate features straight off and save the feature vector..
    
    
    
    #df = mig.getCaseData(mig.CASE_NAMES[fileid])
       
    # get modified version of label column (extended from single point)
    # the settings here have influence on how the getAggregate function will
    # pickup the target labels!
    #df.flag = mig.labelSequence(df, 0, 5)

    
    # select every day between 03:00 and 12:00 (hours)
    # hour = df.index.hour; selector = ((hour>3) & (hour<12)); dfm = df[selector]
    # dfmorning = mig.getMorningActivity(dfm, 5, 6, 64); dfm.iloc[dfmorning]
    #
    #tmp = df.loc['2002-06-07 05:00:00':'2002-06-07 12:00:00']
    #tmpx = mig.getMorningActivity(tmp, 4, 15)
    #df = df.resample('2t').mean()
    
    # example of deriving sleeping periods + visualization
    # dfa = df.iloc[:9000]; sleep_a = mig.sleepTime(dfa); 
    # dfa = dfa.assign(sleep=sleep_a.values); 
    # dfa.plot(secondary_y=['sleep'], kind='area', alpha=0.2)
    
    print 'Total:Target (count): ', df.flag.count(), df.flag[df.flag].count()

    # Feature calculation step:
    # calculate tsfresh feature set, passing window and step size
    #X, y = css.makeTSFeatureSet(df, 200, 5, False)

    # get every 5th datapoint in window as feature
    #feats = makeSerialFeatureSet(df)
    
    # get basic stats as features
    #X, y = css.makeBasicFeatureSet(df, 2000, 5, label_by='start')

    #css.runGS(X,y,3)

    #print('Total : Processed (count): ', X.shape, y[y>0].count())
    
    
#    x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
#    y = np.array([[2,2], [3,3], [4,4]])


    # ***** DTW output  *****

#    import scipy.spatial.distance as spd
#
#    distances.sort_values().head()
#    distances.sort_values().tail()
#    #choose from list..
#    
#    other_id = distances.index.get_loc('2005-08-23 08:34:00')
#
#    spd.correlation(query.values, df.value[other_id:other_id+121].values)
#    spd.correlation(query.values, query.values)
#    spd.correlation(query.values, df.value[other_id+999:other_id+1120].values)
#
#    spd.cosine(query.values, query.values)
#    spd.cosine(query.values, df.value[other_id:other_id+121].values)
#    spd.cosine(query.values, df.value[other_id+999:other_id+1120])
#
#    spd.euclidean(query.values, query.values)
#    spd.euclidean(query.values, df.value[other_id:other_id+121].values)
#    spd.euclidean(query.values, df.value[other_id+999:other_id+1120].values)

    
    


