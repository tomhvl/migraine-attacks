# -*- coding: utf-8 -*-
"""
Script to Extract TSF features and save to disk for future use.

Created on Thu Apr 06 21:27:36 2017

@author: Tomba
"""

import pandas as pd
import numpy as np
from datetime import datetime
import migraine_processing as mig
import classifiers as css

import tsfresh as tsf

if __name__=="__main__":

    
    
    label = 'start'
    start_time_main = datetime.now()
    # done (400,200), (1000,100), (1000, 25), (500,25), (500,50), (500,100),
    #   (300,30), (1000,50), (300,50), (200,25)
    for win, step in list([(200,25)]):
        pre = step
        post = 0
    
        for fileid in range(2, 3):#len(mig.CASE_NAMES)-1):
            df = mig.getCaseData(mig.CASE_NAMES[fileid])
            start_time = datetime.now()
            
            print '*** Filename: ', mig.CASE_NAMES[fileid]
            print 'pre/post: %d/%d  win/stride: %d/%d  label:%s' % (pre,post,win,step,label)
        
            if fileid==2:
                df = df.loc[:'2002-06-24 00:00:00']  # for the sb_dataset
        
            df.flag = mig.labelSequence(df, pre, post)
               
            print 'Total:Target (count): ', df.flag.count(), df.flag[df.flag].count()
        
       
            # ************** Feature calculation step: **************************
            # calculate tsfresh feature set, passing window and step size
            name = "features/X_y_feats_ts%dx%d_pp%d-%d_cid%s_filter.csv" % (win, step, 
                                            pre, post, fileid)
            
            X, y = css.makeTSFeatureSet(df, win, step, label_by=label)
            Xres = np.append(X, np.reshape(y, (-1,1)), axis=1)
            result = pd.DataFrame(Xres)
            result.to_csv(name, sep=';', mode='a')
        
            print 'Saving to %s' % name
            print 'Elapsed time %.2f mins' % ((datetime.now() - start_time).total_seconds()/60.0)
        
        
    print 'Total elapsed time %.2f mins' % ((datetime.now() - start_time_main).total_seconds()/60.0)
    
    