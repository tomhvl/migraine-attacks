# -*- coding: utf-8 -*-
"""
Read logs and extract stats into summary file for classification experiments.

(Logs made by redirecting output from experiment runs to result files)

Created on Thu Apr 13 15:34:35 2017

@author: Tomba
"""
import re, os, itertools
import pandas as pd
import numpy as np
import migraine_processing as mig
import classifiers as css

#filename = "experiments/Results_BAS_02.txt"
#filename = "experiments/Results_200x25_variants.txt"
#filename = "experiments/Results_200x25_tsfresh.txt"
#filename = "experiments/Results_200x25_tsfresh_filter.txt"
#filename = "experiments/Results_200x25_basic_end.txt"
filename = "experiments/Results_200x25_basic_filter.txt"
#filename = "experiments/Results_2000x5_tsfresh.txt"
#filename = "experiments/Results_500x25_tsfresh.txt"
#filename = "experiments/Results_500x25_basic.txt"
#filename = "experiments/Results_500x25_basic_filter.txt"


numbers = re.compile('\d+(?:\.\d+)?')
imbClass = {'SMOTE':1, 'ClusterCentroids':2, 'SMOTETomek':3, 'None':0}

main = pca = poly = bstk = imbal = 0
temp, result = [], []
alg = exs = fea = tar = pr = rec = f1 = sp = 0
time = -1
 
with open(filename, 'r') as f:
    content = f.readlines()

for i in range(0, len(content)):
    line = content[i]
    
    
    if line.startswith("pre/post"):
        main += 1
        exs = fea = tar = alg = pca = poly = bstk = imbal = 0
        time = -1
        
        
    if line.startswith("pca_target:"):
        pca, poly, bstk = map(int, numbers.findall(line))

    if line.startswith("Imbalance:"):
        line = line[line.rfind("Imbalance:")+10:line.find("(")].strip()
        imb = imbClass.get(line)
        if imb is None:
            imb = -1      
        
    if line.startswith("('Total : Processed (count):"):
        exs, fea, tar = map(int, numbers.findall(line))
             
    # sentence = re.sub(r"\s+", "", sentence, flags=re.UNICODE)
    if line.translate(None, ' \t').startswith("ACCMCCKAP"):
        line = content[i+1].strip()
        acc, mcc, kap, auc, geo, iba = map(float, numbers.findall(line))
        #geo = iba = -1
        line = (content[i+2] + content[i+3]).strip()
        tn, fp, fn, tp = map(int, numbers.findall(line))
       
        temp = np.array([main, alg, acc, mcc, kap, auc, geo, iba, tn, fp, fn, tp])
        i += 2
        
    if line.lstrip().startswith("avg / total"):
        pr, rec, f1, sp = map(float, numbers.findall(line))
        temp = np.append(temp, [pr, rec, f1, sp, pca, poly, bstk, exs, fea, tar,
                                imb, time])
        
        alg += 1
        result.append(temp)#.flatten()) 
        
    
    if line.startswith("Elapsed time"):
        time,  = map(float, numbers.findall(line))
        # stamps the total test time on all the alg runs in this group going backwards
        for i in range(len(result)-1, len(result)-alg-1, -1):
            result[i][-1] = time


                    

    
# did we get the complete data?
if time < 0:
    print 'Last test incomplete!'
    
    
dfresult = pd.DataFrame(result, 
                        columns=['test','alg', 'acc', 'mcc', 'kap', 'auc', 'geo',
                                 'iba', 'TN', 'FP', 'FN', 'TP', 'pr', 're', 'f1',
                                 'sp', 'pca', 'poly', 'ksel','exs', 'feats', 
                                 'tvars', 'imb', 'time'])
    
dfresult[['test','alg','TN','FN','TP','FP','sp','pca','poly','ksel','exs','feats',
          'tvars', 'imb']] = dfresult[['test', 'alg', 'TN','FN','TP','FP', 'sp',
            'pca', 'poly', 'ksel', 'exs', 'feats', 'tvars', 'imb']].astype(int)

print dfresult.head()

# output to file
outfile = filename.replace("experiments/", "experiments/SUMMARY_")
outfile = outfile.replace("txt", "csv")
dfresult.to_csv(outfile, sep=",", header=True, index=True)
print "Saved: ", outfile
