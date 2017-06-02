# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:55:00 2017

Pipeline experiments.

This is not intended to be user-friendly, sorry. 

@author: TaTuLec
"""
import sys
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import migraine_processing as mig
import classifiers as css

from imblearn.metrics import make_index_balanced_accuracy as iba
from imblearn.metrics import geometric_mean_score as geomsc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import (TomekLinks, ClusterCentroids, CondensedNearestNeighbour,
                                     InstanceHardnessThreshold, AllKNN)
from imblearn.ensemble import BalanceCascade, EasyEnsemble 
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing, metrics
from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import GridSearchCV

# based on modified version of code published at:
# blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.mets = {}

    def fit(self, X, y, cv=5, use_pca=0, k_select=0, imbal=None, verbose=2):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            
            cvk = StratifiedKFold(n_splits=cv, random_state=7)
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=.3, random_state=505)
            

            
            #scaling            
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # apply   (use only on scaled data)
            if imbal != None:
                X_train, y_train = imbal.fit_sample(X_train, y_train)            
            
            # Build estimator from PCA and Univariate selection: 
            if k_select>0 and use_pca>0:
                pca = PCA(n_components=use_pca, whiten=True)
                sk = SelectKBest(k=k_select)
                combined_features = FeatureUnion([("pca", pca), ("univ_select", sk)])
                X_train = combined_features.fit(X_train, y_train).transform(X_train) 
                X_test = combined_features.transform(X_test)
            else:
                # apply PCA  (use only on scaled data)
                if use_pca>0:
                    pca = PCA(n_components=use_pca, whiten=True).fit(X_train)
                    X_train = pca.transform(X_train)
                    X_test = pca.transform(X_test)
                    
                # apply selectKBest   (use only on scaled data)
                if k_select>0:
                    sk = SelectKBest(k=k_select).fit(X_train, y_train)
                    X_train = sk.transform(X_train)
                    X_test = sk.transform(X_test) 
                




 
            scorer = metrics.make_scorer(combinedScorer, greater_is_better=True)
            gs = GridSearchCV(model, params, cv=cvk, n_jobs=4, 
                              verbose=verbose, scoring=scorer, error_score=-1.0)
            gs.fit(X_train, y_train)
            self.grid_searches[key] = gs            
            self.mets[key] = self.log_metrics(gs, X_test, y_test)
            

    def log_metrics(self, classifier, X_test, y_test):
        """ Performs metric calculations and prints out to console."""
        metres = {}
        #res = clf.score(X_test, y_test)
        pred = classifier.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)

        metres['kappa'] = metrics.cohen_kappa_score(y_test, pred)
        metres['auc'] = metrics.auc(fpr, tpr)
        metres['acc'] = metrics.accuracy_score(y_test, pred)
        metres['model_score'] = classifier.score(X_test, y_test)
        metres['prec_c1'] = metrics.precision_score(y_test, pred)
        metres['rec_c1'] = metrics.recall_score(y_test, pred)
        metres['f1-score_c1'] = metrics.f1_score(y_test, pred)
        metres['prec_c0'] = metrics.precision_score(y_test, pred, pos_label=0)
        metres['rec_c0'] = metrics.recall_score(y_test, pred, pos_label=0)
        metres['f1-score_c0'] = metrics.f1_score(y_test, pred, pos_label=0)
        metres['confusion'] = metrics.confusion_matrix(y_test, pred)
        #metres['pr_re_f1_sp'] = metrics.precision_recall_fscore_support(y_test, pred)
        balgeomsc = iba(alpha=0.1, squared=True)(geomsc)
        
        print 'BEST:', classifier.best_estimator_
        print '\t ACC \t MCC \t KAP \t AUC \t GEOM \t IBA'
        print ("\t%.3f \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f" % (metres['acc'],
                    metres['model_score'], metres['kappa'], metres['auc'],
                    geomsc(y_test, pred), balgeomsc(y_test, pred)))
        print metres['confusion']
        print metrics.classification_report(y_test, pred)
        print ''
        return metres

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, metres):
            d = {
                    'estimator': key,
                    'acc': metres['acc'],
                    'model_score': metres['model_score'],
                    'kappa': metres['kappa'],
                    'auc': metres['auc'],
                    'prec_c1': metres['prec_c1'],
                    'rec_c1': metres['rec_c1'],
                    'f1_c1': metres['f1-score_c1'],
                    'prec_c0': metres['prec_c0'],
                    'rec_c0': metres['rec_c0'],
                    'f1_c0': metres['f1-score_c0'],
                    'conf_matrix': metres['confusion'],
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'mean_score': np.mean(scores),
                    'sd_score': np.std(scores)
            }
            return pd.Series(dict(d.items()))

        gsc = self.grid_searches
        rows = [row(k, gsc[k].cv_results_['mean_test_score'], self.mets[k]) 
                     for k in self.keys]
        
        df = pd.concat(rows, axis=1).T.sort_values(by=[sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'sd_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

def combinedScorer(y_test, pred):
    """ combine existing balanced scorers in an attempt to provide a more
    accurate scoring function in context of extremely imblanaced data."""
    mcc = metrics.matthews_corrcoef(y_test, pred)
    return mcc

  

classifiers = {
    'LogisticRegression': LogisticRegression(n_jobs=4),
    'ExtraTreesClassifier': ExtraTreesClassifier(n_jobs=4),
    'RandomForestClassifier': RandomForestClassifier(n_jobs=4),
    'AdaBoostClassifier': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
    'MLPClassifier': MLPClassifier(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(n_jobs=4),
    'VotingClassifier': VotingClassifier(estimators=[
            ('lr', LogisticRegression(n_jobs=4)),
            ('rf', RandomForestClassifier(n_jobs=4)),
            ('knn', KNeighborsClassifier(n_jobs=4))], weights=[1,1,1], n_jobs=4),
    'SGDClassifier': SGDClassifier(n_jobs=4),
    'GaussianNB': GaussianNB()
}
#
paramsA = { 
    'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 5, 10],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'intercept_scaling': [0.1, 0.01, 1, 5, 10],
            'solver': ['lbfgs', 'liblinear', 'sag']},
    'ExtraTreesClassifier': { 
            'n_estimators': [15, 30, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [8, 16, 32, None],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'max_features': ['auto', 0.75, 0.33]},
    'RandomForestClassifier': {
            'n_estimators': [15, 30, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [8, 16, 32, None],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'max_features': ['auto', 0.75, 0.33]},
    'AdaBoostClassifier':  {
            'n_estimators': [15, 50, 80],
            'base_estimator__criterion': ['gini', 'entropy'],
            'base_estimator__splitter': ['best', 'random'],
            'base_estimator__max_depth': [4, 8, 16, 32],
            'base_estimator__class_weight': [{1:5}, {1:50}, {1:500}, 'balanced'],
            'learning_rate': [0.5, 1.0, 2.0]},
    'MLPClassifier': {
            'alpha': [0.001, 0.01, 0.1, 0.5],
            'solver': ['lbfgs'], # 'sgd', 'adam'],
            'activation': ['tanh', 'relu'],
            'learning_rate': ['adaptive', 'invscaling'],
            'hidden_layer_sizes': [(240,40,), (60,20,10,), (500,)]},
    'SVC': {
            'decision_function_shape': ['ovr'],
            'C': [0.01, 0.1, 0.5, 1, 10],
            'gamma': [0.01, 0.1, 1, 10],
            'class_weight': [{1:3}, {1:50}, {1:500}, 'balanced'],
            'kernel': ['linear', 'rbf']},
    'KNeighborsClassifier': {
            'weights': ['distance', 'uniform'],
            'p': [1, 2, 3, 4],
            'metric': ['minkowski'],
            'leaf_size': [30, 5],
            'n_neighbors': [1, 2, 3, 4, 5, 6]},
    'VotingClassifier': {
            'voting': ['soft'], #, 'hard'],
            'lr__C': [0.1, 1.0, 10.0],
            'lr__class_weight': [{1:15}, {1:500}, 'balanced'],
            'rf__n_estimators': [10, 30],
            'rf__class_weight': [{1:15}, {1:500}, 'balanced'],
            'knn__weights': ['distance', 'uniform'],
            'knn__n_neighbors': [2, 4]},
    'SGDClassifier': {
            'alpha': [0.0001, 0.001, 0.1, 0.5, 1 ],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'eta0': [0.01, 0.001, 0.5],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'loss': ['modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['none', 'l1', 'l2', 'elasticnet']},
    'GaussianNB': {}
}



#   Modifications below

if __name__=="__main__":
    fileid = 2
    filename = mig.CASE_NAMES[fileid] 
    df = mig.getCaseData(filename)
    if fileid==2:
        df = df.loc[:'2002-06-24 00:00:00']  # for the sb_dataset
        
    pre = 30; post = 0
    win = 120; step = 30
    label = 'start'
    pca_target = 0 
    poly_degree = 0; k_select=0
    #imbal = SMOTETomek(smote=SMOTE(m_neighbors=100, k_neighbors=3, kind='regular'))
    imbal = None#SMOTE()#SMOTE(k_neighbors=3)#None#CondensedNearestNeighbour()
    
#    orig_stdout = sys.stdout
#    fout = open('experiments/Results_100x30_basic.txt', 'a')
#    sys.stdout = fout
   
    # not needed for pre-calculated feature runs
    df.flag = mig.labelSequence(df, pre, post)
 
 
 
 
    pca = [0]
    poly = [0]
    ksel = [0]
    imb = [ClusterCentroids(), SMOTE(), None]

	
# 	*****************************************	
	
	
    for pca_target, poly_degree, k_select, imbal in list(itertools.product(pca, poly, ksel, imb)):
    
        start_time = datetime.now()
        
        print 'pre/post: %d/%d  win/stride: %d/%d  label:%s ' % (pre,post,win,step,label)
        print 'filename: %s ' % filename #(feature set prepared earlier)
        print 'datetime: %s ' % start_time
        print 'pca_target: %d \t poly degree: %d \t kselect: %d ' % \
                        (pca_target, poly_degree, k_select)
        print 'Imbalance: %s ' % (imbal)                 
    
        
        print 'Extended target (count): ', df.flag.count(), df.flag[df.flag].count()
     
        
        # *********************  Additional filtering **************************
#        import statsmodels.api as sma
#        filt_cycle, filt_trend = sma.tsa.filters.hpfilter(df.value, lamb=1600)
#        df.value = filt_trend
        #plt.plot(filt_trend)
        
     
        
        
        # ********************** Feature calculation step: ********************
        
        # calculate tsfresh feature set, passing window and step size
        #X, y = css.makeTSFeatureSet(df, win, step, label_by=label)
        
        # load preprocessed TS features (UT=unTruncated)
#        X = pd.read_csv("features/X_y_feats_ts200x25_pp25-0_cid2_filter.csv", sep=';')
#        y = X.iloc[:,-1]
#        X = X.as_matrix(columns=X.columns[:-1])
#        
      
        # get basic stats as features
        X, y = css.makeBasicFeatureSet(df, win, step, label_by=label)


        # choose the features we want lagged and create additional features
        #Xlagged = css.makeLaggedFeatureSet(X, lags=1)
        Xlagged = css.makeLaggedFeatureSet(X[:,[3,4]], lags=2)
        X = np.append(X, Xlagged, axis=1)
        

        
    #    # append special features to main matrix
        Xs, _ = css.makeSpecialFeatureSet(df, win, step, label_by=label)       
        X = np.append(X, Xs, axis=1)
        
        # polynomial features : careful here 120min on MLP with p=3, basic
        if poly_degree > 1:
            poly = preprocessing.PolynomialFeatures(poly_degree)
            X = poly.fit_transform(X)
    

        print('Total : Processed (count): ', X.shape, y[y>0].count())

    

    
        print 'Final feature (count): ', X.shape  
    
        #**********************************************************************
    
        experiment = EstimatorSelectionHelper(classifiers, paramsA)
        experiment.fit(X, y, cv=5, use_pca=pca_target, k_select=k_select, imbal=imbal, verbose=1)
    
        report = experiment.score_summary(sort_by='auc')
        print report
        print 'Elapsed time %.2f mins \n' % ((datetime.now() - start_time).total_seconds()/60.0)
        print '************************************************************\n'
 
    print 'Finished.'
    
    
 
