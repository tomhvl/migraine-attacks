# -*- coding: utf-8 -*-
"""
Constants and global settings

Created on Sat Apr 08 10:39:01 2017

@author: Tomba
"""

#FOLDER_NAME = "..\..\PREPPED_DATAPREPPED_DATA"
CASE_NAMES = ["stm_aasane16_modified.csv", "stm_aasane18_modified.csv", \
              "stm_sb_modified.csv", "stm_aasane01_03_modified.csv",
              "stm_kontroll02 (edited).csv", "stm_fusa04 (edited).csv"]

CLASSIFIERS = {
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

PARAMETERS = { 
    'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 5, 10],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'intercept_scaling': [0.1, 0.01, 1, 5, 10],
            'solver': ['lbfgs', 'liblinear', 'sag']},
    'ExtraTreesClassifier': { 
            'n_estimators': [15, 30, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [8, 16, 32, 64],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'max_features': ['auto', 0.75, 0.33]},
    'RandomForestClassifier': {
            'n_estimators': [15, 30, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [8, 16, 32, 64],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'max_features': ['auto', 0.75, 0.33]},
    'AdaBoostClassifier':  {
            'n_estimators': [15, 30, 50],
            'base_estimator__criterion': ['gini', 'entropy'],
            'base_estimator__splitter': ['best', 'random'],
            'base_estimator__max_depth': [4, 8, 16, 32],
            'base_estimator__class_weight': [{1:5}, {1:50}, {1:500}, 'balanced'],
            'learning_rate': [0.5, 1.5, 1.0, 2.0]},
    'MLPClassifier': {
            'alpha': [0.001, 0.01, 0.1, 0.5],
            'solver': ['lbfgs'], # 'sgd', 'adam'],
            'activation': ['tanh', 'relu'],
            'learning_rate': ['adaptive', 'invscaling'],
            'hidden_layer_sizes': [(240,40,), (1000,50,)]},
    'SVC': {
            'decision_function_shape': ['ovr'],
            'C': [0.1, 0.5, 1, 10, 25],
            'gamma': [0.01, 0.1, 1, 10],
            'class_weight': [{1:3}, {1:15}, {1:50}, {1:500}, 'balanced'],
            'kernel': ['linear', 'rbf']},
    'KNeighborsClassifier': {
            'weights': ['distance', 'uniform'],
            'p': [2, 3, 4, 5],
            'leaf_size': [30],
            'n_neighbors': [1, 2, 3, 4, 5, 6]},
    'VotingClassifier': {
            'voting': ['soft', 'hard'],
            'lr__C': [0.1, 1.0, 10.0],
            'lr__class_weight': [{1:15}, {1:500}, 'balanced'],
            'rf__n_estimators': [15, 50],
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


#CLASSIFIERS = {
#        'LogisticRegression': LogisticRegression(n_jobs=4),
#        'GaussianNB': GaussianNB(),
#        'KNeighborsClassifier': KNeighborsClassifier(n_jobs=4),
#        'VotingClassifier': VotingClassifier(estimators=[
#            ('lr', LogisticRegression()),
#            ('rf', RandomForestClassifier()),
#            ('knn', KNeighborsClassifier())], weights=[1,1,1], n_jobs=4)
#}
#
#PARAMETERS = { 
#    'LogisticRegression': {
#            'C': [0.001, 0.01, 0.1, 1, 5, 10],
#            'class_weight': [{1:3}, {1:10}, {1:30}, {1:90}, 'balanced'],
#            'intercept_scaling': [0.1, 0.01, 1, 5],
#            'solver': ['lbfgs', 'liblinear', 'sag']},
#    'GaussianNB': {} 
#}
  

