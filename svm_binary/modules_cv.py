'''
File with modules to optimize classifier and evaluate predictions

@author: ggonzalezp
'''


import hickle as hkl
import pandas as pd
import numpy as np
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, roc_auc_score, auc


import matplotlib.pyplot as plt
from inspect import signature

import os
from custom_metrics import harmonic_f1
from sklearn.decomposition import PCA

def load_data(drug_profiles_path, labels_path):
    drug_profiles = hkl.load(drug_profiles_path)
    labels = hkl.load(labels_path)

    return drug_profiles, labels




def optimize_svm(training_X, training_y, nfolds, outdir, njobs):

    #Creates outdir
    busy = True

    while busy:
        if os.path.isdir(outdir):
            outdir += '_'
        else:
            outdir = outdir +'/'
            os.makedirs(outdir)
            busy = False

    pipe = [('preprocessing', StandardScaler())]
    pipe.append((('compression', PCA())))
    pipe.append(('classifier', svm.LinearSVC(class_weight = 'balanced')))

    parameters = {}
    parameters['classifier__C'] = [10 ** -6,
                                              10 ** -5,
                                              10 ** -4,
                                              10 ** -3,
                                              10 ** -2,
                                              10 ** -1,
                                              10 ** 0
                                              ]

    param_grid = parameters

    class PipelineClassifier(Pipeline):
        _estimator_type = "classifier"

    pipe = PipelineClassifier(pipe)

    best_params = {}
    scores = {}

    print('X_train: %s' % str(training_X.shape))
    print('y_train: %s' % str(training_y.shape))


    grid = GridSearchCV(pipe, param_grid, cv=nfolds, verbose=2, refit='balance', n_jobs=njobs)
    grid.fit(training_X, training_y)

    #Log
    msg = "Best params:\n{}\n".format(grid.best_params_)

    print(msg)
    log_handle = open(outdir + 'params.txt', 'w')
    log_handle.write(msg)



    return grid, outdir


def evaluate_svm(grid, test_X, test_y):




    #Prediction
    y_test_pred = grid.predict(test_X)
    y_test_score = grid.decision_function(test_X)

    print('X_test: %s' % str(test_X.shape))
    print('y_test: %s' % str(test_y.shape))

    #Metrics
        #Accuracy
    acc = accuracy_score(test_y, y_test_pred)

        #AUC ROC
    roc = roc_auc_score(test_y, y_test_score)

        #Precision, Recall
    average_precision = average_precision_score(test_y, y_test_score)
    precision, recall, _ = precision_recall_curve(test_y, y_test_score)
    aupr = auc(recall, precision)

        #F1
    f1_score = harmonic_f1(test_y, y_test_pred)


    #Log
    log = {}
    log['Accuracy'] = acc
    log['AUC ROC'] = roc
    log['AUPR'] = aupr
    log['F1'] = f1_score


    return log











