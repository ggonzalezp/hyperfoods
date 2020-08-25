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

from sklearn.metrics import precision_recall_curve, average_precision_score, balanced_accuracy_score, roc_auc_score, auc, roc_curve


import matplotlib.pyplot as plt
from inspect import signature
import os
from custom_metrics import harmonic_f1
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score as f1_score_s

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
    # pipe = [('compression', PCA())]

    pipe.append((('compression', PCA())))
    pipe.append(('classifier', svm.LinearSVC(class_weight = 'balanced')))
    # pipe = [('classifier', svm.LinearSVC(class_weight = 'balanced'))]

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
    log_handle.write('\n\n')
    log_handle.write(str(grid.best_estimator_))



    return grid, outdir

def optimize_svm_radial(training_X, training_y, nfolds, outdir, njobs):

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
    pipe.append(('classifier', svm.SVC(class_weight = 'balanced', probability=True)))

    parameters = {}
    n_features = float(min(training_X.shape[1], training_X.shape[0]))
    gamma_base = 1.0/n_features
    gamma_startlog = np.log(gamma_base)/np.log(3.0)
    parameters['classifier__C'] = [10 ** -2,
                                   10 ** -1,
                                   10 ** 0,
                                   10 ** 1,
                                   10 ** 2,
                                   ]

    parameters['classifier__gamma'] = [
                           3 ** (gamma_startlog -6),
                           3 ** (gamma_startlog -5),
                           3 ** (gamma_startlog -4),
                           3 ** (gamma_startlog -3),
                           3 ** (gamma_startlog -2),
                           3 ** (gamma_startlog -1),
                           3 ** (gamma_startlog),
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
    log_handle.write('\n\n')
    log_handle.write(str(grid.best_estimator_))



    return grid, outdir


def evaluate_svm_training(grid, test_X, test_y, outdir):




    #Prediction
    y_test_pred = grid.predict(test_X)
    y_test_score = grid.decision_function(test_X)

    print('X_train: %s' % str(test_X.shape))
    print('y_train: %s' % str(test_y.shape))

    #Metrics
        #Accuracy
    acc = balanced_accuracy_score(test_y, y_test_pred)

        #AUC ROC
    roc = roc_auc_score(test_y, y_test_score)

        #Precision, Recall
    average_precision = average_precision_score(test_y, y_test_score)
    precision, recall, _ = precision_recall_curve(test_y, y_test_score)
    aupr = auc(recall, precision)

        #F1
    f1_score = harmonic_f1(test_y, y_test_pred)


    #F1 sklearn
    f1_sklearn = f1_score_s(test_y, y_test_pred)



    #Log
    log_handle = open(outdir + 'log_training', 'w')
    log_handle.write('Balanced accuracy: \t {0:0.4f} \n'.format(acc))
    log_handle.write('AUC ROC: \t {0:0.4f} \n'.format(roc))
    log_handle.write('AUPR: \t {0:0.4f} \n'.format(aupr))
    log_handle.write('F1: \t {0:0.4f} \n'.format(f1_score))
    log_handle.write('Sklearn - F1: \t {0:0.4f} \n'.format(f1_sklearn))



    plt.figure()

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig(outdir+'AUPR.png')


def evaluate_svm(grid, test_X, test_y, outdir):




    #Prediction
    y_test_pred = grid.predict(test_X)
    y_test_score = grid.decision_function(test_X)
    pd.DataFrame(y_test_pred).to_csv(outdir + 'test_pred.csv', index=False, header=None)
    pd.DataFrame(y_test_score).to_csv(outdir + 'test_prob.csv', index=False, header=None)
    pd.DataFrame(test_y).to_csv(outdir +'test_real.csv', header=None, index=False)



    print('X_test: %s' % str(test_X.shape))
    print('y_test: %s' % str(test_y.shape))

    #Metrics
        #Accuracy
    acc = balanced_accuracy_score(test_y, y_test_pred)

        #AUC ROC
    roc = roc_auc_score(test_y, y_test_score)

        #Precision, Recall
    average_precision = average_precision_score(test_y, y_test_score)
    precision, recall, _ = precision_recall_curve(test_y, y_test_score)
    aupr = auc(recall, precision)

        #F1
    f1_score = harmonic_f1(test_y, y_test_pred)
    # SKLEARN F1

    f1_sklearn = f1_score_s(test_y, y_test_pred)



    #Log
    log_handle = open(outdir + 'log', 'w')
    log_handle.write('Balanced accuracy: \t {0:0.4f} \n'.format(acc))
    log_handle.write('AUC ROC: \t {0:0.4f} \n'.format(roc))
    log_handle.write('AUPR: \t {0:0.4f} \n'.format(aupr))
    log_handle.write('F1: \t {0:0.4f} \n'.format(f1_score))
    log_handle.write('Sklearn - F1: \t {0:0.4f} \n'.format(f1_sklearn))



    plt.figure()

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig(outdir+'AUPR.png')



    #ROC AUC curve
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(test_y, y_test_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='pink', lw = lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(outdir + 'AUCROC.png')

    log = {}
    log['Accuracy'] = acc
    log['AUCROC'] = roc
    log['AUPR'] = aupr
    log['F1'] = f1_score
    log['SKF1'] = f1_sklearn

    return log





def optimize_mlp(training_X, training_y, nfolds, outdir, njobs):

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
    # pipe = [('compression', PCA())]

    pipe.append((('compression', PCA())))
    pipe.append(('classifier', MLPClassifier(hidden_layer_sizes=(32, 2), early_stopping=True, max_iter = 500)))
    # pipe = [('classifier', svm.LinearSVC(class_weight = 'balanced'))]

    parameters = {}
    # parameters['classifier__C'] = [10 ** -6,
    #                                           10 ** -5,
    #                                           10 ** -4,
    #                                           10 ** -3,
    #                                           10 ** -2,
    #                                           10 ** -1,
    #                                           10 ** 0
    #                                           ]

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
    log_handle.write('\n\n')
    log_handle.write(str(grid.best_estimator_))



    return grid, outdir

def evaluate_mlp(grid, test_X, test_y, outdir):




    #Prediction
    y_test_pred = grid.predict(test_X)
    y_test_score = grid.predict_proba(test_X)[:,1]

    print('X_test: %s' % str(test_X.shape))
    print('y_test: %s' % str(test_y.shape))

    #Metrics
        #Accuracy
    acc = balanced_accuracy_score(test_y, y_test_pred)

        #AUC ROC
    roc = roc_auc_score(test_y, y_test_score)

        #Precision, Recall
    average_precision = average_precision_score(test_y, y_test_score)
    precision, recall, _ = precision_recall_curve(test_y, y_test_score)
    aupr = auc(recall, precision)

        #F1
    f1_score = harmonic_f1(test_y, y_test_pred)

        #SKLEARN F1
    f1_sklearn = f1_score_s(test_y, y_test_pred)


    #Log
    log_handle = open(outdir + 'log', 'w')
    log_handle.write('Balanced accuracy: \t {0:0.4f} \n'.format(acc))
    log_handle.write('AUC ROC: \t {0:0.4f} \n'.format(roc))
    log_handle.write('AUPR: \t {0:0.4f} \n'.format(aupr))
    log_handle.write('F1: \t {0:0.4f} \n'.format(f1_score))
    log_handle.write('Sklearn - F1: \t {0:0.4f} \n'.format(f1_sklearn))




    plt.figure()

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig(outdir+'AUPR.png')
def evaluate_mlp_training(grid, test_X, test_y, outdir):




    #Prediction
    y_test_pred = grid.predict(test_X)
    y_test_score = grid.predict_proba(test_X)[:,1]

    print('X_train: %s' % str(test_X.shape))
    print('y_train: %s' % str(test_y.shape))

    #Metrics
        #Accuracy
    acc = balanced_accuracy_score(test_y, y_test_pred)

        #AUC ROC
    roc = roc_auc_score(test_y, y_test_score)

        #Precision, Recall
    average_precision = average_precision_score(test_y, y_test_score)
    precision, recall, _ = precision_recall_curve(test_y, y_test_score)
    aupr = auc(recall, precision)

        #F1
    f1_score = harmonic_f1(test_y, y_test_pred)


        #SKLEARN F1
    f1_sklearn = f1_score_s(test_y, y_test_pred)


    #Log
    log_handle = open(outdir + 'log_training', 'w')
    log_handle.write('Balanced accuracy: \t {0:0.4f} \n'.format(acc))
    log_handle.write('AUC ROC: \t {0:0.4f} \n'.format(roc))
    log_handle.write('AUPR: \t {0:0.4f} \n'.format(aupr))
    log_handle.write('F1: \t {0:0.4f} \n'.format(f1_score))
    log_handle.write('Sklearn - F1: \t {0:0.4f} \n'.format(f1_sklearn))



    plt.figure()

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig(outdir+'AUPR.png')


