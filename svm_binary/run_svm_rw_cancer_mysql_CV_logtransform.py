'''

Optimizes and evaluates SVM classification model with standard split

'''
from modules import *
from sklearn.model_selection import train_test_split
import hickle as hkl
import numpy as np
###RW only
for alfa in [ 0, 0.01]:

    if alfa == 0:
        profiles_path = '../dataset/data_hyperfoods_drugcentral/drugs_profiles_approved_mysql_on_onecc_noiso_string_dense.hkl'
    else:
        profiles_path = '../dataset/data_hyperfoods_drugcentral/drugs_profiles_approved_mysql_on_onecc_noiso_string_dense_{}.hkl'.format(alfa)



    labels_path =  '../dataset/data_hyperfoods_drugcentral/drugs_labels_approved_mysql_on_onecc_noiso_string_dense.hkl'

    profiles, labels = load_data(profiles_path, labels_path)

    outdir = 'cancer_5foldCV_logtransform/rw_split_{0}/'.format(alfa)


    dict_metrics = {}
    dict_metrics['balanced_acc'] = []
    dict_metrics['AUCROC'] = []
    dict_metrics['AUPR']= []
    dict_metrics['F1'] = []
    dict_metrics['SKF1'] = []

    #Splits
    for i in range(5):
        outdir_i = outdir + '{}/'.format(i)
        train_indices = hkl.load(
            '../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}/split_cancer_train_indices_approved_mysql_on_onecc_noiso_string_dense.hkl'.format(i))
        test_indices = hkl.load(
            '../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}/split_cancer_test_indices_approved_mysql_on_onecc_noiso_string_dense.hkl'.format(i))
        val_indices = hkl.load(
            '../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}/split_cancer_val_indices_approved_mysql_on_onecc_noiso_string_dense.hkl'.format(i))



        training_X = profiles[np.hstack([train_indices, val_indices])]
        median_X = np.median(training_X)
        if median_X < 1.0:
            median_X = 1.0

        training_X = np.log(training_X + median_X)
        test_X = np.log(profiles[test_indices] + median_X)

        training_y = np.array(labels)[np.hstack([train_indices, val_indices])]
        test_y = np.array(labels)[test_indices]

        #Use CV in train+val to find best hp
        grid, outdir_ii = optimize_svm(training_X, training_y, nfolds = 5, outdir = outdir_i, njobs = 1)

        #Evaluate on test set with model obtained from CV
        # evaluate_svm_training(grid, training_X, training_y, outdir=outdir)

        perf = evaluate_svm(grid, test_X, test_y, outdir = outdir_ii )
        dict_metrics['balanced_acc'].append(perf['Accuracy'])
        dict_metrics['AUCROC'].append(perf['AUCROC'])
        dict_metrics['AUPR'].append(perf['AUPR'])
        dict_metrics['F1'].append(perf['F1'])
        dict_metrics['SKF1'].append(perf['SKF1'])

    balanced_acc = np.array(dict_metrics['balanced_acc'])
    aucroc = np.array(dict_metrics['AUCROC'])
    aupr = np.array(dict_metrics['AUPR'])
    f1 = np.array(dict_metrics['F1'])
    skf1 = np.array(dict_metrics['SKF1'])

    #Log
    log_handle = open(outdir + 'summary_metrics_CV.txt', 'w')
    summ = ' SKF1\tstd\tF1\tstd\tAUCROC\tstd\tAUPR\tstd\tBalanced accuracy\tstd\n' \
           '{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(np.mean(skf1),
                                                                                   np.std(skf1),
                                                                                    np.mean(f1),
                                                                                   np.std(f1),
                                                                                   np.mean(aucroc),
                                                                                   np.std(aucroc),
                                                                                   np.mean(aupr),
                                                                                   np.std(aupr),
                                                                                   np.mean(balanced_acc),
                                                                                   np.std(balanced_acc))
    log_handle.write(summ)
    log_handle.close()
