'''

Optimizes and evaluates SVM classification model with standard split

'''
from modules import *
from sklearn.model_selection import train_test_split
import hickle as hkl
import numpy as np
###RW only
for alfa in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    profiles_path = '../dataset/data_hyperfoods_drugcentral/drugs_profiles_approved_mysql_on_onecc_noiso_string_dense_{}.hkl'.format(alfa)
    labels_path =  '../dataset/data_hyperfoods_drugcentral/drugs_labels_approved_mysql_on_onecc_noiso_string_dense.hkl'

    profiles, labels = load_data(profiles_path, labels_path)

    outdir = 'cancer/rw_standard_split_{0}/'.format(alfa)



    #Standard
    train_indices = hkl.load('../dataset/data_hyperfoods_drugcentral/split_cancer_train_indices_approved_mysql_on_onecc_noiso_string_dense.hkl')
    test_indices = hkl.load('../dataset/data_hyperfoods_drugcentral/split_cancer_test_indices_approved_mysql_on_onecc_noiso_string_dense.hkl')
    val_indices = hkl.load('../dataset/data_hyperfoods_drugcentral/split_cancer_val_indices_approved_mysql_on_onecc_noiso_string_dense.hkl')

    training_X = profiles[np.hstack([train_indices, val_indices])]
    test_X = profiles[test_indices]
    training_y = np.array(labels)[np.hstack([train_indices, val_indices])]
    test_y = np.array(labels)[test_indices]


    grid, outdir = optimize_svm(training_X, training_y, nfolds = 10, outdir = outdir, njobs = 1)
    evaluate_svm_training(grid, training_X, training_y, outdir=outdir)
    evaluate_svm(grid, test_X, test_y, outdir = outdir )

