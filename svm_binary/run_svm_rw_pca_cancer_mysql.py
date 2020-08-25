'''

Optimizes and evaluates SVM classification model with embeddings from different architectures

'''
from modules import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hickle as hkl

###RW only
profiles = []
outdir = 'rw_pca_ac_standard_split/'
for alfa in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    profiles_path = '../final_dataset/drugs_profiles_approved_mysql_on_onecc_noiso_string_dense_{}.hkl'.format(alfa)
    labels_path =  '../final_dataset/drugs_labels_approved_mysql_on_onecc_noiso_string_dense.hkl'

    profiles_tmp, labels = load_data(profiles_path, labels_path)
    profiles.append(profiles_tmp)



profiles = np.array(profiles)
profiles = np.swapaxes(profiles, 0,1) #nsamples x 9 x ngenes


    #Standard
train_indices = hkl.load('../final_dataset/split_cancer_train_indices_approved_mysql_on_onecc_noiso_string_dense.hkl')
test_indices = hkl.load('../final_dataset/split_cancer_test_indices_approved__mysql_on_onecc_noiso_string_dense.hkl')
val_indices = hkl.load('../final_dataset/split_cancer_val_indices_approved_mysql_on_onecc_noiso_string_dense.hkl')

training_X = profiles[np.hstack([train_indices, val_indices])]
test_X = profiles[test_indices]
training_y = np.array(labels)[np.hstack([train_indices, val_indices])]
test_y = np.array(labels)[test_indices]


    #PCA along the 2nd dimension to keep only 1 (1 PCA for each gene)
pca_training_X = []
pca_test_X = []
print('Computing PCA for dimensionality reduction along genes dimension')
for gene_idx in range(training_X.shape[2]):
    matrix_tmp_training = training_X[:,:, gene_idx]
    matrix_tmp_test = test_X[:, :, gene_idx]

    #Standard scaler
    scaler = StandardScaler()
    scaler.fit(matrix_tmp_training)
    matrix_tmp_training = scaler.transform(matrix_tmp_training)
    matrix_tmp_test = scaler.transform(matrix_tmp_test)

    #PCA
    pca = PCA(n_components=1)
    pca.fit(matrix_tmp_training)

    pca_training_X.append(pca.transform(matrix_tmp_training))
    pca_test_X.append(pca.transform(matrix_tmp_test))


pca_training_X = np.swapaxes(np.squeeze(np.array(pca_training_X)), 0,1)
pca_test_X = np.swapaxes(np.squeeze(np.array(pca_test_X)), 0,1)



grid, outdir = optimize_svm(pca_training_X, training_y, nfolds = 10, outdir = outdir, njobs = 25)

evaluate_svm_training(grid, training_X, training_y, outdir=outdir)
evaluate_svm(grid, pca_test_X, test_y, outdir = outdir )

