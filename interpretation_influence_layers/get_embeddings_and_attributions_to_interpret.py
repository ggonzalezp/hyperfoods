####Computes embedding of drugs of interest (Cisplatin) for interpretation
import torch
import sys
sys.path.append('../')
from models.networks_gcn import GCNModel, GCNModelWPathways, ChebModel, ChebModelWPathways, SageModel, SageModelWPathways
import os
from glob import glob
import os.path as osp
import hickle as hkl
from utils import utils, train_eval, DataLoader
import numpy as np
import pandas as pd
from utils_interpretation import preparing_data
from captum.attr import (
    IntegratedGradients,
    LayerConductance)
from sklearn.preprocessing import MinMaxScaler


def get_embeddings_to_interpret(drug_indices_of_interest, dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops):

    model, samples_of_interest, loader, feature_names, device = preparing_data(drug_indices_of_interest, dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)

    scaler = MinMaxScaler()

    #######Computing embeddings
    model.eval()
    list_embeddings = []
    with torch.no_grad():
        for data in loader:
            # data.x is transposed because ig captum_ requires data input nsamples x nfeatures
            # data.edge_index is reshaped because ig captum_ requires this to be nsamples x nfeatures
            x, batch, y = data.x.to(device), \
                          data.batch.to(device), data.y.to(device)
            embedding = model.graph_embedding(x, batch)
            list_embeddings.append(scaler.fit_transform(embedding.detach().cpu().numpy().reshape(-1,1)).reshape(1,-1))


    embeddings = pd.DataFrame(np.concatenate(list_embeddings))
    embeddings.index = drug_indices_of_interest
    embeddings.columns = feature_names

    embeddings.to_csv(outdir+'/drug_embeddings.txt', sep = '\t')

    ##For GSEA - format: transpose with column headers: 'LABEL', 'WEIGHT'
    embeddings = embeddings.transpose()
    embeddings_gsea = pd.concat([pd.DataFrame(embeddings.index), pd.DataFrame(embeddings.values)], 1)
    embeddings_gsea.columns = ['LABEL', 'WEIGHT']                       #Name columns
    embeddings_gsea = embeddings_gsea[embeddings_gsea['LABEL'] != '']   #Remove genes without name

    embeddings_gsea.to_csv(outdir+'/drug_embeddings_gsea.rnk', sep = '\t', index = False)

    return



def get_attributions_to_interpret(drug_indices_of_interest, dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops, n_steps = 100):
    model, samples_of_interest, loader, feature_names, device = preparing_data(drug_indices_of_interest, dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)

    #######Computing attributions
    # print('OUTDIR', outdir)
    scaler = MinMaxScaler()
    model.eval()
    ig = IntegratedGradients(model)

    attributions_list = []

    for sample in samples_of_interest:
        sample = sample.to(device)
        baseline = torch.tensor(0 * sample.x, requires_grad=True)
        sample.x = torch.tensor(sample.x, requires_grad=True)

        attributions, delta = ig.attribute(sample.x.view(1, -1, 1),
                                           baselines=baseline.view(1, -1, 1),
                                           additional_forward_args=(torch.tensor([0], device=device)),
                                           target=1, return_convergence_delta=True,
                                           internal_batch_size=1,
                                           n_steps=n_steps)

        attributions_list.append(attributions.detach().cpu().numpy())
        # print('IG Attributions:', attributions)
        log_handle = open(outdir+'/log_get_attributions_to_interpret.txt', 'w')
        scores = model(baseline.view(1, -1, 1), torch.tensor([0], device=device)).detach().cpu().numpy()[0]
        log_handle.write('n_steps\t{}\n'.format(n_steps))
        log_handle.write('Baseline score\tDelta\t:{}\t{}\t{}\n'.format(scores[0], scores[1],delta.item()))
    log_handle.close()
    # Creates output dataframe
    attributions_list = scaler.fit_transform(np.array(attributions_list).squeeze().reshape(-1,1)).reshape(1,-1)
    attributions = pd.DataFrame(attributions_list)
    attributions.index = drug_indices_of_interest
    attributions.columns = feature_names

    attributions.to_csv(outdir+'/input_features_attribution.txt', sep = '\t')

    ##For GSEA - format: transpose with column headers: 'LABEL', 'WEIGHT'
    attributions = attributions.transpose()
    attributions_gsea = pd.concat([pd.DataFrame(attributions.index), pd.DataFrame(attributions.values)], 1)
    attributions_gsea.columns = ['LABEL', 'WEIGHT']                       #Name columns
    attributions_gsea = attributions_gsea[attributions_gsea['LABEL'] != '']   #Remove genes without name

    attributions_gsea.to_csv(outdir+'/input_features_attribution_gsea.rnk', sep = '\t', index = False)

    return


def get_attributions_to_interpret_lc(drug_indices_of_interest, dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops, n_steps = 100):
    model, samples_of_interest, loader, feature_names, device = preparing_data(drug_indices_of_interest, dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)

    #######Computing attributions
    # print('OUTDIR', outdir)

    scaler = MinMaxScaler()

    model.eval()
    layer = dict([module for module in model.named_modules() ])['fc']   #FC(1) -- will attribute to its outpu
    lc = LayerConductance(model, layer)

    attributions_list = []

    for sample in samples_of_interest:
        sample = sample.to(device)
        baseline = torch.tensor(0 * sample.x, requires_grad=True)
        sample.x = torch.tensor(sample.x, requires_grad=True)

        attributions, delta = lc.attribute(sample.x.view(1, -1, 1),
                                           baselines=baseline.view(1, -1, 1),
                                           additional_forward_args=(torch.tensor([0], device=device)),
                                           target=1, return_convergence_delta=True,
                                           internal_batch_size=1,
                                           n_steps=n_steps)

        attributions_list.append(attributions.detach().cpu().numpy())
        # print('IG Attributions:', attributions)
        log_handle = open(outdir+'/log_get_attributions_to_interpret_lc.txt', 'w')
        scores = model(baseline.view(1, -1, 1), torch.tensor([0], device=device)).detach().cpu().numpy()[0]
        log_handle.write('n_steps\t{}\n'.format(n_steps))
        log_handle.write('Baseline score\tDelta\t:{}\t{}\t{}\n'.format(scores[0], scores[1],delta.item()))
    log_handle.close()

    # Creates output dataframe
    attributions_list = scaler.fit_transform(np.array(attributions_list).squeeze().reshape(-1,1)).reshape(1,-1)
    attributions = pd.DataFrame(attributions_list)
    attributions.index = drug_indices_of_interest
    attributions.columns = feature_names

    attributions.to_csv(outdir+'/embeddings_attribution.txt', sep = '\t')

    ##For GSEA - format: transpose with column headers: 'LABEL', 'WEIGHT'
    attributions = attributions.transpose()
    attributions_gsea = pd.concat([pd.DataFrame(attributions.index), pd.DataFrame(attributions.values)], 1)
    attributions_gsea.columns = ['LABEL', 'WEIGHT']                       #Name columns
    attributions_gsea = attributions_gsea[attributions_gsea['LABEL'] != '']   #Remove genes without name

    attributions_gsea.to_csv(outdir+'/embeddings_attribution_gsea.rnk', sep = '\t', index = False)

    return























# #
# #
# #
# # #################   GCN Models  ################################################
# gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
# #Parameters
# drug_indices_of_interest = [344]
# dataset_dir = './../dataset'
# hidden_gcn = 8
# hidden_fc = 32
# split = 0
# splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
# in_channels = 1
# n_classes = 2
# feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
# model_type = 'gcn'


# #RAW
# norm = False
# for num_layers in [1,2,3,4,5,6]:


#     outdir = './out/gcn/layers_{}/raw/split_{}'.format(num_layers, split)

#     ##find the best model for the split from the log
#     model_path = './../out/gcnmodel/final_CV/layers_{}/raw'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     # print(best_model_info)
#     best_model_path = './../out/gcnmodel/final_CV/layers_{}/raw/{}/{}'.format(num_layers, split, best_model_info)

#     do_layers = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'

#     get_embeddings_to_interpret(drug_indices_of_interest,
#                                         dataset_dir,
#                                         outdir,
#                                         best_model_path,
#                                         norm,
#                                         do_layers,
#                                         batchnorm,
#                                         num_layers,
#                                         hidden_gcn,
#                                         hidden_fc,
#                                         splits_dir,
#                                         in_channels,
#                                         n_classes,
#                                         feature_names_path,
#                                         model_type,
#                                         k_hops = None)


#     get_attributions_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)

#     get_attributions_to_interpret_lc(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)




# #NORM
# norm = True
# for num_layers in [1,2,3,4,5,6]:


#     outdir = './out/gcn/layers_{}/norm/split_{}'.format(num_layers, split)

#     ##find the best model for the split from the log
#     model_path = './../out/gcnmodel/final_CV/layers_{}/norm'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     best_model_path = model_path + '/{}/{}'.format(split, best_model_info)

#     do_layers = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'

#     get_embeddings_to_interpret(drug_indices_of_interest,
#                                         dataset_dir,
#                                         outdir,
#                                         best_model_path,
#                                         norm,
#                                         do_layers,
#                                         batchnorm,
#                                         num_layers,
#                                         hidden_gcn,
#                                         hidden_fc,
#                                         splits_dir,
#                                         in_channels,
#                                         n_classes,
#                                         feature_names_path,
#                                         model_type,
#                                         k_hops = None)


#     get_attributions_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)

#     get_attributions_to_interpret_lc(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)





#
# #################   GCNWPathways Models  ################################################
# #Parameters
# drug_indices_of_interest = [344]
# dataset_dir = './../dataset'
# hidden_gcn = 8
# hidden_fc = 32
# split = 0
# splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
# in_channels = 1
# n_classes = 2
# feature_names_path = './../dataset/pathways_names_onecc_noiso.hkl'
# model_type = 'gcnwpathways'
#
#
# #RAW
# norm = False
# for num_layers in [6]:
#
#
#     outdir = './out/gcnwpathways/layers_{}/raw/split_{}'.format(num_layers, split)
#
#     ##find the best model for the split from the log
#     model_path = './../out/gcnmodelwpathways/final_CV/layers_{}/raw'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
#
#     do_layers = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'
#
#
#     get_embeddings_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     feature_names_path,
#                                     model_type,
#                                     k_hops = None)
#     get_attributions_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)
#
#     get_attributions_to_interpret_lc(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)
#
# #NORM
# norm = True
# for num_layers in [6]:
#
#
#     outdir = './out/gcnwpathways/layers_{}/norm/split_{}'.format(num_layers, split)
#
#     ##find the best model for the split from the log
#     model_path = './../out/gcnmodelwpathways/final_CV/layers_{}/norm'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
#
#     do_layers = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'
#
#
#     get_embeddings_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     feature_names_path,
#                                     model_type,
#                                     k_hops = None)
#
#     get_attributions_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)
#
#     get_attributions_to_interpret_lc(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops = None)
#
#
#
#
# #################   Cheb Models  ################################################
# #Parameters
# drug_indices_of_interest = [344]
# dataset_dir = './../dataset'
# hidden_gcn = 8
# hidden_fc = 32
# split = 0
# splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
# in_channels = 1
# n_classes = 2
# feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
# model_type = 'cheb'


# #RAW
# norm = False
# for num_layers in [1,2,3]:


#     outdir = './out/cheb/layers_{}/raw/split_{}'.format(num_layers, split)

#     ##find the best model for the split from the log
#     model_path = './../out/chebmodel/final_CV/layers_{}/raw'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     # print(best_model_info)
#     best_model_path = model_path+ '/{}/{}'.format(split, best_model_info)

#     do_layers = int(best_model_info.split('_')[-5])
#     k_hops = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'


#     get_embeddings_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     feature_names_path,
#                                     model_type,
#                                     k_hops)

#     get_attributions_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops)

#     get_attributions_to_interpret_lc(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops)
# #NORM
# norm = True
# for num_layers in [1,2,3]:


#     outdir = './out/cheb/layers_{}/norm/split_{}'.format(num_layers, split)

#     ##find the best model for the split from the log
#     model_path = './../out/chebmodel/final_CV/layers_{}/norm'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     best_model_path = model_path + '/{}/{}'.format(split, best_model_info)

#     do_layers = int(best_model_info.split('_')[-5])
#     k_hops = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'


#     get_embeddings_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     feature_names_path,
#                                     model_type,
#                                     k_hops)
#     get_attributions_to_interpret(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops)

#     get_attributions_to_interpret_lc(drug_indices_of_interest,
#                                     dataset_dir,
#                                     outdir,
#                                     best_model_path,
#                                     norm,
#                                     do_layers,
#                                     batchnorm,
#                                     num_layers,
#                                     hidden_gcn,
#                                     hidden_fc,
#                                     splits_dir,
#                                     in_channels,
#                                     n_classes,
#                                     gene_names_path,
#                                     model_type,
#                                     k_hops)




#
# #################   Cheb Models with pathways ################################################
# #Parameters
# drug_indices_of_interest = [344]
# dataset_dir = './../dataset'
# hidden_gcn = 8
# hidden_fc = 32
# split = 0
# splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
# in_channels = 1
# n_classes = 2
# feature_names_path = './../dataset/pathways_names_onecc_noiso.hkl'
# model_type = 'chebwpathways'
#
#
# #RAW
# norm = False
# for num_layers in [1,2,3]:
#
#
#     outdir = './out/chebwpathways/layers_{}/raw/split_{}'.format(num_layers, split)
#
#     ##find the best model for the split from the log
#     model_path = './../out/chebmodelwpathways/final_CV/layers_{}/raw'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     # print(best_model_info)
#     best_model_path = model_path+ '/{}/{}'.format(split, best_model_info)
#
#     do_layers = int(best_model_info.split('_')[-5])
#     k_hops = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'
#     # print('BATCHNORM ', batchnorm)
#
#
#     # get_embeddings_to_interpret(drug_indices_of_interest,
#     #                                 dataset_dir,
#     #                                 outdir,
#     #                                 best_model_path,
#     #                                 norm,
#     #                                 do_layers,
#     #                                 batchnorm,
#     #                                 num_layers,
#     #                                 hidden_gcn,
#     #                                 hidden_fc,
#     #                                 splits_dir,
#     #                                 in_channels,
#     #                                 n_classes,
#     #                                 feature_names_path,
#     #                                 model_type,
#     #                                 k_hops)
#     # get_attributions_to_interpret(drug_indices_of_interest,
#     #                                 dataset_dir,
#     #                                 outdir,
#     #                                 best_model_path,
#     #                                 norm,
#     #                                 do_layers,
#     #                                 batchnorm,
#     #                                 num_layers,
#     #                                 hidden_gcn,
#     #                                 hidden_fc,
#     #                                 splits_dir,
#     #                                 in_channels,
#     #                                 n_classes,
#     #                                 gene_names_path,
#     #                                 model_type,
#     #                                 k_hops)
#     #
#     # get_attributions_to_interpret_lc(drug_indices_of_interest,
#     #                                 dataset_dir,
#     #                                 outdir,
#     #                                 best_model_path,
#     #                                 norm,
#     #                                 do_layers,
#     #                                 batchnorm,
#     #                                 num_layers,
#     #                                 hidden_gcn,
#     #                                 hidden_fc,
#     #                                 splits_dir,
#     #                                 in_channels,
#     #                                 n_classes,
#     #                                 gene_names_path,
#     #                                 model_type,
#     #                                 k_hops)
#
# #NORM
# norm = True
# for num_layers in [1,2,3]:
#
#
#     outdir = './out/chebwpathways/layers_{}/norm/split_{}'.format(num_layers, split)
#
#     ##find the best model for the split from the log
#     model_path = './../out/chebmodelwpathways/final_CV/layers_{}/norm'.format(num_layers)
#     log_path = osp.join(model_path, 'average_cv_performance.txt')
#     log = open(log_path, 'r')
#     best_model_info = log.readlines()[split]
#     best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#     best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
#
#     do_layers = int(best_model_info.split('_')[-5])
#     k_hops = int(best_model_info.split('_')[-3])
#     batchnorm = best_model_info.split('_')[-1] == 'True'
#
#
#     # get_embeddings_to_interpret(drug_indices_of_interest,
#     #                                 dataset_dir,
#     #                                 outdir,
#     #                                 best_model_path,
#     #                                 norm,
#     #                                 do_layers,
#     #                                 batchnorm,
#     #                                 num_layers,
#     #                                 hidden_gcn,
#     #                                 hidden_fc,
#     #                                 splits_dir,
#     #                                 in_channels,
#     #                                 n_classes,
#     #                                 feature_names_path,
#     #                                 model_type,
#     #                                 k_hops)
#     #
#     # get_attributions_to_interpret(drug_indices_of_interest,
#     #                                 dataset_dir,
#     #                                 outdir,
#     #                                 best_model_path,
#     #                                 norm,
#     #                                 do_layers,
#     #                                 batchnorm,
#     #                                 num_layers,
#     #                                 hidden_gcn,
#     #                                 hidden_fc,
#     #                                 splits_dir,
#     #                                 in_channels,
#     #                                 n_classes,
#     #                                 gene_names_path,
#     #                                 model_type,
#     #                                 k_hops)
#     #
#     # get_attributions_to_interpret_lc(drug_indices_of_interest,
#     #                                 dataset_dir,
#     #                                 outdir,
#     #                                 best_model_path,
#     #                                 norm,
#     #                                 do_layers,
#     #                                 batchnorm,
#     #                                 num_layers,
#     #                                 hidden_gcn,
#     #                                 hidden_fc,
#     #                                 splits_dir,
#     #                                 in_channels,
#     #                                 n_classes,
#     #                                 gene_names_path,
#     #                                 model_type,
#     #                                 k_hops)
