###Functions used for interpretation -- get_embeddings and get_attributions

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
from sklearn.preprocessing import MinMaxScaler
from captum.attr import (
    IntegratedGradients,
    LayerConductance)
import subprocess

torch.set_num_threads(5)




#Functions to compute attribution recall score

def preparing_data_all_positive_in_test(dataset_dir, pathway_dir, drug_ids_path, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops):
    device = torch.device('cpu')


    # Creates outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)



    ##########  DATA PREEPROCESSING ###########
    def preprocess_data(data, mean, std):
        data.y = data.y.to(torch.long)
        data.x = (data.x - mean) / (std)
        return data

    # Loads datasets
    drug_fp = dataset_dir
    pathway_fp = pathway_dir
    ac_split_fp = sorted(glob(osp.join(splits_dir,'split_cancer_*_indices_approved_mysql_on_onecc_noiso_string_dense.hkl')))

    drug_ids = drug_ids_path
    drug_ids = hkl.load(drug_ids)
    dict_drug_index_id = dict(zip(range(len(drug_ids)), drug_ids))


    test_data_mask = hkl.load(ac_split_fp[0])
    train_data_mask = hkl.load(ac_split_fp[1])
    val_data_mask = hkl.load(ac_split_fp[2])
    pathway_edge_index = torch.load(pathway_fp)
    dataset = torch.load(drug_fp)
    n_cmt = (pathway_edge_index[1].max() + 1).item()


    #Getting samples of interest in test set (classified as positive)
    test_pred = pd.read_csv(osp.join(best_model_path, '../best_model_results/test_pred.csv')).values.squeeze()
    drug_indices_of_interest = np.where(test_pred)[0].tolist()

    # Data normalization if args.norm
    if norm:
        tmp_train = torch.stack([dataset[i].x for i in train_data_mask], dim=0)
        mean, std = tmp_train.mean(), tmp_train.std()
        samples_of_interest = [preprocess_data(dataset[i], mean, std) for i in drug_indices_of_interest]

    else:
        samples_of_interest = [dataset[i] for i in drug_indices_of_interest]



    ##########  LOADING BEST MODEL ###########

    edge_index = samples_of_interest[0].edge_index.to(device)
    # Creates model and loads best configuration

    if model_type == 'gcn':
        model = GCNModel(in_channels, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, batchnorm = batchnorm, do_layers = do_layers).to(device)
    elif model_type == 'gcnwpathways':
        model = GCNModelWPathways(in_channels, n_classes, num_layers, hidden_gcn, hidden_fc, pathway = pathway_edge_index.to(device), n_cmt = n_cmt, edge_index = edge_index, batchnorm = batchnorm, do_layers = do_layers).to(device)
    elif model_type == 'cheb':
        model = ChebModel(in_channels, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index = edge_index, batchnorm = batchnorm, do_layers = do_layers, k_hops = k_hops).to(device)
    elif model_type == 'chebwpathways':
        model = ChebModelWPathways(in_channels, n_classes, num_layers, hidden_gcn, hidden_fc,  pathway = pathway_edge_index.to(device), n_cmt = n_cmt,  edge_index = edge_index, batchnorm = batchnorm, do_layers = do_layers, k_hops = k_hops).to(device)


    best_model_path = best_model_path + '/checkpoints/best_model.pt'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    loader = DataLoader(samples_of_interest, batch_size=1)
    feature_names = hkl.load(feature_names_path)
    #Capitalizing feature names
    feature_names = [e.upper() for e in feature_names]

    return model, samples_of_interest, loader, feature_names, device


##Computes attributions for all samples in drug_indices_of_interest and returns as data frame
def get_attributions_to_interpret_all_positive_in_test(dataset_dir, pathway_dir, drug_ids_path, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops, n_steps = 100):
    model, samples_of_interest, loader, feature_names, device = preparing_data_all_positive_in_test(dataset_dir, pathway_dir, drug_ids_path, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)

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
        attributions_list.append(scaler.fit_transform(attributions.detach().cpu().numpy().squeeze().reshape(-1,1)).reshape(1,-1))
        # print('IG Attributions:', attributions)
    #     log_handle = open(outdir+'/log_get_attributions_to_interpret.txt', 'w')
    #     scores = model(baseline.view(1, -1, 1), torch.tensor([0], device=device)).detach().cpu().numpy()[0]
    #     log_handle.write('n_steps\t{}\n'.format(n_steps))
    #     log_handle.write('Baseline score\tDelta\t:{}\t{}\t{}\n'.format(scores[0], scores[1],delta.item()))
    # log_handle.close()
    # Creates output dataframe
    # attributions_list = scaler.fit_transform(np.array(attributions_list).squeeze().reshape(-1,1)).reshape(1,-1)
    attributions = pd.DataFrame(np.vstack(attributions_list))
    attributions.columns = feature_names


    return attributions


##Using the data frame generated above, generates gsea reports (individual folders for each sample)
def get_gsea_reports(data_frame, outdir, name, gsea_path = './interpretation_influence_layers/'):
    for i in range(data_frame.shape[0]):
        gene_ranked_dataframe = data_frame.loc[i]
        gene_ranked_dataframe = pd.concat([pd.DataFrame(gene_ranked_dataframe.index), pd.DataFrame(gene_ranked_dataframe.values)], 1)
        gene_ranked_dataframe.columns = ['LABEL', 'WEIGHT']                       #Name columns
        gene_ranked_dataframe = gene_ranked_dataframe[gene_ranked_dataframe['LABEL'] != '']   #Remove genes without name
        outdir_i = outdir + '/{}'.format(i)
        if not os.path.isdir(outdir_i):
            os.makedirs(outdir_i)
        gene_ranked_dataframe.to_csv(outdir_i+'/list.rnk', sep = '\t', index = False)
        command = 'bash {}GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/list.rnk -gmx ./interpretation_influence_layers/GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}_KEGG'.format(gsea_path, outdir_i, outdir_i, name)
        subprocess.call(command, shell = True)
        command = 'rm {}/*/*.png'.format(outdir_i)
        subprocess.call(command, shell = True)
        command = 'rm {}/*/*.html'.format(outdir_i)
        subprocess.call(command, shell = True)
        command = 'rm -r {}/*/edb'.format(outdir_i)
        subprocess.call(command, shell = True)
        command = 'rm {}/*/KEGG*xls'.format(outdir_i)
        subprocess.call(command, shell = True)
    return



##Gets ratio of OR pathways as #AC-related pathways vs total AC pathwways for positive and negative, and as #AC-related pathways vs total OR pathwways for positive and negative
#indir is the path to the directory with the gsea results
def get_ratio_OR_pathways(indir, name, pathway_path):
	dict_results = {}
	#Pathway info
	pathway_info = pd.read_csv(pathway_path, sep = ',')  #Assignation matrix indicating which pathways are anticancer
	dict_pathway_info = dict(zip(pathway_info['pathway_name'], pathway_info['is_cancer_related']))
	total_ac_pathways = sum(pathway_info['is_cancer_related'].values)
	#Paths
	paths = sorted(glob(indir + '/{}_KEGG.GseaPreranked.*/gsea_report_for_na_*.xls'.format(name)))
	neg_path = paths[0]
	pos_path = paths[1]
	#Positive
	info = pd.read_csv(pos_path, sep = '\t')
	info = info[['NAME', 'FDR q-val']]
	name_or = list(info[info['FDR q-val'] < 0.25]['NAME'])
	###Computing ratio of cancer-related/all
	cancer_related = 0
	all = len(name_or)
	for name in name_or:
	    if dict_pathway_info[name] == 1:
	        cancer_related += 1
	if all != 0:
	    ratio = cancer_related / all
	else:
	    ratio = 0
	dict_results['pos_orac_or'] = ratio
	dict_results['pos_orac_ac'] = cancer_related/total_ac_pathways
	#Negative
	info = pd.read_csv(neg_path, sep = '\t')
	info = info[['NAME', 'FDR q-val']]
	name_or = list(info[info['FDR q-val'] < 0.25]['NAME'])
	###Computing ratio of cancer-related/all
	cancer_related = 0
	all = len(name_or)
	for name in name_or:
	    if dict_pathway_info[name] == 1:
	        cancer_related += 1
	if all != 0:
	    ratio = cancer_related / all
	else:
	    ratio = 0
	dict_results['neg_orac_or'] = ratio
	dict_results['neg_orac_ac'] = cancer_related/total_ac_pathways
	return dict_results
