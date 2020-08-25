
###Computes attributions for top correctly classified and misclassified samples
#MODEL: ChebNet 2 layers RAW

import pandas as pd
import os.path as osp
from get_embeddings_and_attributions_to_interpret import get_attributions_to_interpret, get_embeddings_to_interpret
import subprocess

base_path = '../dataset/data_hyperfoods_drugcentral/5foldCVsplits/'

###Model
dataset_dir = '../dataset'
hidden_gcn = 8
hidden_fc = 32
in_channels = 1
n_classes = 2
feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
model_type = 'cheb'
norm = False
num_layers = 2


####################
##Top misclassified
####################
path = osp.join(base_path, 'log_location_drugs_to_interpret_top_3_misclassified.txt')
info = pd.read_csv(path, sep = '\t')
for i in range(len(info)):
	drug_indices_of_interest = [info.at[i, 'DrugIndex']]
	outdir = './attributions_interpretation_top/misclassified/{}'.format(info.at[i, 'DrugID'])
	split = info.at[i, 'Split']
	splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
	model_path = './../out/chebmodel/final_CV/layers_{}/raw'.format(num_layers)
	log_path = osp.join(model_path, 'average_cv_performance.txt')
	log = open(log_path, 'r')
	best_model_info = log.readlines()[split]
	best_model_info = best_model_info.split('\t')[2].split('/')[-1]
	best_model_path = model_path+ '/{}/{}'.format(split, best_model_info)
	do_layers = int(best_model_info.split('_')[-5])
	k_hops = int(best_model_info.split('_')[-3])
	batchnorm = best_model_info.split('_')[-1] == 'True'
	get_attributions_to_interpret(drug_indices_of_interest, 
												dataset_dir, 
												outdir, 
												best_model_path, 
												norm, 
												do_layers, 
												batchnorm, 
												num_layers, 
												hidden_gcn, 
												hidden_fc, 
												splits_dir, 
												in_channels, 
												n_classes, 
												feature_names_path, 
												model_type, 
												k_hops, 
												n_steps = 100)

for i in range(len(info)):
	outdir = './attributions_interpretation_top/misclassified/{}'.format(info.at[i, 'DrugID'])
	name = 'attributions'
	command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/input_features_attribution_gsea.rnk -gmx ./GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir, outdir, name)
	subprocess.call(command, shell = True)



############################
###Top correctly classified
############################
path = osp.join(base_path, 'log_location_drugs_to_interpret_top_3_classified.txt')
info = pd.read_csv(path, sep = '\t')

for i in range(len(info)):
	drug_indices_of_interest = [info.at[i, 'DrugIndex']]
	outdir = './attributions_interpretation_top/classified/{}'.format(info.at[i, 'DrugID'])
	split = info.at[i, 'Split']
	splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)

	model_path = './../out/chebmodel/final_CV/layers_{}/raw'.format(num_layers)
	log_path = osp.join(model_path, 'average_cv_performance.txt')
	log = open(log_path, 'r')
	best_model_info = log.readlines()[split]
	best_model_info = best_model_info.split('\t')[2].split('/')[-1]
	best_model_path = model_path+ '/{}/{}'.format(split, best_model_info)
	do_layers = int(best_model_info.split('_')[-5])
	k_hops = int(best_model_info.split('_')[-3])
	batchnorm = best_model_info.split('_')[-1] == 'True'

	
	
	get_attributions_to_interpret(drug_indices_of_interest, 
												dataset_dir, 
												outdir, 
												best_model_path, 
												norm, 
												do_layers, 
												batchnorm, 
												num_layers, 
												hidden_gcn, 
												hidden_fc, 
												splits_dir, 
												in_channels, 
												n_classes, 
												feature_names_path, 
												model_type, 
												k_hops, 
												n_steps = 100)

for i in range(len(info)):
	outdir = './attributions_interpretation_top/classified/{}'.format(info.at[i, 'DrugID'])
	name = 'attributions'
	command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/input_features_attribution_gsea.rnk -gmx ./GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir, outdir, name)
	subprocess.call(command, shell = True)










###########################
#EMBEDDINGS
###########################

####################
##Top misclassified
####################
path = osp.join(base_path, 'log_location_drugs_to_interpret_top_3_misclassified.txt')
info = pd.read_csv(path, sep = '\t')
for i in range(len(info)):
	drug_indices_of_interest = [info.at[i, 'DrugIndex']]
	outdir = './attributions_interpretation_top/misclassified/{}'.format(info.at[i, 'DrugID'])
	split = info.at[i, 'Split']
	splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
	model_path = './../out/chebmodel/final_CV/layers_{}/raw'.format(num_layers)
	log_path = osp.join(model_path, 'average_cv_performance.txt')
	log = open(log_path, 'r')
	best_model_info = log.readlines()[split]
	best_model_info = best_model_info.split('\t')[2].split('/')[-1]
	best_model_path = model_path+ '/{}/{}'.format(split, best_model_info)
	do_layers = int(best_model_info.split('_')[-5])
	k_hops = int(best_model_info.split('_')[-3])
	batchnorm = best_model_info.split('_')[-1] == 'True'
	get_embeddings_to_interpret(drug_indices_of_interest, 
												dataset_dir, 
												outdir, 
												best_model_path, 
												norm, 
												do_layers, 
												batchnorm, 
												num_layers, 
												hidden_gcn, 
												hidden_fc, 
												splits_dir, 
												in_channels, 
												n_classes, 
												feature_names_path, 
												model_type, 
												k_hops)

for i in range(len(info)):
	outdir = './attributions_interpretation_top/misclassified/{}'.format(info.at[i, 'DrugID'])
	name = 'embeddings'
	command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/drug_embeddings_gsea.rnk -gmx ./GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir, outdir, name)
	subprocess.call(command, shell = True)


for i in range(len(info)):
	outdir = './attributions_interpretation_top/misclassified/{}'.format(info.at[i, 'DrugID'])
	name = 'embeddings_GO_bp'
	command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/drug_embeddings_gsea.rnk -gmx ./GSEA_Linux_4.0.3/c5.bp.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir, outdir, name)
	subprocess.call(command, shell = True)


############################
###Top correctly classified
############################
path = osp.join(base_path, 'log_location_drugs_to_interpret_top_3_classified.txt')
info = pd.read_csv(path, sep = '\t')

for i in range(len(info)):
	drug_indices_of_interest = [info.at[i, 'DrugIndex']]
	outdir = './attributions_interpretation_top/classified/{}'.format(info.at[i, 'DrugID'])
	split = info.at[i, 'Split']
	splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)

	model_path = './../out/chebmodel/final_CV/layers_{}/raw'.format(num_layers)
	log_path = osp.join(model_path, 'average_cv_performance.txt')
	log = open(log_path, 'r')
	best_model_info = log.readlines()[split]
	best_model_info = best_model_info.split('\t')[2].split('/')[-1]
	best_model_path = model_path+ '/{}/{}'.format(split, best_model_info)
	do_layers = int(best_model_info.split('_')[-5])
	k_hops = int(best_model_info.split('_')[-3])
	batchnorm = best_model_info.split('_')[-1] == 'True'

	
	
	get_embeddings_to_interpret(drug_indices_of_interest, 
												dataset_dir, 
												outdir, 
												best_model_path, 
												norm, 
												do_layers, 
												batchnorm, 
												num_layers, 
												hidden_gcn, 
												hidden_fc, 
												splits_dir, 
												in_channels, 
												n_classes, 
												feature_names_path, 
												model_type, 
												k_hops)

for i in range(len(info)):
	outdir = './attributions_interpretation_top/classified/{}'.format(info.at[i, 'DrugID'])
	name = 'embeddings'
	command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/drug_embeddings_gsea.rnk -gmx ./GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir, outdir, name)
	subprocess.call(command, shell = True)


for i in range(len(info)):
	outdir = './attributions_interpretation_top/classified/{}'.format(info.at[i, 'DrugID'])
	name = 'embeddings_GO_bp'
	command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/drug_embeddings_gsea.rnk -gmx ./GSEA_Linux_4.0.3/c5.bp.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir, outdir, name)
	subprocess.call(command, shell = True)