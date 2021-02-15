##Given a trained model and samples of interest, compute attributions and GSEA report

import os.path as osp
import sys
sys.path.append('interpretation_influence_layers')
from interpretation_influence_layers.utils_interpretation_ import *
torch.set_num_threads(5)
from glob import glob
import argparse


#convert string to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 't', 'y', '1', 'true'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--base_outdir', default='test_intertp', type=str, help='Output path')
parser.add_argument('--model_type', default='cheb', help='Model type (cheb/gcn/sage)')
parser.add_argument('--model_path', default='cheb_nlayers_2_hidden_8', help='Path to folder with model')
parser.add_argument('--hidden_gcn', default=8, help='Hidden dimension convolutional layers')
parser.add_argument('--hidden_fc', default=32, help='Hidden dimension prediction layer')
parser.add_argument('--in_channels', default=1, help='Number of input features')
parser.add_argument('--n_classes', default=2, help='Number of classes')
parser.add_argument('--dataset_dir', default='dataset/test/raw_ac.py', help='Path to dataset')
parser.add_argument('--pathway_dir', default='dataset/test/pathway_kegg.py' , help='Path to pathway dataset')
parser.add_argument('--drug_ids_path', default='dataset/test/drugs_ids_approved_mysql_on_onecc_noiso_string_dense.hkl', help='Path to drug ids')
parser.add_argument('--n_splits', default=5, help='Number of splits')
parser.add_argument('--feature_names_path', default='dataset/onecc_noiso_string_gene_names.hkl', help='Path to feature names (genes)')
parser.add_argument('--splits_path', default='dataset/data_hyperfoods_drugcentral/5foldCVsplits/', help='Path to splits')
parser.add_argument('--num_layers', default=2, help='Number of layers')
parser.add_argument('--norm', default=False, type=str2bool, help='Whether data needs to be normalized')
parser.add_argument('--pathway_path', default='interpretation_influence_layers/pathways_info.csv', help='Path to file with pathway information')
args = parser.parse_args()

def compute_attribution_recall_score(base_outdir, model_type, model_path, hidden_gcn, hidden_fc, in_channels, n_classes, dataset_dir, pathway_dir, n_splits, feature_names_path, splits_path, num_layers, norm = False):
	name = 'attributions'
	pos_orac_or = []
	pos_orac_ac = []
	neg_orac_or = []
	neg_orac_ac = []
	if not norm:
		norm = 'raw'
	else:
		norm = 'norm'

	for split in range(n_splits):

		splits_dir = osp.join(splits_path, '{}'.format(split))
		outdir = osp.join(base_outdir,'layers_{}/split_{}'.format(num_layers, split))

		##find the best model for the split from the log
		log_path = osp.join(model_path, 'average_cv_performance.txt')
		log = open(log_path, 'r')
		best_model_info = log.readlines()[split]
		best_model_info = best_model_info.split('\t')[2].split('/')[-1]
		# print(best_model_info)

		######HERE
		best_model_path = osp.join(model_path,'{}/{}'.format(split, best_model_info))

		if model_type == 'gcn':
			do_layers = int(best_model_info.split('_')[-3])
			batchnorm = best_model_info.split('_')[-1] == 'True'
			k_hops = None
		elif model_type =='cheb':
			do_layers = int(best_model_info.split('_')[-5])
			k_hops = int(best_model_info.split('_')[-3])
			batchnorm = best_model_info.split('_')[-1] == 'True'


		data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, pathway_dir, drug_ids_path, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)


		get_gsea_reports(data_frame, outdir, name)

		#Gets ratio OR pathways for all and average
		paths = sorted(glob(outdir+ '/*'))
		for outdir_i in paths:
		    r = get_ratio_OR_pathways(outdir_i, name, pathway_path)
		    pos_orac_or.append(r['pos_orac_or'])
		    pos_orac_ac.append(r['pos_orac_ac'])
		    neg_orac_or.append(r['neg_orac_or'])
		    neg_orac_ac.append(r['neg_orac_ac'])






	log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
	log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
	log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
	                                                        np.std(pos_orac_or),
	                                                        np.mean(pos_orac_ac),
	                                                        np.std(pos_orac_ac),
	                                                        np.mean(neg_orac_or),
	                                                        np.std(neg_orac_or),
	                                                        np.mean(neg_orac_ac),
	                                                        np.std(neg_orac_ac)))
	log_handle.close()





compute_attribution_recall_score(args.base_outdir, 
								 args.model_type, 
								 args.model_path, 
								 args.hidden_gcn, 
								 args.hidden_fc, 
								 args.in_channels, 
								 args.n_classes, 
								 args.dataset_dir, 
								 args.pathway_dir, 
								 args.n_splits, 
								 args.feature_names_path, 
								 args.splits_path, 
								 args.num_layers, 
								 args.norm)


