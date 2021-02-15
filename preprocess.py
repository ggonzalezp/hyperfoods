#Pre-process input dataset
import os
import hickle as hkl
import os
import os.path as osp
import torch
from glob import glob
import numpy as np
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data, DataLoader

outdir = 'dataset/'
os.makedirs(outdir, exist_ok=True)


#Pre-processing pathway assignation matrix

#Loads KEGG
pathway_kegg_fp = './dataset/pathways_matrix_onecc_noiso_filled.hkl'
pathway_kegg = hkl.load(pathway_kegg_fp)

# generate pathway_edge_index from pathway matrix
pathway_kegg_mat = torch.from_numpy(pathway_kegg.values)
pathway_kegg_edge_index = pathway_kegg_mat.nonzero().t()

pathway_kegg_data_fp = osp.join(outdir, 'pathway_kegg.py')
torch.save(pathway_kegg_edge_index, pathway_kegg_data_fp)




#Pre-processing dataset

ac_gene_fp = sorted(glob('./dataset/drugs_profiles_approved_mysql_on_onecc_noiso_string_dense.hkl'))
ac_label_fp = './dataset/drugs_labels_approved_mysql_on_onecc_noiso_string_dense.hkl'
ac_label = np.array(hkl.load(ac_label_fp))

# binary feature for each gene
ac_gene_list = []
for i in range(len(ac_gene_fp)):
    ac_gene_list.append(hkl.load(ac_gene_fp[i]))
    
    
#Loads PPI 
ppi_fp = './dataset/onecc_noiso_string_matrix_interactome_data_filtered.hkl'
ppi = hkl.load(ppi_fp)


# generate graph topology and edge_attr
edge_index = torch.from_numpy(ppi[:, 0:2]).transpose(dim0=0, dim1=1)
assert (is_undirected(edge_index) is True), "ppi graph should be undirected graph"
edge_attr = torch.from_numpy(ppi[:, 2]).to(torch.float32) / 1000.0


# create data with single feature 
ac_data_list = list([] for i in range(ac_gene_list[0].shape[0]))

for i in range(len(ac_gene_list)):
    for idx, vector in enumerate(ac_gene_list[i]):
        ac_data_list[idx].append(torch.from_numpy(vector))                


new_ac_data_list = []
for data in ac_data_list:
    new_ac_data_list.append(torch.stack(data, dim=0).t())
    
    
    
    
# create torch_geometric.data.Data list
def create_data_list(node_feature_mat, labels):
    # list: node_feature_mat with dim [num_drugs, -]
    # torch.Tensor for each feature with dim [num_nodes, feature_dim]
    data_list = []
    if labels is not None:
        for feature, label in zip(node_feature_mat, labels):
            data_list.append(Data(x=feature.type(torch.float32), edge_index=edge_index, edge_attr=edge_attr.type(torch.float32), y=torch.tensor([label]).type(torch.long)))
        return data_list
    else:
        for feature in node_feature_mat:
            data_list.append(Data(x=feature.type(torch.float32), edge_index=edge_index, edge_attr=edge_attr.type(torch.float32)))
        return data_list

ac_gene_data = create_data_list(new_ac_data_list, ac_label)

#Saves dataset
ac_data_fp = osp.join(outdir, 'raw_ac.py')

torch.save(ac_gene_data, ac_data_fp)

