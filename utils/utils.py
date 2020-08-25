import os
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
import numpy as np
from glob import glob


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def to_batch(edge_index, batch_size, row_nodes, col_nodes):
    # designed for the case of row_nodes != col_nodes
    edge_index_list = []
    device = edge_index.device
    row, col = edge_index
    row_num_nodes = row_nodes
    col_num_nodes = col_nodes
    num_nodes = torch.LongTensor([row_num_nodes,
                                  col_num_nodes]).unsqueeze(-1).to(device)
    cumsum = 0
    for _ in range(batch_size):
        edge_index_list.append(edge_index + cumsum)
        cumsum += num_nodes
    return torch.cat(edge_index_list, dim=1)


def find_file(dir, key):
    "return the file path of a key-file within sorted file names in dir"
    "key : str"
    if key is None:
        return None
    folders = sorted(glob(os.path.join(dir, '*')))
    idx = 0
    for i, s in enumerate(folders):
        if key in s:
            idx = i
            break
        if i == len(folders) - 1:
            raise NotFoundException('not found file with key {}'.format(key))
    return folders[idx]


def save_embeddings(model, dataloader, device, out_dir, file_name, epoch=None):
    import hickle as hkl
    from .train_eval import to_embeddings
    z = to_embeddings(model, dataloader, device)
    if epoch is not None:
        fp = os.path.join(out_dir,
                          '{}_embeddings_{:03d}.hkl'.format(file_name, epoch))
    else:
        fp = os.path.join(out_dir, '{}_embeddings.hkl'.format(file_name))
    hkl.dump(z.numpy(), fp)
    # drug_embeddings, patient_embeddings = split_embeddings(
    #     z, num_drug, num_patient)
    # drug_fp = os.path.join(out_dir, 'drug_embeddings.hkl')
    # patient_fp = os.path.join(out_dir, 'patient_embeddings.hkl')
    # hkl.dump(drug_embeddings.numpy(), drug_fp)
    # hkl.dump(patient_embeddings.numpy(), patient_fp)


def att_weights(data, weight, att, bias, device):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    row, col = edge_index
    edge_index_i = col
    weight, att, bias = weight.to(device), att.to(device), bias.to(device)

    x = torch.mm(x, weight).view(-1, 8, 8)
    x_i = x[col]
    x_j = x[row]
    num_nodes = x.size(0)

    alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)
    alpha = F.leaky_relu(alpha, negative_slope=0.2)
    alpha = softmax(alpha, edge_index_i, num_nodes)
    alpha = alpha.mean(dim=1)
    return alpha.cpu().numpy()


def model_att_weights(model, dataset, device):
    child = next(iter(model.children()))
    weight = child.weight.cpu().detach()
    att = child.att.cpu().detach()
    bias = child.att.cpu().detach()

    alpha_list = []
    for idx, data in enumerate(dataset):
        alpha_list.append(att_weights(data, weight, att, bias, device))
        print(idx)
    return np.array(alpha_list)


def pos_neg_indices(dataset):
    y_list = []
    for data in dataset:
        y_list.append(data.y)
    y_array = np.array(y_list)
    return (y_array == 1).nonzero()[0], (y_array == 0).nonzero()[0]
