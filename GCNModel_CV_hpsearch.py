###GCN Model script for cross-validation




import argparse
import os.path as osp
import numpy as np
import torch
import hickle as hkl
from glob import glob

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from utils import utils, train_eval, DataLoader
from utils.writer import Writer
from models.networks_gcn import GCNModel
from utils import train_eval

import json
import pandas as pd
from sklearn.metrics import roc_curve, auc
from inspect import  signature
import matplotlib.pyplot as plt
torch.set_num_threads(2)
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--dataset_dir', default='./dataset/test', help='input data path ')
parser.add_argument('--out_dir', default='./out_test_cv', help='output data path ')


parser.add_argument('--device_idx', default=8, type=int)

#Dataset
parser.add_argument('--norm', default=True, help='Whether to standardize dataset')

# network hyperparameters
parser.add_argument('--num_layers', default=2, type=int, help='num layers')

parser.add_argument('--hidden_gcn', default=8, type=int, help='hidden gcn')

parser.add_argument('--hidden_fc', default=32, help='hidden fc')


parser.add_argument('--feature_agg_mode', default='cat', type=str, help='Feature aggregation mode: cat/base/sum')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
# parser.add_argument('--lr', type=float, default=5e-3, help='Learning Rate')
parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay per epoch')
parser.add_argument('--decay_step', type=int, default=1)
# parser.add_argument('--regularization', type=float, default=1e-5, help='L2 regularization')



# training hyperparameters
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--epochs',type=int, default=50, help='number of epochs to train')

# others
parser.add_argument('--seed', type=int, default=1,help='random seed (default: 1)')

args = parser.parse_args()


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


args.norm = str2bool(args.norm)

if args.seed is not None:
    import numpy as np
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# load dataset
def preprocess_data(data, mean, std):
    data.y = data.y.to(torch.long)
    #     data.x = data.x[:, 2].unsqueeze(-1)
    data.x = (data.x - mean) / (std)
    return data


def cal_weights(dataset):
    labels = []
    for data in dataset:
        labels.append(data.y)
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negcnive = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    return torch.tensor([n_full / (2 * n_negcnive), n_full / (2 * n_positive)])



drug_fp = osp.join(args.dataset_dir, 'raw_ac.py')
pathway_fp = osp.join(args.dataset_dir, 'pathway_kegg.py')


##CV
###For each of the folds, we use the validation set to find optimal hyperparameters (we train models with a grid search and evaluate the model with the lowest valdation loss value)

best_models_names = ''
perfs_rounds = []
for i in range(5):

    ac_split_fp = sorted(glob('./dataset/5foldCVsplits/{}/split_cancer_*_indices_approved_mysql_on_onecc_noiso_string_dense.hkl'.format(i)))

    out_dir_i = args.out_dir + '/{}'.format(i)

    device = torch.device('cuda', args.device_idx)

    ####DATA PROCESSING
    test_data_mask = hkl.load(ac_split_fp[0])
    train_data_mask = hkl.load(ac_split_fp[1])
    val_data_mask = hkl.load(ac_split_fp[2])
    pathway_edge_index = torch.load(pathway_fp)
    dataset = torch.load(drug_fp)

    if args.norm:
        tmp_train = torch.stack([dataset[i].x for i in train_data_mask], dim=0)
        mean, std = tmp_train.mean(), tmp_train.std()

        train_dataset = [
            preprocess_data(dataset[i], mean, std) for i in train_data_mask
        ]
        val_dataset = [
            preprocess_data(dataset[i], mean, std) for i in val_data_mask
        ]
        test_dataset = [
            preprocess_data(dataset[i], mean, std) for i in test_data_mask
        ]
    else:
        train_dataset = [dataset[i] for i in train_data_mask]
        val_dataset = [dataset[i] for i in val_data_mask]
        test_dataset = [dataset[i] for i in test_data_mask]

    sample_weights = cal_weights(train_dataset)

    n_cmt = (pathway_edge_index[1].max() + 1).item()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    edge_index = train_dataset[0].edge_index.to(device)
    edge_weights = train_dataset[0].edge_attr.to(device)

    if train_dataset[0].x.dim() == 2:
        in_channels = train_dataset[0].x.size(1)
    else:
        in_channels = 1

    n_classes = 2

    n_genes = train_dataset[0].x.size(0)



    ######OPTIMIZING MODEL


    start_epoch = 1

    best_model_path = ''
    best_val_loss = 1e1000
    best_bn = -1
    best_do = -1


    for lr in [0.0005, 0.005]:
        for l2 in [0.00001, 0.0001, 0.0005]:
            for do_layers in [1,2]:
                for bn in [True, False]:
                    args.checkpoints_dir = out_dir_ii = out_dir_i + '/lr_{}_l2_{}_do_{}_bn_{}/checkpoints'.format(lr,l2, do_layers, bn)
                    utils.makedirs(args.checkpoints_dir)
                    out_dir_ii = out_dir_i + '/lr_{}_l2_{}_do_{}_bn_{}'.format(lr,l2, do_layers, bn)
                    writer = Writer(args, outdir=out_dir_ii)


                    #MODEL DEFINITION
                    model = GCNModel(in_channels, n_classes, args.num_layers,
                                     args.hidden_gcn, args.hidden_fc, edge_index, mode=args.feature_agg_mode,batchnorm=bn, do_layers = do_layers, edge_weights = edge_weights).to(device)

                    args.model_parameters = utils.count_parameters(model)
                    #     writer.save_args()
                    print('Number of parameters: {}'.format(args.model_parameters))
                    print(model)



                    # OPTIMIZER
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_decay)

                    val_loss = train_eval.gcnmodel_run_es(
                        model,
                        train_loader,
                        val_loader,
                        test_loader,
                        sample_weights.to(device),
                        start_epoch,
                        args.epochs,
                        optimizer,
                        scheduler,
                        writer,
                        device)


                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = out_dir_ii
                        best_bn = bn
                        best_do = do_layers





    best_models_names += 'split\t{}\t{}\t'.format(i, best_model_path)
    best_model_path = best_model_path + '/checkpoints/best_model.pt'


    ###Eval on test set
    model = GCNModel(in_channels, n_classes, args.num_layers, args.hidden_gcn, args.hidden_fc, edge_index, mode=args.feature_agg_mode, batchnorm=best_bn, do_layers = best_do, edge_weights = edge_weights).to(device)

    checkpoint = torch.load(best_model_path)
    best_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    perf = train_eval.gcnmodel_eval(model, test_loader, sample_weights.to(device), device)
    best_models_names += '\tBalanced accuracy\tF1\tAUCROC\tAUPR\tPosPrec\tPosRecall\tNegPrec\tNegRecall\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n'.format(  perf['balanced_acc'],
                                                                                 perf['f1'],
                                                                                 perf['roc_auc'],
                                                                                 perf['aupr'],
                                                                                 perf['pos_prec'],
                                                                                 perf['pos_recall'],
                                                                                 perf['neg_prec'],
                                                                                perf['neg_recall']

                                                                                  )

    perfs_rounds.append(perf)



    ######### PLOTTING CURVES

    precision = perf['precision_i']
    recall = perf['recall_i']
    average_precision = perf['average_precision_i']
    y_test = perf['y_test_i']
    y_prob = perf['y_prob_i']
    y_pred = perf['y_pred_i']

    #Saving labels
    outdir_plots = out_dir_i+'/best_model_results'
    if not osp.isdir(outdir_plots):
        os.makedirs(outdir_plots)

    pd.DataFrame(y_pred).to_csv(outdir_plots + '/test_pred.csv', index=False, header=None)
    pd.DataFrame(y_prob).to_csv(outdir_plots + '/test_prob.csv', index=False, header=None)
    pd.DataFrame(y_test).to_csv(outdir_plots +'/test_real.csv', header=None, index=False)



    ##Plot AUROC and AUPR curves

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
    plt.savefig(outdir_plots +'/AUPR.png')



    #ROC AUC curve
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_prob)
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
    plt.savefig(outdir_plots + '/AUCROC.png')







#SUMMARY CV METRICS
balanced_accs = []
rocaucs = []
f1s = []
auprs = []
losses = []
pos_precs = []
pos_recalls = []
neg_precs = []
neg_recalls = []
for perf in perfs_rounds:
    balanced_accs.append(perf['balanced_acc'])
    rocaucs.append(perf['roc_auc'])
    f1s.append(perf['f1'])
    auprs.append(perf['aupr'])
    losses.append(perf['loss'])
    pos_precs.append(perf['pos_prec'])
    pos_recalls.append(perf['pos_recall'])
    neg_precs.append(perf['neg_prec'])
    neg_recalls.append(perf['neg_recall'])

balanced_accs = np.array(balanced_accs)
rocaucs = np.array(rocaucs)
f1s = np.array(f1s)
auprs = np.array(auprs)
losses = np.array(losses)

summ = best_models_names +  ' Balanced accuracy\tstd\tF1\tstd\tAUCROC\tstd\tAUPR\tstd\tPosPrec\tstd\tPosRecall\tstd\tNegPrec\tstd\tNegRecall\tstd\n' \
       '{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(   np.mean(balanced_accs),
                                                                                   np.std(balanced_accs),
                                                                                    np.mean(f1s),
                                                                                   np.std(f1s),
                                                                                   np.mean(rocaucs),
                                                                                   np.std(rocaucs),
                                                                                   np.mean(auprs),
                                                                                   np.std(auprs),
                                                                                  np.mean(pos_precs),
                                                                                  np.std(pos_precs),
                                                                                  np.mean(pos_recalls),
                                                                                  np.std(pos_recalls),
                                                                                  np.mean(neg_precs),
                                                                                  np.std(neg_precs),
                                                                                  np.mean(neg_recalls),
                                                                                  np.std(neg_recalls)

                                                                                  )
print(summ)
log_handle = open(args.out_dir + '/average_cv_performance.txt', 'w')
log_handle.write(summ)
log_handle.close()
