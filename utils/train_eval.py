import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, balanced_accuracy_score, average_precision_score
import matplotlib.pyplot as plt




#Functions to train and evaluate models


def gcnmodel_run(model, train_loader, val_loader, test_loader, sample_weights,
              start_epoch, epochs, optimizer,
              scheduler, writer, device):

    for epoch in range(start_epoch, epochs + 1):
        t = time.time()

        train_loss = gcnmodel_train(model, optimizer, train_loader,
                                 sample_weights,
                                 device)
        t_duration = time.time() - t
        t_duration_per_batch = (time.time() - t) / len(train_loader)
        val_perf = gcnmodel_eval(model, val_loader, sample_weights
                            , device)
        perf = gcnmodel_eval(model, test_loader, sample_weights
                          , device)
        scheduler.step()

        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': perf['loss'],
            'val_loss': val_perf['loss'],
            'acc': perf['acc'],
            'f1': perf['f1'],
            'roc_auc': perf['roc_auc'],
            'aupr': perf['aupr'],
            't_duration': t_duration,
            'balanced_acc':perf['balanced_acc'],
            't_duration_per_batch': t_duration_per_batch
        }

        if writer is not None:
            writer.gcnmodel_print_info(info)
            writer.gcnmodel_plot_info(info)
            writer.save_checkpoint(model, optimizer, scheduler, epoch)

    writer.close()


def gcnmodel_run_es(model, train_loader, val_loader, test_loader, sample_weights
                 , start_epoch, epochs, optimizer,
                 scheduler, writer, device, eval = True):

    #ES params
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    tol = 1e-4

    for epoch in range(start_epoch, epochs + 1):
        t = time.time()

        train_loss = gcnmodel_train(model, optimizer, train_loader,
                                 sample_weights,
                                 device)
        t_duration = time.time() - t
        t_duration_per_batch = (time.time() - t) / len(train_loader)


        val_perf = gcnmodel_eval(model, val_loader, sample_weights,
                              device)

        if eval == True:
            perf = gcnmodel_eval(model, test_loader, sample_weights,
                               device)

            info = {
                'current_epoch': epoch,
                'epochs': epochs,
                'train_loss': train_loss,
                'test_loss': perf['loss'],
                'val_loss': val_perf['loss'],
                'acc': perf['acc'],
                'f1': perf['f1'],
                'roc_auc': perf['roc_auc'],
                'aupr': perf['aupr'],
                't_duration': t_duration,
                'balanced_acc':perf['balanced_acc'],
                't_duration_per_batch': t_duration_per_batch
            }
            # import pdb;      pdb.set_trace()
            if writer is not None:
                writer.gcnmodel_print_info(info)
                writer.gcnmodel_plot_info(info)

        scheduler.step()
        if writer is not None:
            writer.save_checkpoint(model, optimizer, scheduler, epoch)

        if best_val_loss - val_perf['loss'] >= tol:
            counter = 0
            best_val_loss = val_perf['loss']
            writer.save_best_model(model, optimizer, scheduler, epoch)
        else:
            counter += 1
            if counter > patience:  #If counter is greater than patience, stops training
                print('Early stopping..')
                writer.close()
                break

    writer.close()
    return best_val_loss


def gcnmodel_train(model, optimizer, loader, sample_weights, device):
    model.train()
    # import pdb;  pdb.set_trace()
    total_loss = 0
    weight = sample_weights.to(device)
    for data in loader:

        optimizer.zero_grad()
        x, batch, y = data.x.to(device), data.batch.to(device), data.y.to(
            device)

        pred = model(x, batch)


        #MLP loss
        loss = F.nll_loss(pred, y, weight=weight)


        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(loader)






def gcnmodel_eval(model, loader, sample_weights,
               device):
    model.eval()

    weight = sample_weights.to(device)
    total_loss = 0
    correct = 0
    y_prob_list = []
    y_test_list = []
    y_pred_list = []
    with torch.no_grad():
        for data in loader:
            x, batch, y = data.x.to(device), data.batch.to(device), data.y.to(
                device)
            pred = model(x, batch)


            #MLP loss
            loss = F.nll_loss(pred, y, weight=weight)



            total_loss += loss.item()

            pred_ind = pred.max(1)[1]
            y_test_list.append(y.cpu().numpy())
            y_prob_list.append(np.exp(pred[:, 1].cpu().numpy()))
            y_pred_list.append(pred_ind.cpu().numpy())
            correct += pred_ind.eq(y).sum().item()

    y_test = np.concatenate(y_test_list).reshape(-1)
    y_prob = np.concatenate(y_prob_list).reshape(-1)
    y_pred = np.concatenate(y_pred_list).reshape(-1)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    aupr = auc(recall, precision)
    acc = correct / len(loader.dataset)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    pos_prec = precision_score(y_test, y_pred)
    neg_prec = precision_score(y_test, y_pred, pos_label=0)
    pos_recall = recall_score(y_test, y_pred)
    neg_recall = recall_score(y_test, y_pred, pos_label=0)

    average_precision = average_precision_score(y_test, y_prob)

    perf = {
        'f1': f1,
        'roc_auc': roc_auc,
        'aupr': aupr,
        'acc': acc,
        'loss': total_loss / len(loader),
        'pos_prec': pos_prec,
        'pos_recall': pos_recall,
        'neg_prec': neg_prec,
        'neg_recall': neg_recall,
        'balanced_acc': balanced_acc,
        'precision_i': precision,
        'recall_i': recall,
        'average_precision_i':average_precision,
        'y_test_i': y_test,
        'y_prob_i': y_prob,
        'y_pred_i': y_pred



    }

    return perf