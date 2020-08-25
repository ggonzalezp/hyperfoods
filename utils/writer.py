import os
import time
import torch
import json
from utils.utils import makedirs

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None
"""
args.out_dir
    args.checkpoints_dir
args.name
"""


class Writer:
    def __init__(self, args, outdir = None):
        self.args = args
        self.outdir = outdir
        if outdir is not None:
            self.log_file = os.path.join(
            outdir, 'log_{:s}.txt'.format(
                time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))

            if SummaryWriter is not None:
                self.display = SummaryWriter(outdir)
            else:
                self.display = None


        else:

            self.log_file = os.path.join(
            args.out_dir, 'log_{:s}.txt'.format(
                time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))

            if SummaryWriter is not None:
                self.display = SummaryWriter(self.args.out_dir)
            else:
                self.display = None

    def print_info(self, info):
        message = (
            'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, '
            'Test Loss: {:.4f}, Test Acc: {:.3f}, F1: {:.3f}, roc_auc: {:.3f}'
        ).format(info['current_epoch'], info['epochs'], info['t_duration'],
                 info['train_loss'], info['test_loss'], info['test_acc'],
                 info['f1'], info['roc_auc'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def plot_info(self, info):
        if self.display:
            self.display.add_scalars('data/loss', {
                'train_loss': info['train_loss'],
                'test_loss': info['test_loss']
            }, info['current_epoch'])
            self.display.add_scalar('data/acc', info['test_acc'],
                                    info['current_epoch'])

    def classifier_print_info(self, info):
        message = (
            'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, '
            'Test Loss: {:.4f}, Val Loss: {:.4f}, '
            'Test Acc: {:.3f}, F1: {:.3f}, real_f1: {:.3f}, roc_auc: {:.3f}, aupr: {:.4f}'
        ).format(info['current_epoch'], info['epochs'], info['t_duration'],
                 info['train_loss'], info['test_loss'], info['val_loss'],
                 info['acc'], info['f1'], info['real_f1'], info['roc_auc'],
                 info['aupr'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def classifier_plot_info(self, info):
        if self.display:
            self.display.add_scalars(
                'data/loss', {
                    'train_loss': info['train_loss'],
                    'val_loss': info['val_loss'],
                    'test_loss': info['test_loss']
                }, info['current_epoch'])
            self.display.add_scalar('data/acc', info['acc'],
                                    info['current_epoch'])
            self.display.add_scalar('data/f1', info['f1'],
                                    info['current_epoch'])
            self.display.add_scalar('data/real_f1', info['real_f1'],
                                    info['current_epoch'])
            self.display.add_scalar('data/roc_auc', info['roc_auc'],
                                    info['current_epoch'])
            self.display.add_scalar('data/aupr', info['aupr'],
                                    info['current_epoch'])

    def gcnmodel_print_info(self, info):
        message = (
            'Epoch: {}/{}, Duration: {:.3f}s, Duration per batch: {:.3f}s, Train Loss: {:.4f}, '
            'Test Loss: {:.4f}, Val Loss: {:.4f},  '
            'Test Balanced Acc: {:.3f}, F1: {:.3f}, real_f1: {:.3f}, roc_auc: {:.3f}, aupr: {:.4f}'
        ).format(info['current_epoch'], info['epochs'], info['t_duration'], info['t_duration_per_batch'],
                 info['train_loss'], info['test_loss'],
                  info['val_loss'],
                  info['balanced_acc'], info['f1'], info['real_f1'],
                 info['roc_auc'], info['aupr'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def gcnmodel_plot_info(self, info, model = None):
        if self.display:
            self.display.add_scalars('data/loss', {
                'train_loss': info['train_loss'],
                'test_loss': info['test_loss'],
                'val_loss': info['val_loss']
            }, info['current_epoch'])

            self.display.add_scalar('data/test/acc', info['acc'],
                                    info['current_epoch'])
            self.display.add_scalar('data/test/f1', info['f1'],
                                    info['current_epoch'])
            self.display.add_scalar('data/test/real_f1', info['real_f1'],
                                    info['current_epoch'])
            self.display.add_scalar('data/test/roc_auc', info['roc_auc'],
                                    info['current_epoch'])
            self.display.add_scalar('data/test/aupr', info['aupr'],
                                    info['current_epoch'])
            # if model is not None:
            #     for tag, parm in model.named_parameters():
            #          self.display.add_histogram('data/'+tag, parm.grad.data.cpu().numpy(), info['current_epoch'])

    def gcnmodel_print_info_splits(self, info):
        message = (
            'Epoch: {}/{}, Duration: {:.3f}s,  Train Loss: {:.4f}, '
            'Val Loss: {:.4f}, '
            'Acc: {:.3f}, F1: {:.3f}, real_f1: {:.3f}, roc_auc: {:.3f}, aupr: {:.4f}'
        ).format(info['current_epoch'], info['epochs'], info['t_duration'],
                 info['train_loss'], info['val_loss'],  info['acc'], info['f1'], info['real_f1'],
                 info['roc_auc'], info['aupr'])

        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def gcnmodel_plot_info_splits(self, info):
        if self.display:
            self.display.add_scalar('data/train_loss', info['train_loss'],
                                    info['current_epoch'])
            self.display.add_scalar('data/val_loss', info['val_loss'],
                                    info['current_epoch'])
            self.display.add_scalar('data/acc', info['acc'],
                                    info['current_epoch'])
            self.display.add_scalar('data/f1', info['f1'],
                                    info['current_epoch'])
            self.display.add_scalar('data/real_f1', info['real_f1'],
                                    info['current_epoch'])
            self.display.add_scalar('data/roc_auc', info['roc_auc'],
                                    info['current_epoch'])
            self.display.add_scalar('data/aupr', info['aupr'],
                                    info['current_epoch'])




    def plot_model_wts(self, model, epoch):
        if self.display:
            for name, param in model.named_parameters():
                self.display.add_histogram(name,
                                           param.clone().cpu().data.numpy(),
                                           epoch)



    def save_args(self):
        with open(os.path.join(self.args.out_dir, 'args.txt'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        f = open(os.path.join(self.args.out_dir, 'args.log'), 'w')
        for k, v in sorted(vars(self.args).items()):
            f.write('%s: %s\n' % (str(k), str(v)))
        f.close()

    def load_args(self, fp):
        with open(fp, 'r') as f:
            self.args.__dict__ = json.load(f)



    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))

            #Delete previous checkpoint
            #Only if epoch > 2
        if epoch >=3:
            os.remove(os.path.join(self.args.checkpoints_dir,
                                 'checkpoint_{:03d}.pt'.format(epoch - 1 )))

    def save_best_model(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'best_model.pt'.format(epoch)))



    def close(self):
        if self.display is not None:
            self.display.close()
