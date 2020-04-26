from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter, ProgressMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate

# addition
from datetime import datetime
from models.moco_builder import MoCo, resnet12ForMoco

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='checkpoints', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='tb_results', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='Dataset', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    # gpu setting
    parser.add_argument('-g', '--use_gpu', type=str, default='1', metavar='N', help='Use specific gpu number. default=\'1\' ')
    
    ### moco setting
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--lamda', default=0.5, type=float,
                        help='Lambda value between classification loss and contrastive loss. '
                             'Lambda*cls_loss + (1-Lambda)*contr_loss '
                             '(default: 0.5)')

    # opt = parser.parse_args("""--save_freq 1 --learning_rate 0.1
    #                            --model resnet12 --trial debug --use_gpu 1 --dist-url 77 
    #                            --lr_decay_epochs 60,80,120,160 --epochs 200
    #                            --mlp --moco-dim 128 --moco-k 65536 --lamda 0.5
    #                         """.split())
    opt = parser.parse_args()

    # gpu setting
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.use_gpu

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = ('moco_{0.model}_{0.dataset}_lr_{0.learning_rate}_decay_{0.weight_decay}_trans_{0.transform}'.format(opt) + 
                     '_dim{0.moco_dim}_K{0.moco_k}_lam{0.lamda}'.format(opt))
    if opt.mlp:
        opt.model_name = '{}_mlp'.format(opt.model_name)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)
    if opt.data_aug:
        opt.model_name = '{}_dataAug'.format(opt.model_name)
    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()
    if opt.num_workers < 0:
        opt.num_workers = opt.n_gpu * 2
    opt.logfile = os.path.join(opt.save_folder, 'log.txt')
    print(opt)
    with open(opt.logfile, 'w') as f:
        for key in opt.__dict__.keys():
            print(f'{key} : {opt.__dict__[key]}', file=f)
        print(f'Start time : {datetime.now():%Y-%m-%d} {datetime.now():%H:%M:%S}', file=f)
    return opt


def main():
    opt = parse_option()
    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans, is_moco=True),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans, is_moco=True),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    ## moco setting
    if 'localhost' not in opt.dist_url:
       opt.dist_url = f'tcp://localhost:1{int(opt.dist_url):04d}' 
    torch.distributed.init_process_group(backend='nccl', init_method=opt.dist_url, 
                                         world_size=1, rank=0)
    # define model                                         
    model = MoCo(resnet12ForMoco, n_cls,
                 opt.moco_dim, opt.moco_k, opt.moco_m, opt.moco_t, opt.mlp)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
            model.lamda = model.module.lamda
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    best_acc = 0.0
    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_log = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        for i, name in enumerate(['train_acc', 'train_contra_acc', 'train_total_loss', 'train_loss', 'train_contra_loss']):
            logger.log_value(name, train_log[i], epoch)

        val_log = validate_moco(val_loader, model, criterion, opt)

        for i, name in enumerate(['test_acc', 'test_acc_top5', 'test_contra_accTop1', 'test_contra_accTop5',
                                  'test_total_loss', 'test_loss', 'test_contra_loss']):
            logger.log_value(name, val_log[i], epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, save_file)
        if best_acc < val_log[0]:
            print('==> Best Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_best.pth')
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)
    with open(opt.logfile, 'a') as f:
        print(f'End time : {datetime.now():%Y-%m-%d} {datetime.now():%H:%M:%S}', file=f)


def train(epoch, train_loader, model, criterion, optimizer, opt):
    """
    One epoch training
    Args:
    Return:
        top1 classification accuracy
        top1 contrastive accuracy
        total loss
        classification loss
        contrastive loss
    """
    model.train()

    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Total Loss', ':.4f')
    losses_cls = AverageMeter('Class Loss', ':.4f')
    losses_contr = AverageMeter('Contr Loss', ':.4f')
    top1_cls = AverageMeter('Acc@1', ':.3f')
    top5_cls = AverageMeter('Acc@5', ':.3f')
    top1_contr = AverageMeter('Acc@1', ':.3f')
    top5_contr = AverageMeter('Acc@5', ':.3f')

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, losses_cls, top1_cls, top5_cls, 
                              losses_contr, top1_contr, top5_contr,],
                             prefix=f'Epoch: [{epoch}]')

    end = time.time()
    for idx, (input_q, input_k, target_cls, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_q = input_q.float()
        input_k = input_k.float()
        if torch.cuda.is_available():
            input_q = input_q.cuda()
            input_k = input_k.cuda()
            target_cls = target_cls.cuda()

        # ===================forward=====================
        # classification forward
        logits_cls = model(input_q)
        # contrastive forward
        logits_contr, target_contr = model(input_q, input_k)
        # classification loss and contrastive loss
        loss_cls = criterion(logits_cls, target_cls)
        loss_contr = criterion(logits_contr, target_contr)
        loss = model.lamda*loss_cls + (1-model.lamda)*loss_contr

        acc_cls = accuracy(logits_cls, target_cls, topk=(1, 5))
        acc_contr = accuracy(logits_contr, target_contr, topk=(1, 5))
        
        losses_cls.update(loss_cls.item(), input_q.size(0))
        losses_contr.update(loss_contr.item(), input_q.size(0))
        losses.update(losses_cls.val + losses_contr.val, input_q.size(0))
        top1_cls.update(acc_cls[0][0], input_q.size(0))
        top5_cls.update(acc_cls[1][0], input_q.size(0))
        top1_contr.update(acc_contr[0][0], input_q.size(0))
        top5_contr.update(acc_contr[1][0], input_q.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            msg = progress.display(idx)
        
    print(f' * Acc@1 {top1_cls.avg:.3f} Acc@5 {top5_cls.avg:.3f}')

    return top1_cls.avg, top1_contr.avg, losses.avg, losses_cls.avg, losses_contr.avg

@ torch.no_grad()
def validate_moco(val_loader, model, criterion, opt):
    """One epoch validation"""
    batch_time = AverageMeter('Time', ':.3f')
    losses = AverageMeter('Total Loss', ':.4f')
    losses_cls = AverageMeter('Class Loss', ':.4f')
    losses_contr = AverageMeter('Contr Loss', ':.4f')
    top1_cls = AverageMeter('Acc@1', ':.3f')
    top5_cls = AverageMeter('Acc@5', ':.3f')
    top1_contr = AverageMeter('Acc@1', ':.3f')
    top5_contr = AverageMeter('Acc@5', ':.3f')
    
    progress = ProgressMeter(len(val_loader),
                             [batch_time, losses, losses_cls, top1_cls, top5_cls, 
                              losses_contr, top1_contr, top5_contr,],
                             prefix='Val ')
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for idx, (input_q, input_k, target_cls, _) in enumerate(val_loader):
        input_q = input_q.float()
        input_k = input_k.float()
        if torch.cuda.is_available():
            input_q = input_q.cuda()
            input_k = input_k.cuda()
            target_cls = target_cls.cuda()

        # ===================forward=====================
        # classification forward
        logits_cls = model(input_q)
        # contrastive forward
        logits_contr, target_contr = model(input_q, input_k, is_DeEnqueue=False)
        # classification loss and contrastive loss
        loss_cls = criterion(logits_cls, target_cls)
        loss_contr = criterion(logits_contr, target_contr)

        acc_cls = accuracy(logits_cls, target_cls, topk=(1, 5))
        acc_contr = accuracy(logits_contr, target_contr, topk=(1, 5))
        
        losses_cls.update(loss_cls.item(), input_q.size(0))
        losses_contr.update(loss_contr.item(), input_q.size(0))
        losses.update(losses_cls.val + losses_contr.val, input_q.size(0))
        top1_cls.update(acc_cls[0][0], input_q.size(0))
        top5_cls.update(acc_cls[1][0], input_q.size(0))
        top1_contr.update(acc_contr[0][0], input_q.size(0))
        top5_contr.update(acc_contr[1][0], input_q.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            progress.display(idx)

    print(f' * Acc@1 {top1_cls.avg:.3f} Acc@5 {top5_cls.avg:.3f}')

    return top1_cls.avg, top5_cls.avg, top1_contr.avg, top5_contr.avg, losses.avg, losses_cls.avg, losses_contr.avg

if __name__ == '__main__':
    main()
