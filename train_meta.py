"""
Realistic Setting for Few-shot Classification
Meta-Learing
"""

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
from eval.meta_eval import meta_test, normalize
from eval.cls_eval import validate

# addition
from datetime import datetime

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
    parser.add_argument('--n_aug_support_samples', default=1, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    # gpu setting
    parser.add_argument('-g', '--gpu_id', type=str, default='1', metavar='N', help='Use specific gpu number. default=\'1\' ')

    # realistic seeting
    parser.add_argument('--realistic_prob', type=float, default=0.0, help='The probability of sampling for realistic setting (default : 0.0)')

    # meta setting
    parser.add_argument('--use_logit', action='store_true', help='using logit')
    parser.add_argument('--is_norm', action='store_true', help='using normalization when get a prototype feature of each class')

    # opt = parser.parse_args("""--model_path checkpoints --tb_path tb_results --data_root Dataset --save_freq 1 --learning_rate 0.1
    #                            --model convnet4 --trial debug --gpu_id 1 --realistic_prob 0.1 --n_ways 5 --n_shots 5
    #                         """.split())
    ### jupyter
    # opt, _ = parser.parse_known_args("""--model_path checkpoints --tb_path tb_results --data_root Dataset --save_freq 1 --learning_rate 0.1
    #                                     --model resnet12 --trial debug --gpu_id 1 --realistic_prob 0.5
    #                                  """.split())
    opt = parser.parse_args()

    # gpu setting
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

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
    
    opt.model_name = 'Meta_{0.n_shots}shot_{0.n_ways}way_{0.model}_{0.dataset}' \
                     '_lr_{0.learning_rate}_decay_{0.weight_decay}_trans_{0.transform}'.format(opt)
    
    if opt.realistic_prob:
        opt.model_name = 'Realistic_Prob{0.realistic_prob}_{0.model_name}'.format(opt)

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
    print('#'*100)
    print('{:#^100}'.format(' Meta Learning '))
    if opt.realistic_prob:
        print('{:#^100}'.format(' Realistic Setting '))
    print('#'*100)
    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_trainloader = DataLoader(MetaImageNet(args=opt, partition=train_partition, train_transform=train_trans, pretrain=True,
                                                   realistic_prob=opt.realistic_prob),
                                  batch_size=opt.test_batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=test_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=test_trans,
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

    # model
    model = create_model(opt.model, n_cls, opt.dataset)

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
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_log = meta_train(epoch, meta_trainloader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        for i, name in enumerate(['train_acc', 'train_loss']):
            logger.log_value(name, train_log[i], epoch)

        val_log = meta_validate(epoch, meta_valloader, model, criterion, opt)

        for i, name in enumerate(['test_acc', 'test_loss']):
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

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)
    with open(opt.logfile, 'a') as f:
        print(f'End time : {datetime.now():%Y-%m-%d} {datetime.now():%H:%M:%S}', file=f)


def meta_train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc', ':.3f')

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1],
                             prefix=f'Epoch: [{epoch}]')

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        # ===================data=====================
        support_xs, _, query_xs, query_ys = data
        if torch.cuda.is_available():
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            query_ys = query_ys.cuda()
        batch_size, _, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)    # (25*3*84*84)
        query_xs = query_xs.view(-1, channel, height, width)        # (75*3*84*84)

        if opt.use_logit:
            support_features = model(support_xs).view(support_xs.size(0), -1)
            query_features = model(query_xs).view(query_xs.size(0), -1)
        else:
            feat_support, _ = model(support_xs, is_feat=True)
            support_features = feat_support[-1].view(support_xs.size(0), -1)    # (25*64)
            feat_query, _ = model(query_xs, is_feat=True)
            query_features = feat_query[-1].view(query_xs.size(0), -1)          # (75*64)

        if opt.is_norm:
            support_features = normalize(support_features)
            query_features = normalize(query_features)

        target = query_ys.view(-1)        # (75)

        # ===================forward=====================

        emb_s = support_features.view(opt.n_ways, opt.n_shots, support_features.size(1)).mean(1)    # (5*64)
        emb_s = torch.unsqueeze(emb_s, 0)           # (1, 5, 64)  (1, Nc, dim)
        emb_q = torch.unsqueeze(query_features, 1)  # (75, 1, 64) (Nq, 1, dim)

        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)
        loss = criterion(-dist, target)
        acc = accuracy(-dist, target, topk=(1,))

        losses.update(loss.item(), target.size(0))
        top1.update(acc[0].squeeze(), target.size(0))

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

    print(f' * Acc {top1.avg:.3f}')
    
    return top1.avg, losses.avg

@ torch.no_grad()
def meta_validate(epoch, val_loader, model, criterion, opt):
    """One epoch training"""
    model.eval()

    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc', ':.3f')

    progress = ProgressMeter(len(val_loader),
                             [batch_time, data_time, losses, top1],
                             prefix=f'Test: [{epoch}]')

    end = time.time()
    for idx, data in enumerate(val_loader):
        data_time.update(time.time() - end)

        # ===================data=====================
        support_xs, _, query_xs, query_ys = data
        if torch.cuda.is_available():
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            query_ys = query_ys.cuda()
        batch_size, _, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)    # (25*3*84*84)
        query_xs = query_xs.view(-1, channel, height, width)        # (75*3*84*84)

        if opt.use_logit:
            support_features = model(support_xs).view(support_xs.size(0), -1)
            query_features = model(query_xs).view(query_xs.size(0), -1)
        else:
            feat_support, _ = model(support_xs, is_feat=True)
            support_features = feat_support[-1].view(support_xs.size(0), -1)    # (25*64)
            feat_query, _ = model(query_xs, is_feat=True)
            query_features = feat_query[-1].view(query_xs.size(0), -1)          # (75*64)

        if opt.is_norm:
            support_features = normalize(support_features)
            query_features = normalize(query_features)

        target = query_ys.view(-1)        # (75)

        # ===================forward=====================

        emb_s = support_features.view(opt.n_ways, opt.n_shots, support_features.size(1)).mean(1)    # (5*64)
        emb_s = torch.unsqueeze(emb_s, 0)           # (1, 5, 64)  (1, Nc, dim)
        emb_q = torch.unsqueeze(query_features, 1)  # (75, 1, 64) (Nq, 1, dim)

        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)
        loss = criterion(-dist, target)
        acc = accuracy(-dist, target, topk=(1,))

        losses.update(loss.item(), target.size(0))
        top1.update(acc[0].squeeze(), target.size(0))

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            msg = progress.display(idx)

    print(f' * Acc {top1.avg:.3f}')
    return top1.avg, losses.avg

if __name__ == '__main__':
    main()