# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import os
import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import sys

NUM_GPU = 8

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' #

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# correct: ans=1, incorrect: ans=0
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

import random

def train(model, rank, loss_fn, optimizer, train_loader):
    model.train() # training
    train_class_loss = 0.0
    correct_num = 0
    for (input_tensor, slideID, class_label) in train_loader:
        #
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            optimizer.zero_grad() # initialization
            class_prob, class_hat, A = model(input_tensor[bag_num])
            # calculate loss
            class_loss = loss_fn(class_prob, class_label[bag_num])
            train_class_loss += class_loss.item()

            class_loss.backward() # backpropagation
            optimizer.step() # update parameters
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num

def valid(model, rank, loss_fn, test_loader):
    model.eval() # validating
    test_class_loss = 0.0
    correct_num = 0
    for (input_tensor, slideID, class_label, pos) in test_loader:

        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num])
            # calculate loss
            class_loss = loss_fn(class_prob, class_label[bag_num])
            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return test_class_loss, correct_num

def train_model(rank, world_size, cv_test):
    setup(rank, world_size)

    ##################setting#######################################
    EPOCHS = 10 # total epochs
    mag = '40x' # magnification: '40x' or '20x' or '10x' or '5x'
    ################################################################

    # split data
    random.seed(0)
    num_cv = 5
    train_all = []
    valid_all = []
    test_all = []
    train_label = []
    valid_label = []
    test_label = []

    list_subtype = [['DLBCL'],['FL'],['AITL'],['ATLL'],['CHL'],['RL']]
    list_use = ['DLBCL','FL','AITL','ATLL','CHL','RL']

    num_data = []
    max_main = 100 # maximum number of each class

    # train:valid:test = 3:1:1
    label = 0
    for group_subtype in list_subtype:
        train_group = []
        for subtype in group_subtype:
            if subtype in list_use:
                max_sample = max_main
                list_id = np.loadtxt(f'./slideID/{subtype}.txt', delimiter=',', dtype='str').tolist()
                random.seed(0)
                random.shuffle(list_id)
                list_id = list_id[0:max_sample]
                num_e = len(list_id) // num_cv
                num_r = len(list_id) % num_cv
                tmp_all = []
                for cv in range(num_cv):
                    tmp = []
                    for i in range(num_e):
                        tmp.append(list_id.pop(0))
                    if cv < num_r:
                        tmp.append(list_id.pop(0))
                    tmp_all.append(tmp)
                train_tmp = tmp_all[cv_test%5] + tmp_all[(cv_test+1)%5] + tmp_all[(cv_test+2)%5]
                train_group += train_tmp
                train_all += train_tmp
                valid_all += tmp_all[(cv_test+3)%5]
                test_all += tmp_all[(cv_test+4)%5]
                train_tmp = [label] * len(train_tmp)
                valid_tmp = [label] * len(tmp_all[(cv_test+3)%5])
                test_tmp = [label] * len(tmp_all[(cv_test+4)%5])
                train_label += train_tmp
                valid_label += valid_tmp
                test_label += test_tmp
        num_data.append(len(train_group))
        label += 1

    n_class = len(num_data) # number of classes
    num_all = sum(num_data) # number of cases
    for i in range(len(num_data)):
        if num_data[i] == 0:
            num_data[i] = 0
        else:
            num_data[i] = num_all/(num_data[i]*n_class)
    weights = torch.tensor(num_data).to(rank, non_blocking=True)

    # create dataset
    i = 0
    train_dataset = []
    for slideID in train_all:
        class_label = train_label[i]
        train_dataset.append([slideID, class_label])
        i += 1
    i = 0
    valid_dataset = []
    for slideID in valid_all:
        class_label = valid_label[i]
        valid_dataset.append([slideID, class_label])
        i += 1

    # log file
    log = f'./train_log/MIL6c_log_cv-{cv_test}.csv'

    if rank == 0:
        # write header
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "time"]
        f_writer.writerow(csv_header)
        f.close()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create model
    from model import feature_extractor, MLP_attention, fc_label, MIL
    #
    feature_extractor = feature_extractor()
    MLP_attention = MLP_attention(2048, 512)
    fc_label = fc_label(2048, 512, n_class=n_class)
    #
    model = MIL(feature_extractor, MLP_attention, fc_label, n_class)
    model = model.to(rank)
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    #
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    #
    ddp_model = DDP(model, device_ids=[rank])

    # define loss function
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # specify optimizer
    lr = 0.001
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # start training
    for epoch in range(EPOCHS):

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start_t = time.time()

        data_train = Dataset.imgDataset(
            train=True,
            transform=transform,
            dataset=train_dataset,
            mag=mag,
            bag_num=50,
            bag_size=100,
            epoch=epoch
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, rank=rank)

        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
            sampler=train_sampler
        )

        class_loss, acc = train(ddp_model, rank, loss_fn, optimizer, train_loader)

        scheduler.step()

        train_loss += class_loss
        train_acc += acc

        data_valid = Dataset.imgDataset(
            train=False,
            transform=transform,
            dataset=valid_dataset,
            mag=mag,
            bag_num=50,
            bag_size=100,
            epoch=epoch
        )

        valid_sampler = torch.utils.data.distributed.DistributedSampler(data_valid, rank=rank)

        valid_loader = torch.utils.data.DataLoader(
            data_valid,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=2,
            sampler=valid_sampler
        )

        class_loss, acc = valid(ddp_model, rank, loss_fn, valid_loader)

        valid_loss += class_loss
        valid_acc += acc

        train_loss /= float(len(train_loader.dataset))
        train_acc /= float(len(train_loader.dataset))
        valid_loss /= float(len(valid_loader.dataset))
        valid_acc /= float(len(valid_loader.dataset))
        elapsed_t = time.time() - start_t

        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc, elapsed_t])
        f.close()
        # save model parameters
        if rank == 0:
            model_params = f'./model_params/MIL6c_cv-{cv_test}_epoch-{epoch}.pth'
            torch.save(ddp_model.module.state_dict(), model_params)

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    num_gpu = NUM_GPU # number of GPU

    args = sys.argv
    cv_test = int(args[1]) # fold for test

    #args : argments of train_model
    #nprocs : number of GPU
    mp.spawn(train_model, args=(num_gpu, cv_test), nprocs=num_gpu, join=True)
