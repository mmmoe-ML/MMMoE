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

def train(model, device, loss_fn, optimizer, train_loader):
    model.train() # training
    train_class_loss = 0.0
    correct_num = 0
    for (slideID, class_label, fcm_tensor) in train_loader:
        #
        input_tensor = fcm_tensor.to(device)
        class_label = class_label.to(device)
        for bag_num in range(input_tensor.shape[0]):
            optimizer.zero_grad() # initialization
            class_prob, class_hat = model(input_tensor[bag_num])
            # calculate loss
            class_loss = loss_fn(class_prob, class_label[bag_num])
            train_class_loss += class_loss.item()

            class_loss.backward() # backpropagation
            optimizer.step() # update parameters
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num

def valid(model, device, loss_fn, test_loader):
    model.eval() # validating
    test_class_loss = 0.0
    correct_num = 0
    for (slideID, class_label, fcm_tensor) in test_loader:
        #
        input_tensor = fcm_tensor.to(device)
        class_label = class_label.to(device)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat = model(input_tensor[bag_num])
            # calculate loss
            class_loss = loss_fn(class_prob, class_label[bag_num])
            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return test_class_loss, correct_num

def train_model(cv_test, m_dim, wd):

    ##################setting#######################################
    EPOCHS = 50 # toral epochs
    device = 'cpu'
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

    list_subtype = [['DLBCL','FL'],['AITL','ATLL'],['CHL','RL']]
    list_use = ['DLBCL','FL','AITL','ATLL','CHL','RL']

    num_data = []
    max_main = 100

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

    n_class = len(num_data)
    num_all = sum(num_data)
    for i in range(len(num_data)):
        if num_data[i] == 0:
            num_data[i] = 0
        else:
            num_data[i] = num_all/(num_data[i]*n_class)
    weights = torch.tensor(num_data).to(device, non_blocking=True)

    data_all = np.genfromtxt('./FCM.csv', delimiter=',', filling_values=np.nan, dtype='str')
    data_all = data_all[1:,:]
    list_id = data_all[:,0].tolist() # slideID

    # create dataset
    i = 0
    train_dataset = []
    for slideID in train_all:
        class_label = train_label[i]
        train_dataset.append([slideID, class_label, data_all[list_id.index(slideID),2:20].astype(np.float32)])
        i += 1
    i = 0
    valid_dataset = []
    for slideID in valid_all:
        class_label = valid_label[i]
        valid_dataset.append([slideID, class_label, data_all[list_id.index(slideID),2:20].astype(np.float32)])
        i += 1

    # log file
    makedir('train_log')
    log = f'train_log/nn3c_log_cv-{cv_test}.csv'

    # write header
    f = open(log, 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "time"]
    f_writer.writerow(csv_header)
    f.close()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create model
    from model import MLP_fcm
    #
    model = MLP_fcm(i_dim=18, m_dim=m_dim,n_class=n_class)
    model = model.to(device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # specify optimizer
    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # start training
    for epoch in range(EPOCHS):

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start_t = time.time()

        data_train = Dataset.FCMDataset(
            train=True,
            dataset=train_dataset,
        )

        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
        )

        class_loss, acc = train(model, device, loss_fn, optimizer, train_loader)

        scheduler.step()

        train_loss += class_loss
        train_acc += acc

        data_valid = Dataset.FCMDataset(
            train=False,
            dataset=valid_dataset,
        )

        valid_loader = torch.utils.data.DataLoader(
            data_valid,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=2,
        )

        class_loss, acc = valid(model, device, loss_fn, valid_loader)

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
        if epoch > 0:
            makedir('model_params')
            model_params = f'./model_params/nn3c_cv-{cv_test}_epoch-{epoch}.pth'
            torch.save(model.state_dict(), model_params)

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    args = sys.argv
    cv_test = int(args[1]) # fold for test
    m_dim = int(args[2]) #
    wd = float(args[3]) # weight decay

    train_model(cv_test, m_dim, wd)
