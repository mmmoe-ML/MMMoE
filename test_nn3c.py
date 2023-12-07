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

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def select_nn(log_file):
    train_log = np.loadtxt(log_file, delimiter=',', dtype='str')
    valid_loss = train_log[1:,3].astype(np.float32)
    loss_list = []
    total_epoch = valid_loss.shape[0]
    for i in range(int(total_epoch)):
        tmp = valid_loss[i]
        if i < 10:
            loss_list.append(1000000)
        else:
            loss_list.append(tmp)
    return loss_list.index(min(loss_list))

import random

def test(model, device, test_loader, output_file):
    model.eval() # testing

    for (slideID, class_label, fcm_tensor) in test_loader:
        input_tensor = fcm_tensor.to(device)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat = model(input_tensor[bag_num])

            class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
            class_softmax = class_softmax.tolist() #

            # write cassification results
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            slideid_tlabel_plabel = [slideID[bag_num], int(class_label[bag_num]), class_hat] + class_softmax # [slideID, label, prediction] + [y_prob]
            f_writer.writerow(slideid_tlabel_plabel)
            f.close()

def test_model(cv_test, m_dim, wd):

    ##################setting#######################################
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

    # superclass
    list_subtype = [['DLBCL','FL'],['AITL','ATLL'],['CHL','RL']]
    list_use = ['DLBCL','FL','AITL','ATLL','CHL','Reactive']

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

    data_all = np.genfromtxt('./FCM.csv', delimiter=',', filling_values=np.nan, dtype='str')
    data_all = data_all[1:,:]
    list_id = data_all[:,0].tolist() # slideID

    # create dataset
    i = 0
    test_dataset = []
    for slideID in test_all:
        class_label = test_label[i]
        test_dataset.append([slideID, class_label, data_all[list_id.index(slideID),2:20].astype(np.float32)])
        i += 1

    log = f'train_log/nn3c_log_cv-{cv_test}.csv'
    epoch_m = select_nn(log)
    makedir('test_result')
    result = f'test_result/nn3c_test_cv-{cv_test}.csv'
    makedir('model_params')
    model_params = f'./model_params/nn3c_cv-{cv_test}_epoch-{epoch_m}.pth'

    f = open(result, 'w')
    f.close()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create model
    from model import MLP_fcm
    #
    model = MLP_fcm(i_dim=18, m_dim=m_dim,n_class=n_class)
    model.load_state_dict(torch.load(model_params,map_location='cpu'))
    model = model.to(device)

    # preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    data_test = Dataset.FCMDataset(
        train=False,
        dataset=test_dataset,
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    test(model, device, test_loader, result)

if __name__ == '__main__':

    args = sys.argv
    cv_test = int(args[1]) # fold for test
    m_dim = int(args[2]) #
    wd = float(args[3]) # weight decay

    test_model(cv_test, m_dim, wd)
