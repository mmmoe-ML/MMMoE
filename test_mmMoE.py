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

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def select_epoch(log_file):
    train_log = np.loadtxt(log_file, delimiter=',', dtype='str')
    valid_loss = train_log[1:,3].astype(np.float32)
    loss_list = []
    total_epoch = valid_loss.shape[0]/NUM_GPU
    for i in range(int(total_epoch)):
        tmp = valid_loss[i*NUM_GPU:(i+1)*NUM_GPU]
        if i < 2:
            loss_list.append(1000000)
        else:
            loss_list.append(np.sum(tmp))
    return loss_list.index(min(loss_list))

import random

def test(model, device, test_loader, output_file):
    model.eval() # testing

    for (input_tensor, slideID, class_label, fcm_tensor, pos_list) in test_loader:
        input_tensor = input_tensor.to(device)
        fcm_tensor = fcm_tensor.to(device)
        #
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    class_prob, class_hat, g_w, att = model(input_tensor[bag_num], fcm_tensor[bag_num])
                    class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
                    class_softmax = class_softmax.tolist() #

            # write classification results
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            slideid_tlabel_plabel = [slideID[bag_num], int(class_label[bag_num]), class_hat] + class_softmax# + g_w # [slideID, label, prediction] + [y_prob]
            f_writer.writerow(slideid_tlabel_plabel)
            pos_x = []
            pos_y = []
            for pos in pos_list:
                pos_x.append(int(pos[0]))
                pos_y.append(int(pos[1]))
            f_writer.writerow(pos_x) # x coordinate
            f_writer.writerow(pos_y) # y coordinate
            g_w = g_w.cpu()
            att = att.cpu().squeeze(1)
            g_w1 = []
            g_w2 = []
            g_w3 = []
            att1 = []
            for i in range(g_w.shape[0]):
                g_w1.append(float(g_w[i,0]))
                g_w2.append(float(g_w[i,1]))
                g_w3.append(float(g_w[i,2]))
                att1.append(float(att[i]))
            f_writer.writerow(g_w1)
            f_writer.writerow(g_w2)
            f_writer.writerow(g_w3)
            f_writer.writerow(att1)
            f.close()

def test_model(cv_test, t, params_fix):

    ##################setting#######################################
    EPOCHS = 10 # total epochs
    mag = '40x' # magnification: '40x' or '20x' or '10x' or '5x'
    device = 'cuda:0' # use single GPU
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

    n_class = len(num_data) # number of classes
    num_all = sum(num_data) # number of cases

    data_all = np.genfromtxt('./FCM.csv', delimiter=',', filling_values=np.nan, dtype='str')
    data_all = data_all[1:,:]
    list_id = data_all[:,0].tolist() # slideID

    i = 0
    test_dataset = []
    for slideID in test_all:
        class_label = test_label[i]
        test_dataset.append([slideID, class_label, data_all[list_id.index(slideID),2:20].astype(np.float32)])
        i += 1

    #
    if params_fix == 0:
        log = f'./train_log/mmMoE_log_cv-{cv_test}_t-{t}.csv'
        result = f'./test_result/mmMoE_test_cv-{cv_test}_t-{t}.csv'
    else:
        log = f'./train_log/mmMoE_log_cv-{cv_test}_t-{t}_fix.csv'
        result = f'./test_result/mmMoE_test_cv-{cv_test}_t-{t}_fix.csv'

    epoch_m = select_epoch(log)

    if params_fix == 0:
        model_params = f'./model_params/mmMoE_cv-{cv_test}_epoch-{epoch_m}_t-{t}.pth'
    else:
        model_params = f'./model_params/mmMoE_cv-{cv_test}_epoch-{epoch_m}_t-{t}_fix.pth'

    f = open(result, 'w')
    f.close()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create model
    from model import feature_extractor, MLP_attention, fc_sub, fc_label, MLP_mmgating, mmMoE, fc_FCM
    #
    feature_extractor = feature_extractor()
    MLP_attention = MLP_attention(512, 128)
    fc_subB = fc_sub()
    fc_subT = fc_sub()
    fc_subH = fc_sub()
    fc_label = fc_label(512, 128, n_class=n_class)
    fc_fcmMoE = fc_FCM(i_dim=18, m_dim=128)
    fc_gating = MLP_mmgating(fc_fcmMoE, 3)
    MoE = mmMoE(feature_extractor, fc_subB, fc_subT, fc_subH, fc_gating, MLP_attention, fc_label, n_class=n_class, t=t)
    MoE.load_state_dict(torch.load(model_params,map_location='cpu'))
    MoE = MoE.to(device)

    # preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    data_test = Dataset.mmDataset(
        train=False,
        transform=transform,
        dataset=test_dataset,
        mag=mag,
        bag_num=50,
        bag_size=100
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    test(MoE, device, test_loader, result)

if __name__ == '__main__':

    args = sys.argv
    cv_test = int(args[1]) # fold for test
    t = float(args[2]) # temparature parameter
    params_fix = int(args[3]) # fix sub-networks (=1) or not (=0)

    test_model(cv_test, t, params_fix)
