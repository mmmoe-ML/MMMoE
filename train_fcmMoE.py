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

def train(model, rank, loss_fn, optimizer, scaler, train_loader):
    model.train() # training
    train_class_loss = 0.0
    correct_num = 0

    for (input_tensor, slideID, class_label, fcm_tensor) in train_loader:
        #
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            optimizer.zero_grad() # initialization

            with torch.cuda.amp.autocast(enabled=True):
                class_prob, class_hat, g_w, att = model(input_tensor[bag_num], fcm_tensor[bag_num])
                # calculate loss
                class_loss = loss_fn(class_prob, class_label[bag_num])
                train_class_loss += class_loss.item()

            scaler.scale(class_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num

def valid(model, rank, loss_fn, test_loader):
    model.eval() # validating
    test_class_loss = 0.0
    correct_num = 0

    for (input_tensor, slideID, class_label, fcm_tensor, pos) in test_loader:
        #
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    class_prob, class_hat, g_w, att = model(input_tensor[bag_num], fcm_tensor[bag_num])
                    # calculate loss
                    class_loss = loss_fn(class_prob, class_label[bag_num])

            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return test_class_loss, correct_num

def train_model(rank, world_size, cv_test, t, params_fix):
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
    for i in range(len(num_data)):
        if num_data[i] == 0:
            num_data[i] = 0
        else:
            num_data[i] = num_all/(num_data[i]*n_class)
    weights = torch.tensor(num_data).to(rank, non_blocking=True)

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logB = f'./train_log/MIL2B_log_cv-{cv_test}.csv'
    epoch_m = select_epoch(logB)
    model_paramsB = f'./model_params/MIL2B_cv-{cv_test}_epoch-{epoch_m}.pth'
    logT = f'./train_log/MIL2T_log_cv-{cv_test}.csv'
    epoch_m = select_epoch(logT)
    model_paramsT = f'./model_params/MIL2T_cv-{cv_test}_epoch-{epoch_m}.pth'
    logH = f'./train_log/MIL2H_log_cv-{cv_test}.csv'
    epoch_m = select_epoch(logH)
    model_paramsH = f'./model_params/MIL2H_cv-{cv_test}_epoch-{epoch_m}.pth'

    # create model
    from model import feature_extractor, MLP_attention, fc_label, fc_sub, MLP_fcmgating, MIL2, fcmMoE
    #
    feature_extractorB = feature_extractor()
    MLP_attentionB = MLP_attention(512, 128)
    fc_subB = fc_sub()
    fc_labelB = fc_label(512, 128, n_class=2)
    #
    MIL2B = MIL2(feature_extractorB, MLP_attentionB, fc_subB, fc_labelB, 2)
    MIL2B.load_state_dict(torch.load(model_paramsB,map_location='cpu'))
    feature_extractorMoE = MIL2B.feature_ex # use pre-trained feature extractor
    fc_subMoEB = MIL2B.fc_sub # use pre-trained sub-network for B-cell

    feature_extractorT = feature_extractor()
    MLP_attentionT = MLP_attention(512, 128)
    fc_subT = fc_sub()
    fc_labelT = fc_label(512, 128, n_class=2)
    #
    MIL2T = MIL2(feature_extractorT, MLP_attentionT, fc_subT, fc_labelT, 2)
    MIL2T.load_state_dict(torch.load(model_paramsT,map_location='cpu'))
    fc_subMoET = MIL2T.fc_sub # use pre-trained sub-network for T-cell

    feature_extractorH = feature_extractor()
    MLP_attentionH = MLP_attention(512, 128)
    fc_subH = fc_sub()
    fc_labelH = fc_label(512, 128, n_class=2)
    #
    MIL2H = MIL2(feature_extractorH, MLP_attentionH, fc_subH, fc_labelH, 2)
    MIL2H.load_state_dict(torch.load(model_paramsH,map_location='cpu'))
    fc_subMoEH = MIL2H.fc_sub # use pre-trained sub-network for Others

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    MLP_gatingMoE = MLP_fcmgating (3)

    MLP_attentionMoE = MLP_attention(512, 128) #
    fc_labelMoE = fc_label(512, 128, n_class=n_class) #
    MoE = fcmMoE(feature_extractorMoE, fc_subMoEB, fc_subMoET, fc_subMoEH, MLP_gatingMoE, MLP_attentionMoE, fc_labelMoE, n_class=n_class, t=t)

    MoE = MoE.to(rank)
    process_group = torch.distributed.new_group([i for i in range(world_size)])

    for param in MoE.feature_ex.parameters():
        param.requires_grad = False
    if params_fix == 1:
        for param in MoE.expert1.parameters():
            param.requires_grad = False
        for param in MoE.expert2.parameters():
            param.requires_grad = False
        for param in MoE.expert3.parameters():
            param.requires_grad = False

    #
    MoE = nn.SyncBatchNorm.convert_sync_batchnorm(MoE, process_group)
    #
    ddp_model = DDP(MoE, device_ids=[rank])

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

    use_amp = True # on/off amp

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # log file
    if params_fix == 0:
        log = f'./train_log/fcmMoE_log_cv-{cv_test}_t-{t}.csv'
    else:
        log = f'./train_log/fcmMoE_log_cv-{cv_test}_t-{t}_fix.csv'

    if rank == 0:
        # write header
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "time"]
        f_writer.writerow(csv_header)
        f.close()

    # start training
    for epoch in range(EPOCHS):

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start_t = time.time()

        data_train = Dataset.mmDataset(
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

        class_loss, acc = train(ddp_model, rank, loss_fn, optimizer, scaler, train_loader)

        scheduler.step()

        train_loss += class_loss
        train_acc += acc

        data_valid = Dataset.mmDataset(
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
            if params_fix == 0:
                model_params = f'./model_params/fcmMoE_cv-{cv_test}_epoch-{epoch}_t-{t}.pth'
            else:
                model_params = f'./model_params/fcmMoE_cv-{cv_test}_epoch-{epoch}_t-{t}_fix.pth'
            torch.save(ddp_model.module.state_dict(), model_params)

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    num_gpu = NUM_GPU # number of GPU

    args = sys.argv
    cv_test = int(args[1]) # fold for test
    t = float(args[2]) # temparature parameter
    params_fix = int(args[3]) # fix sub-networks (=1) or not (=0)

    #args : argments of train_model
    #nprocs : number of GPU
    mp.spawn(train_model, args=(num_gpu, cv_test, t, params_fix), nprocs=num_gpu, join=True)
