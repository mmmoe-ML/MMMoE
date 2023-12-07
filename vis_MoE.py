import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import random

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def str_to_float(l):
    return float(l)

def draw_heatmap(slideID, att_dir, save_dir):
    b_size = 224 # size of image patch
    t_size = 4 # size of an image patch in thumbnail
    img_g1 = cv2.imread(f'./Data/thumb/{slideID}_thumb.tif')
    img_g2 = cv2.imread(f'./Data/thumb/{slideID}_thumb.tif')
    img_g3 = cv2.imread(f'./Data/thumb/{slideID}_thumb.tif')
    img_a1 = cv2.imread(f'./Data/thumb/{slideID}_thumb.tif')
    img_a2 = cv2.imread(f'./Data/thumb/{slideID}_thumb.tif')
    img_a3 = cv2.imread(f'./Data/thumb/{slideID}_thumb.tif')

    w, h = img_g1.shape[1], img_g1.shape[0]
    w_num = w // t_size
    h_num = h // t_size

    cv2.imwrite(f'{save_dir}/{slideID}.tif', img_g1)

    att_file = f'{att_dir}/{slideID}.csv'
    att_data = np.loadtxt(att_file, delimiter=',')

    att = []
    pos_x = []
    pos_y = []
    g1_list = att_data[:,2].astype(np.float32).tolist()
    g2_list = att_data[:,3].astype(np.float32).tolist()
    g3_list = att_data[:,4].astype(np.float32).tolist()
    att1_list = att_data[:,5].astype(np.float32).tolist()
    att2_list = att_data[:,5].astype(np.float32).tolist()
    att3_list = att_data[:,5].astype(np.float32).tolist()
    att1_max = max(att1_list)
    att1_min = min(att1_list)
    att2_max = max(att2_list)
    att2_min = min(att2_list)
    att3_max = max(att3_list)
    att3_min = min(att3_list)
    for j in range (len(att1_list)):
        att1_list[j] = g1_list[j] * (att1_list[j] - att1_min) / (att1_max - att1_min) # normalize attention
        att2_list[j] = g2_list[j] * (att2_list[j] - att2_min) / (att2_max - att2_min) #
        att3_list[j] = g3_list[j] * (att3_list[j] - att3_min) / (att3_max - att3_min) #
    pos_x = att_data[:,0].astype(np.int).tolist()
    pos_y = att_data[:,1].astype(np.int).tolist()

    cmap = plt.get_cmap('jet')

    for i in range (len(att1_list)):
        cval = cmap(float(g1_list[i]))
        cv2.rectangle(img_g1, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        cval = cmap(float(g2_list[i]))
        cv2.rectangle(img_g2, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        cval = cmap(float(g3_list[i]))
        cv2.rectangle(img_g3, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        cval = cmap(float(att1_list[i]))
        cv2.rectangle(img_a1, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        cval = cmap(float(att2_list[i]))
        cv2.rectangle(img_a2, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        cval = cmap(float(att3_list[i]))
        cv2.rectangle(img_a3, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)

    cv2.imwrite(f'{save_dir}/{slideID}_g1.tif', img_g1)
    cv2.imwrite(f'{save_dir}/{slideID}_g2.tif', img_g2)
    cv2.imwrite(f'{save_dir}/{slideID}_g3.tif', img_g3)
    cv2.imwrite(f'{save_dir}/{slideID}_g1a.tif', img_a1)
    cv2.imwrite(f'{save_dir}/{slideID}_g2a.tif', img_a2)
    cv2.imwrite(f'{save_dir}/{slideID}_g3a.tif', img_a3)

if __name__ == "__main__":
    args = sys.argv
    model = args[1]
    t = float(args[2])

    att_dir = f'./{model}_att_t-{t}_fix'
    save_dir = f'./{model}_vis_t-{t}_fix'
    makedir(save_dir)

    list_subtype = ['DLBCL','FL','AITL','ATLL','CHL','RL']

    max_sample = 100

    data_all = []
    random.seed(0)

    for subtype in list_subtype:
        list_id = np.loadtxt(f'./slideID/{subtype}.txt', delimiter=',', dtype='str').tolist()
        random.seed(0)
        random.shuffle(list_id)
        list_id = list_id[0:max_sample]
        data_all += list_id

    for slideID in data_all:
        draw_heatmap(slideID, att_dir, save_dir)
