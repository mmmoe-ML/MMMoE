import os
import glob
import random
from PIL import Image, ImageStat
import numpy as np
import cv2
import csv
import sys
import math
import openslide

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

args = sys.argv

DATA_PATH = './Data'

list_subtype = ['DLBCL','FL','AITL','ATLL','CHL','RL']
data_all = []
for subtype in list_subtype:
    tmp_id = np.loadtxt(f'./slideID/{subtype}.txt', delimiter=',', dtype='str').tolist()
    data_all = data_all + tmp_id

makedir('./Data/csv')

svs_list = os.listdir(f'{DATA_PATH}/svs')

b_size = 224 # size of image patch
t_size = 4 # size of an image patch in thumbnail

for slideID in data_all:

    svs_fn = [s for s in svs_list if slideID in s]
    svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')
    width,height = svs.dimensions
    b_w = width // b_size # number of image patches for x-axis
    b_h = height // b_size # number of image patches for y-axis

    thumb = Image.new('RGB',(b_w * t_size, b_h * t_size))   # thumbnail
    thumb_s = Image.new('L',(b_w, b_h)) # saturation

    ite = 1

    while ite > 0:

        for h in range(b_h):
            for w in range(b_w):
                #
                b_img = svs.read_region((w*b_size,h*b_size),0,(b_size,b_size)).convert('RGB')
                r_img = b_img.resize((t_size, t_size), Image.BILINEAR)  #

                if ite == 1:
                    thumb.paste(r_img, (w * t_size, h * t_size))

                b_array = np.array(b_img)

                #
                R_b, G_b, B_b = cv2.split(b_array)
                Max_b = np.maximum(np.maximum(R_b, G_b), B_b)
                Min_b = np.minimum(np.minimum(R_b, G_b), B_b)
                Sat_b = Max_b - Min_b
                img_g = b_img.convert('L')
                s_img = Image.fromarray(Sat_b)
                statS = ImageStat.Stat(s_img)
                statV = ImageStat.Stat(img_g)
                b_ratio = B_b / R_b
                if (statV.mean[0] < 255 * 0.95 or np.count_nonzero(G_b > 255 * 0.9) < 224*224 / 2) and np.mean(b_ratio) > 0.9 and np.count_nonzero(b_ratio > 1.1 - 0.05 * (ite - 1)) > 224*224 / (128 * (2 ** (ite - 1))):
                    thumb_s.putpixel((w,h), round(statS.mean[0]))
                else:
                    thumb_s.putpixel((w,h), 0)

        makedir(f'{DATA_PATH}/thumb')
        if ite == 1:
            thumb.save(f'{DATA_PATH}/thumb/{slideID}_thumb.tif')    #標本サムネイル保存

        s_array = np.asarray(thumb_s)   #
        ret, s_mask = cv2.threshold(s_array, 0, 255, cv2.THRESH_OTSU) # Otsu method

        num_i = np.count_nonzero(s_mask)
        if num_i < 100:
            ite += 1
        else:
            pos = np.zeros((num_i,2))
            i = 0
            for h in range(b_h):
                for w in range(b_w):
                    if not s_mask[h,w] == 0:
                        pos[i][0] = w * b_size
                        pos[i][1] = h * b_size
                        i = i + 1
            ite = -1

    np.savetxt(f'{DATA_PATH}/csv/{slideID}.csv', pos, delimiter=',', fmt='%d')
