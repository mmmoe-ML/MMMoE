# Description of code

## train_MIL6c.py

Code to learn 6-class MIL classification model to obtain common feature extractor.\

## train_MIL2B.py, train_MIL2T.py, train_MIL2H.py

Code to learn 2-class MIL classification model to obtain initial sub-networks. \

## train_mmMoE.py

Code to learn the propsoed multimodal MoE model. 

Following codes provides the training of the proposed model for fold #1 in 5-fold cross-validation.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL6c.py 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2B.py 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2T.py 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2H.py 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_mmMoE.py 1 0.8 0

## test_mmMoE.py

Code to evaluate the model.
Following codes provides the evaluation results of test datal for fold #1 in 5-fold cross-validation.

CUDA_VISIBLE_DEVICES=0 python test_mmMoE.py 1 0.8 0

## model.py

Code describing the models.

## Dataset.py

Code to create dataset (bags).

## Required Data

./Data : Whole slide images (SVS) \
./csv/[DLBCL,FL,AITL,ATLL,CHL,RL].csv : files written coordinate of each patch\
./slideID/[DLBCL,FL,AITL,ATLL,CHL,RL].txt : files written slideID of each subtype\
./FCM.csv : FCM data
