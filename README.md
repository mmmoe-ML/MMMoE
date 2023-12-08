# Description of code

In our source codes, training are parallelly processed using 8 GPUs, and testing are processed using a single GPU.

## train_MIL6c.py

Code to learn 6-class MIL classification model to obtain common feature extractor. \
Following command provides the training of the model for the 1st fold in 5-fold cross-validation.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL6c.py 1

## train_MIL2B.py, train_MIL2T.py, train_MIL2H.py

Code to learn 2-class MIL classification model to obtain initial sub-networks. \
train_MIL2B.py, train_MIL2T.py, train_MIL2H.py are the model for B-cell, T-cell, Others, respectively.\
Following command provides the training of the B-cell model for the 1st fold in 5-fold cross-validation.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2B.py 1

## train_mmMoE.py

Code to learn the propsoed multimodal MoE model. 

Following command provides the training of the proposed model for 1st fold in 5-fold cross-validation, where the temperature parameter is set to 0.8 and the sub-networks will be re-trained.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_mmMoE.py 1 0.8 0


Following commands provide the sequential training of the proposed model including pre-training feature extractor and sub-networks for 1st fold in 5-fold cross-validation

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL6c.py 1\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2B.py 1\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2T.py 1\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_MIL2H.py 1\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_mmMoE.py 1 0.8 0

## test_mmMoE.py

Code to evaluate the model.\
Following code provides the evaluation results of test data for 1st fold in 5-fold cross-validation.

CUDA_VISIBLE_DEVICES=0 python test_mmMoE.py 1 0.8 0

## train_\*.py, test_\*.py

Code to train and evaluate other models in the literature.

## att_mmMoE.py

Code to compute gating weights and attention weights in the proposed method.\
Following command provide caluclated each weight using the trained model.

CUDA_VISIBLE_DEVICES=0 python att_mmMoE.py 1 0.8 0

## vis_MoE.py

Code to visualize calculated gating and attention weights.\
Following command provide visualizaion results.

CUDA_VISIBLE_DEVICES=0 python vis_mmMoE.py mmMoE 0.8

## eval_MoE.py, eval_oracle.py, eval_hie.py

Code to compute evaluation measures for MoE-architecture method, method 1, and method 3, respectively.

## export_csv.py

Code to write cordinates of the tissue regions in WSIs as the pre-processing.

## model.py

Code describing the models.

## Dataset.py

Code to create dataset (bags).

## Required Data

./Data/svs/ : Whole slide images (SVS) \
./Data/csv/[DLBCL,FL,AITL,ATLL,CHL,RL].csv : files written coordinate of each patch\
./slideID/[DLBCL,FL,AITL,ATLL,CHL,RL].txt : files written slideID of each subtype\
./FCM.csv : FCM data
