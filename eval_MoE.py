# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import sys

# スライド単位の事後確率とラベルのリストを返す
def get_slide_prob_label(csv_file, n_class):
    pred_corpus = {}
    label_corpus = {}
    slide_id_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row)==(n_class+3)):
                slide_id = row[0]
                prob_list = []
                for i in range(n_class):
                    prob_list.append(float(row[i+3])) # [DLBCLの確率, FLの確率, RLの確率]
                if(slide_id not in pred_corpus):
                    pred_corpus[slide_id] = []
                    label_corpus[slide_id] = int(row[1]) #正解ラベル

                pred_corpus[slide_id].append(prob_list)
                if(slide_id not in slide_id_list):
                    slide_id_list.append(slide_id)
    # slide単位の事後確率計算
    slide_prob = []
    true_label_list = []
    pred_label_list = []
    max_prob = []

    for slide_id in slide_id_list:
        prob_list= pred_corpus[slide_id]
        bag_num = len(prob_list) # Bagの数

        sum_prob = [0] * n_class
        for prob in prob_list:
            for i in range(len(prob)):
                sum_prob[i] += np.log(float(prob[i]))
        mean_prob = []
        for i in range(n_class):
            sum_prob[i] = np.exp(sum_prob[i] / bag_num)
            mean_prob.append(sum_prob[i])

        #mean_probを出力
        slide_prob.append(mean_prob)
        true_label_list.append(label_corpus[slide_id])

        #ここでクラス確率のリストとラベルがセットになってる

        pred_label_list.append(mean_prob.index(max(mean_prob)))
        max_prob.append(max(mean_prob)/sum(mean_prob))

    return slide_id_list, slide_prob, true_label_list, pred_label_list, max_prob

def eval(model, params, n_class):

    #output_file = f'test_predict/{model}_predict_{params}tmp.csv'
    #output_cmat = f'test_predict/{model}_cmat_{params}tmp.csv'


    cv_metric = np.zeros((5,4))

    for i in range(5):

        result_file = f'test_result/{model}_test_cv-{i+1}_{params}.csv'

        slide_id_tmp, slide_prob_tmp, true_label_tmp, pred_label_tmp, max_prob_tmp = get_slide_prob_label(result_file, n_class)

        np_true = np.array(true_label_tmp)
        np_pred = np.array(pred_label_tmp)

        AUC_class = np.zeros((n_class))

        for j in range(n_class):

            bin_true = np.zeros((len(slide_id_tmp)))
            prob_bin = np.zeros((len(slide_id_tmp)))

            for k in range(len(slide_id_tmp)):

                #print(np_true[k])

                if int(np_true[k]) == j:
                    bin_true[k] = 1

                #prob_bin = np.zeros((len(slide_id_tmp)))
                prob_bin[k] = slide_prob_tmp[k][j]
                #print(slide_prob_tmp[k])
                #print(prob_bin[k])
                #print(bin_true[k])
                #print(prob_bin[k])

            AUC_class[j] = roc_auc_score(bin_true, prob_bin)
            #print(AUC_class[j])

            #y=input()

        cv_metric[i,0] = accuracy_score(np_true, np_pred)
        cv_metric[i,1] = precision_score(np_true, np_pred, average='macro')
        cv_metric[i,2] = recall_score(np_true, np_pred, average='macro')
        cv_metric[i,3] = np.mean(AUC_class)
        #= str(f1_score(np_true, np_pred, average='macro'))

    print(f'accuracy: {np.mean(cv_metric[:,0])}, se: {np.std(cv_metric[:,0], ddof=1) / np.sqrt(5)}')
    print(f'precision: {np.mean(cv_metric[:,1])}, se: {np.std(cv_metric[:,1], ddof=1) / np.sqrt(5)}')
    print(f'recall: {np.mean(cv_metric[:,2])}, se: {np.std(cv_metric[:,2], ddof=1) / np.sqrt(5)}')
    print(f'AUC: {np.mean(cv_metric[:,3])}, se: {np.std(cv_metric[:,3], ddof=1) / np.sqrt(5)}')
    #print('macroF1',f1_score(np_true, np_pred, average='macro'))

    #np.savetxt(output_file, output, delimiter=',',fmt="%s")

    #cmat = confusion_matrix(np_true, np_pred)

    #np.savetxt(output_cmat, cmat, delimiter=',',fmt="%s")

    #print('acc', accuracy_score(np_true, np_pred))
    #print('macroF1',f1_score(np_true, np_pred, average='macro'))

if __name__ == "__main__":

    args = sys.argv
    model = args[1]
    params = args[2]
    n_class = int(args[3])

    eval(model, params, n_class) #, t)
