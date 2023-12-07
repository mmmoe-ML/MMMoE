# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
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
        slide_prob.append(mean_prob)
        true_label_list.append(label_corpus[slide_id])

        pred_label_list.append(mean_prob.index(max(mean_prob)))
        max_prob.append(max(mean_prob)/sum(mean_prob))

    return slide_id_list, slide_prob, true_label_list, pred_label_list, max_prob

def eval(model, n_class, t, fix):

    output_file = f'test_predict/{model}_predict_t-{t}_fix{fix}.csv'
    output_cmat = f'test_predict/{model}_cmat_t-{t}_fix{fix}.csv'

    slide_id_list = []
    slide_prob_list = []
    true_label_list = []
    pred_label_list = []
    max_prob_list = []

    for i in range(5):

        if fix == 2:
            result_file = f'test_result/{model}_test_cv-{i+1}_t-{t}_lr-3_fix2.csv'
        elif fix == 1:
            result_file = f'test_result/{model}_test_cv-{i+1}_t-{t}_lr-3_fix.csv'
        else:
            result_file = f'test_result/{model}_test_cv-{i+1}_t-{t}_lr-3.csv'

        slide_id_tmp, slide_prob_tmp, true_label_tmp, pred_label_tmp, max_prob_tmp = get_slide_prob_label(result_file, n_class)

        slide_id_list += slide_id_tmp
        #slide_prob_list += slide_prob_tmp
        true_label_list += true_label_tmp
        pred_label_list += pred_label_tmp
        max_prob_list += max_prob_tmp

    output = np.zeros((len(slide_id_list)+1,4),dtype='object')

    for i in range (len(slide_id_list)):
        output[i,0] = slide_id_list[i]
        output[i,1] = str(true_label_list[i])
        output[i,2] = str(pred_label_list[i])
        output[i,3] = str(max_prob_list[i])

    np_true = np.array(true_label_list)
    np_pred = np.array(pred_label_list)

    output[len(slide_id_list),0] = str(accuracy_score(np_true, np_pred))
    output[len(slide_id_list),1] = 0 #str(precision_score(np_true, np_pred))
    output[len(slide_id_list),2] = 0 #str(recall_score(np_true, np_pred))
    output[len(slide_id_list),3] = str(f1_score(np_true, np_pred, average='macro'))

    np.savetxt(output_file, output, delimiter=',',fmt="%s")

    cmat = confusion_matrix(np_true, np_pred)

    np.savetxt(output_cmat, cmat, delimiter=',',fmt="%s")

    print('acc', accuracy_score(np_true, np_pred))
    print('macroF1',f1_score(np_true, np_pred, average='macro'))

if __name__ == "__main__":

    args = sys.argv
    model = args[1]
    n_class = int(args[2])
    t = float(args[3])
    fix = int(args[4])
    #t = float(args[3])

    eval(model, n_class, t, fix) #, t)
