


import json
import numpy as np
import argparse

import torch
import torch.nn as nn

#from sklearn.metrics import recall_score, precision_score, f1_score
#from sklearn.metrics import classification_report
from sklearn.metrics import coverage_error, roc_auc_score, average_precision_score, label_ranking_loss

from model import FocalLoss



def one_error(y_true, y_pred):
    count = 0
    for y_p, y_t in zip(y_pred, y_true):
        top_cls = np.argmax(y_p)
        if y_t[top_cls] != 1:
            count += 1

    return count / len(y_pred)
    


def main(args):
    temp_id_gold_dir = './work/temp_id_gold.json'
    temp_id_gold = json.load(open(temp_id_gold_dir))

    params_dict = json.load(open('./out_test/params.json'))
    loss_fn = FocalLoss(gamma=2, alpha=None)


    y_true = [] 
    for _lo_id, lo_id_dict in temp_id_gold.items():
        for _fb_unit_id, temp_data_dict in lo_id_dict['temp_data'].items():
            y_t = temp_data_dict['temp_id']
            flag_count = y_t.count(1)
            assert flag_count != 0
            y_true.append(y_t)
    y_true = np.array(y_true)
    
    #n_lo_id = len(y_true)
    #n_temp_id = len(y_true[0])
    np_y_true = np.array(y_true)
    n_each_temp_id = np.sum(np_y_true, axis=0)

    print('--- {} model ---'.format(args.baseline_model))
    
    for iter_i in range(params_dict['iteration_size']):

        data_split_dict = json.load(
            open(
                './out_test/iter_{}/data_split_fold_0.json'.format(iter_i)
            )
        )
        test_data_ids = data_split_dict['test']['data_ids']
        test_data_ids = np.array(test_data_ids)

        y_true_i = y_true[test_data_ids]
        print("data size: {}".format(y_true_i.shape))

        n_batch = y_true_i.shape[0]
        n_label = y_true_i.shape[1]


        if args.baseline_model == 'random':
            ##################
            ## Random Model ##
            ##################
            y_pred_logits = torch.randn(n_batch, n_label)
            y_pred_i = torch.sigmoid(y_pred_logits)
            y_pred_i = y_pred_i.detach().numpy().copy()
        
        elif args.baseline_model == 'majority':
            ####################
            ## Majority Model ##
            ####################
            majority_position = np.argmax(n_each_temp_id)
            y_pred_i = np.zeros((n_batch, n_label-1))
            y_pred_i = np.insert(y_pred_i, majority_position, 1, axis=1)
        
        elif args.baseline_model == 'sampling':
            ####################
            ## Sampling Model ##
            ####################
            y_pred_i = np.exp(n_each_temp_id) / np.sum(np.exp(n_each_temp_id))
            y_pred_i = np.tile(y_pred_i, (n_batch, 1))
            print(y_pred_i)


        loss = loss_fn(
                torch.from_numpy(y_pred_i.astype(np.float32)),
                torch.from_numpy(y_true_i.astype(np.float32))
                )
        coverage = coverage_error(y_true_i, y_pred_i)

        #mAP_m = average_precision_score(y_true_i, y_pred_i, average='micro')
        mAP_w = average_precision_score(y_true_i, y_pred_i, average='weighted')
        #mAP_s = average_precision_score(y_true_i, y_pred_i, average='samples')

        roc_auc = roc_auc_score(y_true_i, y_pred_i, average='weighted')

        rank_loss = label_ranking_loss(y_true_i, y_pred_i)
        one_err = one_error(y_true_i, y_pred_i)

        #y_pred_01 = np.where(y_pred >= 0.5, 1, 0)
        #micro_f1 = f1_score(y_pred_01, y_true, average='micro')
        #macro_f1 = f1_score(y_pred_01, y_true, average='macro')        

        print(
            'Iter: {}\tLoss: {:.3f}\t \
            OneError: {:.3f}\tCoverageLoss: {:.3f}\t \
            RankingLoss: {:.3f}\tmAP weighted: {:.3f}\t \
            ROC AUC weighted: {:.3f}'.format(
                iter_i, loss, one_err, coverage, rank_loss, mAP_w, roc_auc
                )
            )
    





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-bm', '--baseline-model', default='sampling', choices=['random', 'majority', 'sampling'],
        help='baseline model.'
    )

    args = parser.parse_args()
    
    main(args)



