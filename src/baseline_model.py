


import json
import numpy as np
import argparse

import torch
import torch.nn as nn

from evaluation_metrics import f1_threshold_score
from evaluation_metrics import pr_auc_scores_average, roc_auc_scores_average
from evaluation_metrics import one_error_score, coverage_score, ranking_loss_score



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



def main(args):
    temp_id_gold_dir = './work/temp_id_gold.json'
    temp_id_gold = json.load(open(temp_id_gold_dir))
    temp_id_info_dir = './work/temp_id_info.json'
    temp_id_info = json.load(open(temp_id_info_dir))

    temp_ids = list(temp_id_info.keys())

    params_dict = json.load(open('{}/params.json'.format(args.directory)))    

    y_true = [] 
    for _lo_id, lo_id_dict in temp_id_gold.items():
        for _fb_unit_id, temp_data_dict in lo_id_dict['temp_data'].items():
            y_t = temp_data_dict['temp_id']
            flag_count = y_t.count(1)
            assert flag_count != 0
            y_true.append(y_t)
    y_true = np.array(y_true)
    
    np_y_true = np.array(y_true)
    n_each_temp_id = np.sum(np_y_true, axis=0)

    print('--- {} model ---'.format(args.baseline_model))
    
    for iter_i in range(params_dict['iteration_size']):

        data_split_dict = json.load(
            open(
                '{}/iter_{}/data_split_fold_0.json'.format(args.directory, iter_i)
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
            y_pred_logits = np.random.randn(n_batch, n_label)
            y_pred_i = sigmoid(y_pred_logits)
        
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

        
        pr_scores, pr_average = pr_auc_scores_average(y_true_i, y_pred_i)
        roc_scores, roc_average = roc_auc_scores_average(y_true_i, y_pred_i)

        one_err_scores, one_err_average = one_error_score(y_true_i, y_pred_i)
        coverage = coverage_score(y_true_i, y_pred_i)
        rank_loss = ranking_loss_score(y_true_i, y_pred_i)


        print(
            'Iter\t{}\nOneError\t{:.3f}\nCoverageLoss\t{:.3f}\nRankingLoss\t{:.3f}\nPR score\t{:.3f}\nROC score\t{:.3f}'.format(
                iter_i, one_err_average, coverage, rank_loss, pr_average, roc_average
                )
            )
        print('PR/ROC score details')
        for i, (pr_s, roc_s) in enumerate(zip(pr_scores, roc_scores)):
            print(
                'temp{}\t{:.3f}\t{:.3f}'.format(
                    temp_ids[i],
                    pr_s,
                    roc_s
                )
            )
        


        print()
        #print(
        #    'F1 Micro: {:.3f}\tF1 Macro: {:.3f}'.format(
        #        micro_f1, macro_f1
        #        )
        #    )

        break
    





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-bm', '--baseline-model', default='sampling', choices=['random', 'majority', 'sampling'],
        help='baseline model.'
    )
    parser.add_argument(
        '-dir', '--directory',
        help='output file directory.'
    )

    args = parser.parse_args()
    
    main(args)



