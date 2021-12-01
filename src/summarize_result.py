

import matplotlib.pyplot as plt
import json
import numpy as np
import argparse



parser = argparse.ArgumentParser()

parser.add_argument(
    '-dir', '--directory',
    help='file directory'
)
args = parser.parse_args()




param_info_file = args.directory + '/params.json'
param_info = json.load(open(param_info_file))
n_fold = param_info['fold_size']
n_iter = param_info['iteration_size']

temp_id_info_file = './work/temp_id_info.json'
temp_id_info = json.load(open(temp_id_info_file))
temp_ids = list(temp_id_info.keys())


for i in range(n_iter):
    test_loss, test_one_err, test_coverage, test_rank_loss = [], [], [], []
    test_pr_scores, test_pr_averages, test_roc_scores, test_roc_averages = [], [], [], []
    #test_f1_micro, test_f1_macro = [], []
    for j in range(n_fold):
        test_result_file = '{}/iter_{}/results_fold_{}.json'.format(args.directory, i, j)
        test_result = json.load(open(test_result_file))
        pred = test_result['prediction']
        true = test_result['gold']


        loss = test_result['loss']
        one_err = test_result['one_error']
        coverage = test_result['coverage_error']
        rank_loss = test_result['rank_loss']
        pr_score = test_result['PR']['scores']
        pr_average = test_result['PR']['average']
        roc_score = test_result['ROC']['scores']
        roc_average = test_result['ROC']['average']
        
        test_loss.append(loss)
        test_one_err.append(one_err)
        test_coverage.append(coverage)
        test_rank_loss.append(rank_loss)
        test_pr_scores.append(pr_score)
        test_pr_averages.append(pr_average)
        test_roc_scores.append(roc_score)
        test_roc_averages.append(roc_average)
        #test_f1_micro.append(f1_micro)
        #test_f1_macro.append(f1_macro)

    print(
        'Iter\t{}\nLoss\t{:.3f}\nOneError\t{:.3f}\nCoverageLoss\t{:.3f}\nRankingLoss\t{:.3f}'.format(
            i, np.mean(test_loss),
            np.mean(test_one_err), np.mean(test_coverage),
            np.mean(rank_loss)
        )
    )
    print(
        'PR score\t{:.3f}\nROC score\t{:.3f}'.format(
            np.mean(test_pr_averages),
            np.mean(test_roc_averages)
        )
    )
    print('PR/ROC score details')
    for i, (pr_s, roc_s) in enumerate(zip(
                            np.mean(test_pr_scores, axis=0),
                            np.mean(test_roc_scores, axis=0)    
                            )):
        print(
            'temp{}\t{:.3f}\t{:.3f}'.format(
                temp_ids[i],
                pr_s,
                roc_s
            )
        )
    
    
    #print(
    #    'F1 Micro: {:.3f}\tF1 Macro: {:.3f}'.format(
    #        np.mean(test_f1_micro), np.mean(test_f1_macro)
    #    )
    #)
    print()

    break

