
import os
import sys
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation_metrics import f1_threshold_score_label
from evaluation_metrics import pr_auc_scores_label, roc_auc_scores_label
from evaluation_metrics import one_error_score_label, coverage_score_label, ranking_loss_score_label



parser = argparse.ArgumentParser()
parser.add_argument(
    '-md', '--mode', choices=['train', 'test'], default='train',
    help='Train / Test'
)
parser.add_argument(
    '-dir', '--directory',
    help='output file directory.'
)

args = parser.parse_args()


gamma_list = [0, 0.5, 1, 2, 5]

if args.mode == 'train':
    ##############
    # train data #
    ##############
    losses = []
    pr_averages_m, pr_averages_w, pr_averages_s = [], [], []

    for gamma in gamma_list:
        tr_result_file = '{}/out_test_gm={}/iter_0/train_log_fold_0.json'.format(args.directory, gamma)

        tr_result_data = json.load(open(tr_result_file))

        y_val_preds = tr_result_data['prediction']
        y_val_trues = tr_result_data['gold']
        
        #train_losses = result_data['train_losses']
        valid_losses = tr_result_data['val_losses']
        
        mAP_micro = tr_result_data['pr_averages']['micro']
        mAP_weighted = tr_result_data['pr_averages']['weighted']
        mAP_samples = tr_result_data['pr_averages']['samples']

        epoch = [e for e, _ in enumerate(valid_losses)]

        losses.append(valid_losses)
        pr_averages_m.append(mAP_micro)
        pr_averages_w.append(mAP_weighted)
        pr_averages_s.append(mAP_samples)
        

    #print(losses)
    #print(probs)
    #print(mAPs)
    #print(pr_averages)
    #print(roc_averages)


    """
    ## probability - loss
    plt.plot(probs[0], losses[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(probs[1], losses[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(probs[2], losses[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(probs[3], losses[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(probs[4], losses[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=12)
    plt.show()
    """

    ## mAP micro - loss
    plt.plot(pr_averages_m[0], losses[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(pr_averages_m[1], losses[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(pr_averages_m[2], losses[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(pr_averages_m[3], losses[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(pr_averages_m[4], losses[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
    plt.show()

    ## mAP weighted - loss
    plt.plot(pr_averages_w[0], losses[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(pr_averages_w[1], losses[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(pr_averages_w[2], losses[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(pr_averages_w[3], losses[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(pr_averages_w[4], losses[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
    plt.show()

    ## mAP samples - loss
    plt.plot(pr_averages_s[0], losses[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(pr_averages_s[1], losses[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(pr_averages_s[2], losses[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(pr_averages_s[3], losses[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(pr_averages_s[4], losses[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
    plt.show()

    ## epoch - mAP
    plt.plot(epoch, pr_averages_m[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(epoch, pr_averages_m[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(epoch, pr_averages_m[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(epoch, pr_averages_m[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(epoch, pr_averages_m[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)
    plt.show()

    """
    ## ROC - loss
    plt.plot(roc_averages[0], losses[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(roc_averages[1], losses[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(roc_averages[2], losses[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(roc_averages[3], losses[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(roc_averages[4], losses[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
    plt.show()


    ## epoch - ROC
    plt.plot(epoch, roc_averages[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(epoch, roc_averages[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(epoch, roc_averages[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(epoch, roc_averages[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(epoch, roc_averages[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)
    plt.show()
    """

else:
    ###############
    # result data #
    ###############
    pr_averages_m, pr_averages_w, pr_averages_s = [], [], []

    for gamma in gamma_list:
        te_result_file = '{}/out_test_gm={}/iter_0/results_fold_0.json'.format(args.directory, gamma)
        te_result_data = json.load(open(te_result_file))

        preds = te_result_data['prediction']
        golds = te_result_data['gold']

        mAP_micro = te_result_data['PR']['average']['micro']
        mAP_weighted = te_result_data['PR']['average']['weighted']
        mAP_samples = te_result_data['PR']['average']['samples']

        pr_averages_m.append(mAP_micro)
        pr_averages_w.append(mAP_weighted)
        pr_averages_s.append(mAP_samples)


    print(pr_averages_m)
    print(np.argmax(pr_averages_m))
    print(pr_averages_w)
    print(np.argmax(pr_averages_w))
    print(pr_averages_s)
    print(np.argmax(pr_averages_s))

    #print(one_errors)
    #print(cov_losses)
    #print(rank_losses)
    
