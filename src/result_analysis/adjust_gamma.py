
import os
import sys
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation_metrics import f1_threshold_score
from evaluation_metrics import pr_auc_scores_average, roc_auc_scores_average
from evaluation_metrics import one_error_score, coverage_score, ranking_loss_score



parser = argparse.ArgumentParser()
parser.add_argument(
    '-md', '--mode', choices=['train', 'test'], default='train',
    help='Train / Test'
)
parser.add_argument(
    '-av', '--average', choices=[None, 'macro', 'weighted'], default='macro',
    help='how to calculate PR/ROC AUC score.'
)
args = parser.parse_args()


gamma_list = [0, 0.5, 1, 2, 5]

if args.mode == 'train':
    ##############
    # train data #
    ##############
    losses, probs= [], []
    pr_averages, roc_averages = [], []

    for gamma in gamma_list:
        tr_result_file = './out_test_gm={}/iter_0/train_log_fold_0.json'.format(gamma)

        tr_result_data = json.load(open(tr_result_file))

        y_val_preds = tr_result_data['prediction']
        y_val_trues = tr_result_data['gold']
        
        #train_losses = result_data['train_losses']
        valid_losses = tr_result_data['val_losses']
        
        #coverages = result_data['coverage_error']
        #mAP = tr_result_data['mAP']
        #roc_auc = result_data['ROC_AUC']


        epoch = [e for e, _ in enumerate(valid_losses)]
        
        prob = []
        pr_average_stock, roc_average_stock = [], []
        for y_val_pred, y_val_true in zip(y_val_preds, y_val_trues):
            np_y_val_pred = np.array(y_val_pred)
            np_y_val_true = np.array(y_val_true)
            gold_index = np.where(np_y_val_true==1)
            p = np.average(np_y_val_pred[gold_index])
            prob.append(p)

            _pr_scores, pr_average = pr_auc_scores_average(np.array(y_val_true), np.array(y_val_pred), average=args.average)
            _roc_scores, roc_average = roc_auc_scores_average(np.array(y_val_true), np.array(y_val_pred), average=args.average)

            pr_average_stock.append(pr_average)
            roc_average_stock.append(roc_average)

        probs.append(prob)
        pr_averages.append(pr_average_stock)
        roc_averages.append(roc_average_stock)
        losses.append(valid_losses)
        

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

    ## mAP - loss
    plt.plot(pr_averages[0], losses[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(pr_averages[1], losses[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(pr_averages[2], losses[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(pr_averages[3], losses[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(pr_averages[4], losses[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
    plt.show()


    ## epoch - mAP
    plt.plot(epoch, pr_averages[0], label='gamma={}'.format(gamma_list[0]))
    plt.plot(epoch, pr_averages[1], label='gamma={}'.format(gamma_list[1]))
    plt.plot(epoch, pr_averages[2], label='gamma={}'.format(gamma_list[2]))
    plt.plot(epoch, pr_averages[3], label='gamma={}'.format(gamma_list[3]))
    plt.plot(epoch, pr_averages[4], label='gamma={}'.format(gamma_list[4]))

    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)
    plt.show()

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


else:
    ###############
    # result data #
    ###############
    pr_ws = []
    prses = []
    one_errors = []
    cov_losses = []
    rank_losses = []
    pr_averages, roc_averages = [], []

    for gamma in gamma_list:
        te_result_file = './out_test_gm={}/iter_0/results_fold_0.json'.format(gamma)
        te_result_data = json.load(open(te_result_file))

        preds = te_result_data['prediction']
        golds = te_result_data['gold']

        preds = np.array(preds)
        golds = np.array(golds)

        pr_scores, pr_average = pr_auc_scores_average(golds, preds, average=args.average)
        roc_scores, roc_average = roc_auc_scores_average(golds, preds, average=args.average)

        #pr_ws.append(pr_w)
        
        pr_averages.append(pr_average)
        roc_averages.append(roc_average)

        
        one_err_scores, one_err_average = one_error_score(golds, preds)
        coverage = coverage_score(golds, preds)
        rank_loss = ranking_loss_score(golds, preds)

        one_errors.append(one_err_average)
        cov_losses.append(coverage)
        rank_losses.append(rank_loss)


    print(pr_averages)
    print(roc_averages)
    print(np.argmax(pr_averages))
    print(np.argmax(roc_averages))

    #print(one_errors)
    #print(cov_losses)
    #print(rank_losses)
    
