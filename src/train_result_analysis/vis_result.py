

import matplotlib.pyplot as plt
import json
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument(
    '-md', '--mode', choices=['train', 'test'], default='train',
    help='Train / Test'
)
parser.add_argument(
    '-dir', '--directory',
    help='file directory'
)
args = parser.parse_args()




if args.mode == 'test':
    param_info_file = args.directory + '/params.json'
    param_info = json.load(open(param_info_file))
    n_fold = param_info['fold_size']
    n_iter = param_info['iteration_size']


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
            'Iter: {}\tLoss: {:.3f}\t \
            OneError: {:.3f}\tCoverageLoss: {:.3f}\t \
            RankingLoss: {:.3f}'.format(
                i, np.mean(test_loss),
                np.mean(test_one_err), np.mean(test_coverage),
                np.mean(rank_loss)
            )
        )
        print(
            'PR score: {:.3f}\n \
            {}\n \
            ROC score: {:.3f}\n \
            {}'.format(
                np.mean(test_pr_averages),
                np.mean(test_pr_scores, axis=0),
                np.mean(test_roc_averages),
                np.mean(test_roc_scores, axis=0)
            )
        )
        #print(
        #    'F1 Micro: {:.3f}\tF1 Macro: {:.3f}'.format(
        #        np.mean(test_f1_micro), np.mean(test_f1_macro)
        #    )
        #)
        print()

        break


else:
    result_file = '{}/iter_0/train_log_fold_0.json'.format(args.directory)

    result_data = json.load(open(result_file))

    train_losses = result_data['train_losses']
    valid_losses = result_data['val_losses']
    pr_averages = result_data['pr_averages']
    roc_averages = result_data['roc_averages']

    #f1_micro = result_data['f1_micro']
    #f1_macro = result_data['f1_macro']
    #print(train_losses, valid_losses, coverages)

    epoch = [e for e, _ in enumerate(train_losses)]
    #print(epoch)
    print(len(epoch))

    plt.plot(epoch, train_losses, label="train loss")
    plt.plot(epoch, valid_losses, label="valid loss")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
    plt.show()

    plt.plot(valid_losses, pr_averages, label="mAP")
    plt.show()

    plt.plot(valid_losses, roc_averages, label="ROC_AUC")
    plt.show()

    #plt.plot(epoch, f1_micro, label="f1_micro")
    #plt.plot(epoch, f1_macro, label="f1_macro")

