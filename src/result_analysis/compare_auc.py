

import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(args):
    temp_id_gold_dir = './work/temp_id_gold.json'
    temp_id_gold = json.load(open(temp_id_gold_dir))
    temp_id_info_dir = './work/temp_id_info.json'
    temp_id_info = json.load(open(temp_id_info_dir))

    temp_ids = list(temp_id_info.keys())

    params_dict = json.load(open('{}/out_test_1/params.json'.format(args.directory)))
    iter_size = params_dict['iteration_size']
    fold_size = params_dict['fold_size']

    y_true_all = [] 
    for _lo_id, lo_id_dict in temp_id_gold.items():
        for _fb_unit_id, temp_data_dict in lo_id_dict['temp_data'].items():
            y_t = temp_data_dict['temp_id']
            flag_count = y_t.count(1)
            assert flag_count != 0
            y_true_all.append(y_t)
    y_true_all = np.array(y_true_all)
    
    np_y_true = np.array(y_true_all)
    n_each_temp_id = np.sum(np_y_true, axis=0)

    
    for iter_i in range(iter_size):

        #################
        # data settings #
        #################
        ## random baseline model
        data_split_dict = json.load(
            open(
                '{}/out_test_1/iter_{}/data_split_fold_0.json'.format(args.directory, iter_i)
            )
        )
        test_data_ids = data_split_dict['test']['data_ids']
        test_data_ids = np.array(test_data_ids)

        y_trues = y_true_all[test_data_ids]
        #print("data size: {}".format(y_true_i.shape))

        n_batch, n_label = y_trues.shape[0], y_trues.shape[1]
        y_pred_logits_base1 = np.random.randn(n_batch, n_label)
        y_preds_base1 = sigmoid(y_pred_logits_base1)
        
        
        ## roberta based classifier 01 #
        y_preds_list = []
        for fold_i in range(fold_size):
            result_data_rbc1_dir = '{}/out_test_1/iter_{}/results_fold_{}.json'.format(args.directory, iter_i, fold_i)
            result_data_rbc1 = json.load(open(result_data_rbc1_dir))
            y_preds = result_data_rbc1['prediction']
            y_preds = np.array(y_preds)
            y_preds_list.append(y_preds)

        y_preds_rbc1 = y_preds_list[0] + y_preds_list[1] + y_preds_list[2] + y_preds_list[3] + y_preds_list[4]
        y_preds_rbc1 = y_preds_rbc1 / fold_size


        ## roberta based classifier 02 #
        y_preds_list = []
        for fold_i in range(fold_size):
            result_data_rbc1_dir = '{}/out_test_2/iter_{}/results_fold_{}.json'.format(args.directory, iter_i, fold_i)
            result_data_rbc1 = json.load(open(result_data_rbc1_dir))
            y_preds = result_data_rbc1['prediction']
            y_preds = np.array(y_preds)
            y_preds_list.append(y_preds)

        y_preds_rbc2 = y_preds_list[0] + y_preds_list[1] + y_preds_list[2] + y_preds_list[3] + y_preds_list[4]
        y_preds_rbc2 = y_preds_rbc2 / fold_size


        ## roberta based classifier 03 #
        y_preds_list = []
        for fold_i in range(fold_size):
            result_data_rbc1_dir = '{}/out_test_3/iter_{}/results_fold_{}.json'.format(args.directory, iter_i, fold_i)
            result_data_rbc1 = json.load(open(result_data_rbc1_dir))
            y_preds = result_data_rbc1['prediction']
            y_preds = np.array(y_preds)
            y_preds_list.append(y_preds)

        y_preds_rbc3 = y_preds_list[0] + y_preds_list[1] + y_preds_list[2] + y_preds_list[3] + y_preds_list[4]
        y_preds_rbc3 = y_preds_rbc3 / fold_size        


        assert y_preds_base1.shape == y_preds_rbc1.shape
        assert y_preds_base1.shape == y_preds_rbc2.shape
        assert y_preds_base1.shape == y_preds_rbc3.shape

    
        ##################
        # ROC/PR curve plot #
        ##################
        for i in range(25):
            fpr_base1, tpr_base1, thresholds_base1 = roc_curve(y_trues.T[i], y_preds_base1.T[i])
            fpr_rbc1, tpr_rbc1, thresholds_rbc1 = roc_curve(y_trues.T[i], y_preds_rbc1.T[i])
            fpr_rbc2, tpr_rbc2, thresholds_rbc2 = roc_curve(y_trues.T[i], y_preds_rbc2.T[i])
            fpr_rbc3, tpr_rbc3, thresholds_rbc3 = roc_curve(y_trues.T[i], y_preds_rbc3.T[i])

            pr_base1, rc_base1, thresholds_base1 = precision_recall_curve(y_trues.T[i], y_preds_base1.T[i])
            pr_rbc1, rc_rbc1, thresholds_rbc1 = precision_recall_curve(y_trues.T[i], y_preds_rbc1.T[i])
            pr_rbc2, rc_rbc2, thresholds_rbc2 = precision_recall_curve(y_trues.T[i], y_preds_rbc2.T[i])
            pr_rbc3, rc_rbc3, thresholds_rbc3 = precision_recall_curve(y_trues.T[i], y_preds_rbc3.T[i])
            

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            ax1.plot(fpr_base1, tpr_base1, label='baseline 1')
            ax1.plot(fpr_rbc1, tpr_rbc1, label='roberta based 1')
            ax1.plot(fpr_rbc2, tpr_rbc2, label='roberta based 2')
            ax1.plot(fpr_rbc3, tpr_rbc3, label='roberta based 3')

            ax2.plot(rc_base1, pr_base1, label='baseline 1')
            ax2.plot(rc_rbc1, pr_rbc1, label='roberta based 1')
            ax2.plot(rc_rbc2, pr_rbc2, label='roberta based 2')
            ax2.plot(rc_rbc3, pr_rbc3, label='roberta based 3')

            ax1.legend()
            ax1.set_title('ROC curve template id {}'.format(temp_ids[i]))
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.grid(True)
            
            ax2.legend()
            ax2.set_title('PR curve template id {}'.format(temp_ids[i]))
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.grid(True)

            fig.set_figheight(6)
            fig.set_figwidth(14)

            plt.savefig('{}/img/ROC_PR_curve_template_id_{}.png'.format(args.directory, temp_ids[i]))
            plt.show()


        break
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-dir', '--directory',
        help='output file directory.'
    )

    args = parser.parse_args()
    
    main(args)



