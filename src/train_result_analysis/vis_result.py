

import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import f1_score



"""
param_info_file = './out_test/params.json'
param_info = json.load(open(param_info_file))
n_fold = param_info['fold_size']
n_iter = param_info['iteration_size']



for i in range(n_iter):
    test_loss, test_one_err, test_coverage, test_rank_loss,\
         test_mAP_w, test_roc_auc = [], [], [], [], [], []
    test_f1_micro, test_f1_macro = [], []
    for j in range(n_fold):
        test_result_file = './out_test/iter_{}/results_fold_{}.json'.format(i, j)
        test_result = json.load(open(test_result_file))
        pred = test_result['prediction']
        true = test_result['gold']

        pred = np.where(np.array(pred) >= 0.5, 1, 0)
        f1_micro = f1_score(y_pred=pred, y_true=true, average='micro', zero_division=0)
        f1_macro = f1_score(y_pred=pred, y_true=true, average='macro', zero_division=0)

        loss = test_result['loss']
        one_err = test_result['one_error']
        coverage = test_result['coverage_error']
        rank_loss = test_result['rank_loss']
        mAPs = test_result['mAP']
        roc_auc = test_result['ROC_AUC']
        
        test_loss.append(loss)
        test_one_err.append(one_err)
        test_coverage.append(coverage)
        test_rank_loss.append(rank_loss)
        test_mAP_w.append(mAPs)
        test_roc_auc.append(roc_auc)
        test_f1_micro.append(f1_micro)
        test_f1_macro.append(f1_macro)

    print(
        'Iter: {}\tLoss: {:.3f}\t \
        OneError: {:.3f}\tCoverageLoss: {:.3f}\t \
        RankingLoss: {:.3f}\tmAP micro: {:.3f}\t \
        ROC AUC micro: {:.3f}'.format(
            i, np.mean(test_loss), np.mean(test_one_err), np.mean(test_coverage),
            np.mean(rank_loss), np.mean(test_mAP_w), np.mean(test_roc_auc)
        )
    )
    print(
        'F1 Micro: {:.3f}\tF1 Macro: {:.3f}'.format(
            np.mean(test_f1_micro), np.mean(test_f1_macro)
        )
    )

    break



"""
result_file = './out_test_gm=5/iter_0/train_log_fold_0.json'

result_data = json.load(open(result_file))

train_losses = result_data['train_losses']
valid_losses = result_data['val_losses']
coverages = result_data['coverage_error']
mAP = result_data['mAP']
#roc_auc = result_data['ROC_AUC']
#f1_micro = result_data['f1_micro']
#f1_macro = result_data['f1_macro']
#print(train_losses, valid_losses, coverages)

epoch = [e for e, _ in enumerate(train_losses)]
#print(epoch)
print(len(epoch))

plt.plot(epoch, train_losses, label="train loss")
plt.plot(epoch, valid_losses, label="valid loss")
#plt.plot(epoch, coverages, label="coverage")
#plt.plot(epoch, mAP, label="mAP")
#plt.plot(epoch, roc_auc, label="ROC_AUC")
#plt.plot(epoch, f1_micro, label="f1_micro")
#plt.plot(epoch, f1_macro, label="f1_macro")

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

plt.show()

