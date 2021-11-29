
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import average_precision_score, coverage_error, label_ranking_loss


def one_error(y_true, y_pred):
    one_error_list = []
    for y_p, y_t in zip(y_pred, y_true):
        sorted_idx = np.argsort(-y_p)
        y_p = y_p[sorted_idx]
        y_t = y_t[sorted_idx]

        if np.sum(y_t) == 0:
            score = np.nan
        else:
            count = 0
            for p, t in zip(y_p, y_t):
                if t == 1:
                    break
                else:
                    count += 1
            score = count / len(y_t)

        one_error_list.append(score)


    assert len(one_error_list) == 25

    return np.array(one_error_list)


gamma_list = [0, 0.5, 1, 2, 5]
##############
# train data #
##############
losses, probs, mAPs = [], [], []
for gamma in gamma_list:
    tr_result_file = './out_test_gm={}/iter_0/train_log_fold_0.json'.format(gamma)

    tr_result_data = json.load(open(tr_result_file))

    y_val_preds = tr_result_data['prediction']
    y_val_trues = tr_result_data['gold']
    
    #train_losses = result_data['train_losses']
    valid_losses = tr_result_data['val_losses']
    
    #coverages = result_data['coverage_error']
    mAP = tr_result_data['mAP']
    #roc_auc = result_data['ROC_AUC']

    epoch = [e for e, _ in enumerate(valid_losses)]
    
    prob = []
    for y_val_pred, y_val_true in zip(y_val_preds, y_val_trues):
        np_y_val_pred = np.array(y_val_pred)
        np_y_val_true = np.array(y_val_true)
        gold_index = np.where(np_y_val_true==1)
        p = np.average(np_y_val_pred[gold_index])
        prob.append(p)

    probs.append(prob)
    losses.append(valid_losses)
    mAPs.append(mAP)
    

#print(losses)
#print(probs)
#print(mAPs)


"""
## probability - loss
plt.plot(probs[0], losses[0], label='gamma={}'.format(gamma_list[0]))
plt.plot(probs[1], losses[1], label='gamma={}'.format(gamma_list[1]))
plt.plot(probs[2], losses[2], label='gamma={}'.format(gamma_list[2]))
plt.plot(probs[3], losses[3], label='gamma={}'.format(gamma_list[3]))
plt.plot(probs[4], losses[4], label='gamma={}'.format(gamma_list[4]))

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
plt.show()
"""

## mAP - loss
plt.plot(mAPs[0], losses[0], label='gamma={}'.format(gamma_list[0]))
plt.plot(mAPs[1], losses[1], label='gamma={}'.format(gamma_list[1]))
plt.plot(mAPs[2], losses[2], label='gamma={}'.format(gamma_list[2]))
plt.plot(mAPs[3], losses[3], label='gamma={}'.format(gamma_list[3]))
plt.plot(mAPs[4], losses[4], label='gamma={}'.format(gamma_list[4]))

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
plt.show()


## epoch - mAP
plt.plot(epoch, mAPs[0], label='gamma={}'.format(gamma_list[0]))
plt.plot(epoch, mAPs[1], label='gamma={}'.format(gamma_list[1]))
plt.plot(epoch, mAPs[2], label='gamma={}'.format(gamma_list[2]))
plt.plot(epoch, mAPs[3], label='gamma={}'.format(gamma_list[3]))
plt.plot(epoch, mAPs[4], label='gamma={}'.format(gamma_list[4]))

plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)
plt.show()



###############
# result data #
###############
pr_ws = []
prses = []
one_errors = []
cov_losses = []
rank_losses = []

for gamma in gamma_list:
    te_result_file = './out_test_gm={}/iter_0/results_fold_0.json'.format(gamma)
    te_result_data = json.load(open(te_result_file))

    preds = te_result_data['prediction']
    golds = te_result_data['gold']
    pr_w = te_result_data['mAP']
    prs = average_precision_score(golds, preds, average=None)

    pr_ws.append(pr_w)
    prses.append(prs)

    preds = np.array(preds).T
    golds = np.array(golds).T
    
    one_err_list = one_error(golds, preds)
    one_err = np.nanmean(one_err_list)
    cov_loss = coverage_error(golds, preds)
    rank_loss = label_ranking_loss(golds, preds)

    one_errors.append(one_err)
    cov_losses.append(cov_loss)
    rank_losses.append(rank_loss)


pr_ws = np.array(pr_ws)
prses = np.array(prses)
print(pr_ws)
print(prses)
print(np.argmax(pr_ws))
print(np.argmax(prses, axis=0))

#print(one_errors)
#print(cov_losses)
#print(rank_losses)
