

import matplotlib.pyplot as plt
import json
import numpy as np





param_info_file = './out_test/params.json'
param_info = json.load(open(param_info_file))
n_fold = param_info['fold_size']
n_iter = param_info['iteration_size']



for i in range(n_iter):
    test_loss, test_one_err, test_coverage, test_rank_loss,\
         test_mAP_m, test_mAP_w, test_mAP_s = [], [], [], [], [], [], []
    for j in range(n_fold):
        test_result_file = './out_test/iter_{}/results_fold_{}.json'.format(i, j)
        test_result = json.load(open(test_result_file))
        loss = test_result['loss']
        one_err = test_result['one_error']
        coverage = test_result['coverage_error']
        rank_loss = test_result['rank_loss']
        mAPs = test_result['mAP']
        
        test_loss.append(loss)
        test_one_err.append(one_err)
        test_coverage.append(coverage)
        test_rank_loss.append(rank_loss)
        test_mAP_m.append(mAPs[0])
        test_mAP_w.append(mAPs[1])
        test_mAP_s.append(mAPs[2])

    print(
        'Iter: {}\tLoss: {:.3f}\t \
        OneError: {:.3f}\tCoverageLoss: {:.3f}\t \
        RankingLoss: {:.3f}\tmAP micro: {:.3f}\t \
        mAP weighted: {:.3f}\tmAP samples: {:.3f}'.format(
            i, np.mean(test_loss), np.mean(test_one_err), np.mean(test_coverage),
            np.mean(rank_loss), np.mean(test_mAP_m), np.mean(test_mAP_w), np.mean(test_mAP_s)
        )
    )
    break


"""
result_file = './out_test/iter_0/train_log_fold_0.json'

result_data = json.load(open(result_file))

train_losses = result_data['train_losses']
valid_losses = result_data['val_losses']
coverages = result_data['coverage_error']
mAPs = result_data['mAP']
mAP_micro = [m[0] for m in mAPs]
mAP_weight = [m[1] for m in mAPs]
mAP_sample = [m[2] for m in mAPs]
#print(train_losses, valid_losses, coverages)

epoch = [e for e, _ in enumerate(train_losses)]
#print(epoch)
print(len(epoch))

#plt.plot(epoch, train_losses, label="train loss")
#plt.plot(epoch, valid_losses, label="valid loss")
#plt.plot(epoch, coverages, label="coverage")
plt.plot(epoch, mAP_micro, label="mAP_micro")
plt.plot(epoch, mAP_weight, label="mAP_weighted")
plt.plot(epoch, mAP_sample, label="mAP_sample")

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

plt.show()
"""

