

import matplotlib.pyplot as plt
import json
import numpy as np





param_info_file = './out_test/params.json'
param_info = json.load(open(param_info_file))
n_fold = param_info['fold_size']
n_iter = param_info['iteration_size']


"""
for i in range(n_iter):
    test_loss, test_coverage = [], []
    for j in range(n_fold):
        test_result_file = './out_test/iter_{}/results_fold_{}.json'.format(i, j)
        test_result = json.load(open(test_result_file))
        loss = test_result['loss']
        coverage = test_result['coverage_error']
        
        test_loss.append(loss)
        test_coverage.append(coverage)

    print("Iter:{}\tLoss:{:.3f}\tCoverage:{:.3f}".format(
        i, np.mean(test_loss), np.mean(test_coverage)
        )
    )
"""


result_file = './out_test/iter_0/train_log_fold_0.json'

result_data = json.load(open(result_file))

train_losses = result_data['train_losses']
valid_losses = result_data['val_losses']
coverages = result_data['coverage_error']
#print(train_losses, valid_losses, coverages)

epoch = [e for e, _ in enumerate(train_losses)]
#print(epoch)


plt.plot(epoch, train_losses, label="train loss")
plt.plot(epoch, valid_losses, label="valid loss")
#plt.plot(epoch, coverages, label="coverage")

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

plt.show()


