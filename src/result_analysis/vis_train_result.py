

import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
from numpy.lib.function_base import average
from sklearn.metrics import f1_score



parser = argparse.ArgumentParser()

parser.add_argument(
    '-dir', '--directory',
    help='file directory'
)
args = parser.parse_args()



result_file = '{}/iter_0/train_log_fold_0.json'.format(args.directory)

result_data = json.load(open(result_file))

train_losses = result_data['train_losses']
valid_losses = result_data['val_losses']
pr_averages = result_data['pr_averages']
roc_averages = result_data['roc_averages']


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


prediction = np.array(result_data['prediction'])
gold = np.array(result_data['gold'])

f1_micro, f1_macro = [], []
for prd, gld in zip(prediction, gold):
    f1_mi = f1_score(gld, np.round(prd), average='micro')
    f1_ma = f1_score(gld, np.round(prd), average='macro')
    f1_micro.append(f1_mi)
    f1_macro.append(f1_ma)
#print(f1_micro, f1_macro)

plt.plot(valid_losses, f1_micro, label="f1_micro")
plt.plot(valid_losses, f1_macro, label="f1_macro")
plt.show()
