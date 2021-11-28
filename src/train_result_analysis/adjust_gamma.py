
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import f1_score


gamma_list = [0.5, 1, 2, 5]
losses = []
probs = []

for gamma in gamma_list:
    result_file = './out_test_gm={}/iter_0/train_log_fold_0.json'.format(gamma)

    result_data = json.load(open(result_file))

    y_val_preds = result_data['prediction']
    y_val_trues = result_data['gold']
    
    #train_losses = result_data['train_losses']
    valid_losses = result_data['val_losses']
    
    #coverages = result_data['coverage_error']
    #mAP = result_data['mAP']
    #roc_auc = result_data['ROC_AUC']
    
    for y_val_pred, y_val_true in zip(y_val_preds, y_val_trues):
        np_y_val_pred = np.array(y_val_pred)
        np_y_val_true = np.array(y_val_true)
        gold_index = np.where(np_y_val_true==1)
        prob = np.average(np_y_val_pred[gold_index])
        probs.append(prob)


    losses = valid_losses
    break


print(losses)
print(probs)

plt.plot(probs, losses)


#for i, (loss, prob) in enumerate(zip(losses, probs)):
#    plt.plot(loss[i], prob[i], label="gamma={}".format(gamma_list[i]))

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)
plt.show()
