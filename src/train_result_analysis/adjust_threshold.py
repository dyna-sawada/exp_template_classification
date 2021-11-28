

import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc



fold_i = 0
result_file = './out_test/iter_0/results_fold_{}.json'.format(fold_i)

result_data = json.load(open(result_file))

preds = result_data['prediction']
trues = result_data['gold']

preds = np.array(preds)
trues = np.array(trues)



f1_micro = f1_score(trues, preds.round(), average='micro')
f1_macro = f1_score(trues, preds.round(), average='macro')
f1_sample = f1_score(trues, preds.round(), average='samples')



print(
    'micro: {:.1f}%\tmacro: {:.1f}%\tsample: {:.1f}%'.format(
        f1_micro*100, f1_macro*100, f1_sample*100
    )
)





for i in range(25):
    preds_sample_i = preds[:, i]
    trues_sample_i = trues[:, i]
    print(preds_sample_i, trues_sample_i)

    assert len(preds_sample_i) == len(trues_sample_i)

    fpr, tpr, thresholds = roc_curve(trues_sample_i, preds_sample_i)

    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.show()
    plt.savefig(
        './out_test/iter_0/roc_curve_fold_{}_temp_{}.png'.format(
            fold_i, i
        )
    )



for i in range(25):
    preds_sample_i = preds[:, i]
    trues_sample_i = trues[:, i]
    print(preds_sample_i, trues_sample_i)

    assert len(preds_sample_i) == len(trues_sample_i)

    precision, recall, thresholds = precision_recall_curve(trues_sample_i, preds_sample_i)

    plt.plot(recall, precision, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.show()
    plt.savefig(
        './out_test/iter_0/pr_curve_fold_{}_temp_{}.png'.format(
            fold_i, i
        )
    )




