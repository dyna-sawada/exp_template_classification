


import json
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import coverage_error


def main():
    temp_id_gold_dir = './work/temp_id_gold.json'
    temp_id_gold = json.load(open(temp_id_gold_dir))

    y_true = [] 
    for _lo_id, lo_id_dict in temp_id_gold.items():
        for _fb_unit_id, temp_data_dict in lo_id_dict['temp_data'].items():
            y_t = temp_data_dict['temp_id']
            y_true.append(y_t)
    y_true = np.array(y_true)
    
    print("data size: {}".format(y_true.shape))

    n_batch = y_true.shape[0]
    n_label = y_true.shape[1]

    y_pred_logits = torch.randn(n_batch, n_label)
    y_pred = torch.sigmoid(y_pred_logits)
    y_pred = y_pred.detach().numpy().copy()

    coverage = coverage_error(y_true, y_pred)

    y_pred_01 = np.where(y_pred >= 0.5, 1, 0)
    micro_f1 = f1_score(y_pred_01, y_true, average='micro')
    macro_f1 = f1_score(y_pred_01, y_true, average='macro')

    print(
        'MicroF1:{:.3f}\tMacroF1:{:.3f}\tCoverageLoss:{:.3f}'.format(
            micro_f1, macro_f1, coverage
            )
        )
    



if __name__ == '__main__':
    main()



