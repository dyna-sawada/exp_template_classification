


import json
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report


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
    y_pred = np.random.randint(0, 2, (n_batch, n_label))
    
    micro_f1 = f1_score(y_pred, y_true, average='micro')
    macro_f1 = f1_score(y_pred, y_true, average='macro')

    print(classification_report(y_pred, y_true))
    print('MicroF1:{:.3f}\tMacroF1:{:.3f}'.format(micro_f1, macro_f1))
    



if __name__ == '__main__':
    main()



