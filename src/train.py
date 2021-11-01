
import os
import argparse
import json
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold, GroupKFold

from model import TemplateClassifier
from data import TemplateIdsDataset




def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-out', '--model-dir', required=True,
        help="Output directory."
    )
    
    parser.add_argument(
        '-enc', '--encoder', default='roberta-base',
        help='Encoder'
    )
    parser.add_argument(
        '-enc-ft', '--encoder_finetune', action='store_false',
        help='Fine-tuning'
    )

    parser.add_argument(
        '-it', '--iteration-size', default=5, type=int,
        help="Number of iterations.")
    parser.add_argument(
        '-fd', '--fold-size', default=5, type=int,
        help="K-fold.")
    parser.add_argument(
        '-ep', '--epochs', default=20, type=int,
        help="Max training epochs.")
    parser.add_argument(
        '-bs', '--batch-size', default=2, type=int,
        help="Training batch size.")
    parser.add_argument(
        '-lr', '--learning-rate', default=1e-6, type=float,
        help="Learning rate.")
    parser.add_argument(
        '-ga', '--grad-accum', default=16, type=int,
        help="Gradient accumulation steps.")
    

    return parser.parse_args()


def train(_model, tr_vl_dataset_info, args, device, model_dir):
    #m = model
    #_tokenizer = m.get_tokenizer()
    #os.system("mkdir -p {}".format(args.model_dir))

    with open(os.path.join(args.model_dir, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
            
    logging.info("Config: {}".format(json.dumps(args.__dict__, indent=1)))

    tr_vl_input_ids, tr_vl_attention_masks, tr_vl_labels, sub_lo_ids, tr_vl_data_ids, te_data_ids = \
        tr_vl_dataset_info[0], tr_vl_dataset_info[1], tr_vl_dataset_info[2], tr_vl_dataset_info[3], tr_vl_dataset_info[4], tr_vl_dataset_info[5]


    fold_i = 0
    GKF_fold = GroupKFold(n_splits=args.fold_size).split(tr_vl_input_ids, groups=sub_lo_ids)
    for tr_index, vl_index in GKF_fold:

        tr_data_ids, vl_data_ids = tr_vl_data_ids[tr_index], tr_vl_data_ids[vl_index]
        tr_input_ids, vl_input_ids = tr_vl_input_ids[tr_index], tr_vl_input_ids[vl_index]
        tr_attention_masks, vl_attention_masks = tr_vl_attention_masks[tr_index], tr_vl_attention_masks[vl_index]
        tr_labels, vl_labels = tr_vl_labels[tr_index], tr_vl_labels[vl_index]
        tr_dataset = TensorDataset(tr_input_ids, tr_attention_masks, tr_labels)
        vl_dataset = TensorDataset(vl_input_ids, vl_attention_masks, vl_labels)

        tr_dataloader = DataLoader(tr_dataset, args.batch_size, shuffle=True)
        vl_dataloader = DataLoader(vl_dataset, args.batch_size, shuffle=True)

        m = TemplateClassifier(args, device)

        tr_data_ids, vl_data_ids, te_data_ids = \
            tr_data_ids.tolist(), vl_data_ids.tolist(), te_data_ids.tolist()

        data_split_info = {
                            'train':
                            {
                                'length':len(tr_data_ids),
                                'data_ids':tr_data_ids
                            },
                            'valid':
                            {
                                'length':len(vl_data_ids),
                                'data_ids':vl_data_ids
                            },
                            'test':
                            {
                                'length':len(te_data_ids),
                                'data_ids':te_data_ids
                            }
                        }

        assert len(tr_data_ids) + len(vl_data_ids) + len(te_data_ids) == 567

        with open(os.path.join(model_dir, "data_split_fold_{}.json".format(fold_i)), "w") as f:
            json.dump(data_split_info, f, indent=2)


        logging.info("Fold: {} / {}".format(fold_i, args.fold_size))
        logging.info("Training on {} instances.".format(tr_labels.size()[0]))
        logging.info("Validating on {} instances.".format(vl_labels.size()[0]))
        logging.info("Saving training instance ids to {}.".format(model_dir))
 

        m.fit(
            tr_dataloader,
            vl_dataloader,
            model_dir,
            fold_i
        )

        fold_i += 1



def test(model, te_dataset_info, args, device, model_dir):
    ## Prepare and evaluate the model!
    te_input_ids, te_attention_masks, te_labels = \
        te_dataset_info[0], te_dataset_info[1], te_dataset_info[2]
    te_dataset = TensorDataset(te_input_ids, te_attention_masks, te_labels)
    test_dataloader = DataLoader(te_dataset, args.batch_size, shuffle=False)

    for fold_i in range(args.fold_size):
        m = model.from_pretrained(model_dir, args, device, fold_i)
        #_tokenizer = m.get_tokenizer()

        logging.info("Testing on {} instances.".format(te_labels.size()[0]))

        m.test(
            test_dataloader,
            model_dir,
            fold_i
            )



def main(args):
    
    np.set_printoptions(precision=3, suppress=False)

    MAX_SEQ_LEN = 512
    os.system("mkdir -p {}".format(args.model_dir))

    ## GPU setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    temp_id_data = json.load(open('./work/temp_id_gold.json'))

    model = TemplateClassifier(args, device)
    tokenizer =  model.get_tokenizer()


    logging.info("Loading & Setting up Dataset...")
    data_set = TemplateIdsDataset(
            temp_id_data,
            tokenizer,
            MAX_SEQ_LEN
        )

    data_ids, lo_ids, _lo_speeches, input_ids, attention_masks, labels = \
        data_set.preprocess_dataset()
    
    
    iter_i = 0
    GKF_iter = GroupKFold(n_splits=args.iteration_size).split(input_ids, groups=lo_ids)
    for tr_vl_index, te_index in GKF_iter:
        
        model_dir = os.path.join(args.model_dir, 'iter_{}'.format(iter_i))
        os.system("mkdir -p {}".format(model_dir))

        logging.info("Iteration: {} / {}".format(iter_i, args.iteration_size))

        tr_vl_data_ids, te_data_ids = data_ids[tr_vl_index], data_ids[te_index]
        tr_vl_input_ids, te_input_ids = input_ids[tr_vl_index], input_ids[te_index]
        tr_vl_attention_masks, te_attention_masks = attention_masks[tr_vl_index], attention_masks[te_index]
        tr_vl_labels, te_labels = labels[tr_vl_index], labels[te_index]
        sub_lo_ids = lo_ids[tr_vl_index]
        tr_vl_dataset_info = (
                                tr_vl_input_ids,
                                tr_vl_attention_masks,
                                tr_vl_labels,
                                sub_lo_ids,
                                tr_vl_data_ids,
                                te_data_ids
                            )
        te_dataset_info = (
                                te_input_ids,
                                te_attention_masks,
                                te_labels
                            )

        
        train(model, tr_vl_dataset_info, args, device, model_dir)
        test(model, te_dataset_info, args, device, model_dir)
        
        iter_i += 1




if __name__ == '__main__':
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s- %(name)s - %(levelname)s - %(message)s'
            )

    args = parse_args()

    main(args)


