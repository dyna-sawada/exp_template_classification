
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

from sklearn.model_selection import KFold

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


def train(model, tr_va_dataset, args, device, model_dir):
    #m = model
    #_tokenizer = m.get_tokenizer()
    #os.system("mkdir -p {}".format(args.model_dir))

    with open(os.path.join(args.model_dir, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    
    logging.info("Config: {}".format(json.dumps(args.__dict__, indent=1)))


    t_x = [tr_va_dataset[i][0] for i, _ in enumerate(tr_va_dataset)]
    kf = KFold(n_splits=args.fold_size)
    for fold, (train_index, valid_index) in enumerate(kf.split(t_x)):
        m = model
        train_dataset = Subset(tr_va_dataset, train_index)
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        valid_dataset   = Subset(tr_va_dataset, valid_index)
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False)

        logging.info("Fold: {} / {}".format(fold, args.fold_size))
        logging.info("Training on {} instances.".format(len(train_index)))
        logging.info("Validating on {} instances.".format(len(valid_index)))

        m.fit(
            train_dataloader,
            valid_dataloader,
            model_dir,
            fold
          )


def test(model, te_dataset, args, device, model_dir):
    # Prepare and evaluate the model!
    for fold_i in range(args.fold_size):
        m = model.from_pretrained(model_dir, args, device, fold_i)
        _tokenizer = m.get_tokenizer()

        t_x = [te_dataset[i][0] for i, _ in enumerate(te_dataset)]

        test_dataloader = DataLoader(te_dataset, args.batch_size, shuffle=False)
        logging.info("Testing on {} instances.".format(len(t_x)))

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

    #_lo_ids, _lo_speeches, _input_ids, _attention_masks, _labels = data_set.preprocess_dataset()
    dataset = data_set.make_tensor_dataset()

    
    for iter in range(args.iteration_size):

        model_dir = os.path.join(args.model_dir, 'iter_{}'.format(iter))
        os.system("mkdir -p {}".format(model_dir))

        logging.info("Iteration: {} / {}".format(iter, args.iteration_size))

        n_split = 1 - (1 / args.iteration_size)
        train_size = int(n_split * len(dataset))
        test_size = len(dataset) - train_size
        train_valid_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        train(model, train_valid_dataset, args, device, model_dir)
        test(model, test_dataset, args, device, model_dir)


        break        




if __name__ == '__main__':
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s- %(name)s - %(levelname)s - %(message)s'
            )

    args = parse_args()

    main(args)


