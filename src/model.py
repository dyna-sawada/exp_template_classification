

import os
import json
import logging
from tqdm import tqdm
import pickle
import glob

import numpy as np

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import coverage_error

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup




class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, y_pred, y_true):
        cross_entropy_loss_fn = nn.BCELoss()
        cross_entropy_loss = cross_entropy_loss_fn(y_pred, y_true)
        p_t = ((y_true * y_pred) +
               ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return focal_cross_entropy_loss.mean()


class DummyLR():
    def __init__(self):
        pass
    def step(self):
        pass



class TorchTemplateClassifier(nn.Module):
    def __init__(self, args):
        super(TorchTemplateClassifier, self).__init__()

        self.args = args

        if args.encoder == 'bert-large':
            MODEL_NAME = 'bert-large-cased'
            self.hDim = 1024
        elif args.encoder == 'bert-base':
            MODEL_NAME = 'bert-base-cased'
            self.hDim = 768
        elif args.encoder == 'roberta-large':
            MODEL_NAME = 'roberta-large'
            self.hDim = 1024
        elif args.encoder == 'roberta-base':
            MODEL_NAME = 'roberta-base'
            self.hDim = 768

        
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.docenc = AutoModel.from_config(self.config)
        #self.docenc = AutoModel.from_pretrained(MODEL_NAME)
        self.fc1 = nn.Linear(self.hDim, self.hDim)
        self.fc2 = nn.Linear(self.hDim, 25)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()



    def forward(self, input_ids, attention_mask):
        outputs = self.docenc(input_ids, attention_mask)
        
        # get [CLS] embedding
        cls_out = outputs.last_hidden_state
        out = cls_out[:, 0, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return self.sig(out)



class TemplateClassifier():
    def __init__(self, args, device):
        self.args = args
        self.device =device
        self.classifier = TorchTemplateClassifier(self.args).to(self.device)

        if args.encoder == 'bert-large':
            self.tok = AutoTokenizer.from_pretrained('bert-large-cased')
        elif args.encoder == 'bert-base':
            self.tok = AutoTokenizer.from_pretrained('bert-base-cased')
        elif args.encoder == 'roberta-large':
            self.tok = AutoTokenizer.from_pretrained('roberta-large')
        elif args.encoder == 'roberta-base':
            self.tok = AutoTokenizer.from_pretrained('roberta-base')
        
        ## add new tokens to pre-trained model.
        sp_tokens = ['<PM>', '</PM>', '<LO>', '</LO>', '<FB>', '</FB>']
        self.tok.add_tokens(sp_tokens, special_tokens=True)
        self.classifier.docenc.resize_token_embeddings(len(self.tok))

        self.loss_fn = FocalLoss(gamma=2, alpha=None)
        #self.loss_fn = nn.BCELoss()

    
    @staticmethod
    def from_pretrained(fn_model_dir, args, device, fold_i):

        fn_model = os.path.join(fn_model_dir, "best_model_fold_{}.pt".format(fold_i))
        logging.info("Loading model from {}...".format(fn_model))

        m = TemplateClassifier(args, device)
        m.classifier.load_state_dict(torch.load(fn_model))
        m.classifier = m.classifier.to(device)

        return m


    def get_tokenizer(self):
        return self.tok


    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            fn_save_to_dir,
            fold
            ):
        self.classifier.train()

        if self.args.encoder_finetune:
            self.classifier.docenc.train()
        else:
            self.classifier.docenc.eval()

        if self.args.encoder_finetune:
            no_decay = ['bias', 'LayerNorm.weight']
            trainable_params = [
                {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            t_total = ((len(train_loader) + self.args.batch_size - 1) // self.args.batch_size) * self.args.epochs       # ここ用チェック！ len(xy_train)
            optimizer = AdamW(trainable_params, lr=self.args.learning_rate, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(t_total * 0.1),
                                                        num_training_steps=t_total)

        else:
            trainable_params = list(self.classifier.fc1.parameters()) + list(self.classifier.fc2.parameters())
            optimizer = optim.Adam(trainable_params,
                                   lr=self.args.learning_rate)
            scheduler = DummyLR()

        best_val_mse, best_model = 9999, None
        train_losses, val_losses = [], []
        micro_f1s, macro_f1s, coverages = [], [], []

        logging.info("Start training...")
        logging.info("Trainable parameters: {}".format(len(trainable_params)))

        for epoch in range(self.args.epochs):
            logging.info("Epoch {} / {}".format(1+epoch, self.args.epochs))

            train_loss = self.fit_epoch(train_loader, optimizer, scheduler)
            train_loss = train_loss * self.args.grad_accum

            with torch.no_grad():
                val_loss, y_val_pred, y_val_true = self.validate(valid_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logging.info("Loss Train: {:.3f} Valid: {:.3f}".format(train_loss, val_loss))

            coverage = coverage_error(y_val_true, y_val_pred)
            coverages.append(coverage)

            y_val_pred = np.where(y_val_pred >= 0.5, 1, 0)
            micro_f1 = f1_score(y_pred=y_val_pred, y_true=y_val_true, average='micro', zero_division=0)
            macro_f1 = f1_score(y_pred=y_val_pred, y_true=y_val_true, average='macro', zero_division=0)
            
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)

            logging.info(
                "Micro-F1. Valid: {:.3f}\tMacro-F1. Valid: {:.3f}\tCoverage Loss. Valid: {:.3f}".format(
                    micro_f1, macro_f1, coverage
                    )
                )


            if best_val_mse > val_loss:
                logging.info("Best validation loss!")
                best_model = pickle.dumps(self.classifier.state_dict())
                best_val_mse = val_loss


            # Writing to the log file.
            with open(os.path.join(
                                fn_save_to_dir, 
                                "train_log_fold_{}.json".format(fold)),
                                "w"
                                ) as f:

                train_log = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "micro_f1": micro_f1s,
                    "macro_f1": macro_f1s,
                    "coverage_error": coverages
                    #"best_epoch": epoch_i
                }

                json.dump(train_log, f, indent=2)

        if best_model is not None:
            logging.info("Saving the best model to {}...".format(fn_save_to_dir))
            torch.save(
                    pickle.loads(best_model), 
                    os.path.join(fn_save_to_dir, "best_model_fold_{}.pt".format(fold))
                    )


    def fit_epoch(self, train_loader: torch.utils.data.DataLoader, optimizer, scheduler):
        running_loss = []
        y_preds, y_trues = [], []
        grad_accum_steps = 0

        self.classifier.train()

        if self.args.encoder_finetune:
            self.classifier.docenc.train()
        else:
            self.classifier.docenc.eval()

        optimizer.zero_grad()

        for batch in tqdm(train_loader):
            input_id, attention_mask, y_true = (d.to(self.device) for d in batch)

            # Forward pass
            y_pred = self.classifier(input_id, attention_mask)
            loss = self.loss_fn(y_pred, y_true) / self.args.grad_accum

            y_preds.extend(y_pred.cpu().detach().numpy())
            y_trues.extend(y_true.cpu().detach().numpy())

            running_loss += [loss.item()]
            grad_accum_steps += 1

            # Backward pass
            loss.backward()

            if grad_accum_steps % self.args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        y_preds, y_trues = np.array(y_preds), np.array(y_trues)

        logging.info("Prediction sample:\n{}".format(y_preds))
        logging.info("Target sample:\n{}".format(y_trues.astype(np.int)))
        
        
        coverage = coverage_error(y_trues, y_preds)

        y_preds = np.where(y_preds >= 0.5, 1, 0)


        #logging.info("Acc. Train: {}".format(accuracy_score(y_trues, y_preds)))
        micro_f1_train = f1_score(y_pred=y_preds, y_true=y_trues, average='micro', zero_division=0)
        macro_f1_train = f1_score(y_pred=y_preds, y_true=y_trues, average='macro', zero_division=0)
        logging.info(
            "Micro-F1. Train: {:.3f}\tMacro-F1. Train: {:.3f}\tCoverage Loss. Train: {:.3f}".format(
                micro_f1_train, macro_f1_train, coverage
                )
            )

        return np.mean(running_loss)


    def validate(self, valid_loader: torch.utils.data.DataLoader):
        running_loss = []

        self.classifier.eval()
        self.classifier.docenc.eval()

        y_preds, y_trues = [], []

        for batch in tqdm(valid_loader):
            input_id, attention_mask, y_true = (d.to(self.device) for d in batch)

            y_pred = self.classifier(input_id, attention_mask)
            loss = self.loss_fn(y_pred, y_true)

            y_preds.extend(y_pred.cpu().detach().numpy())
            y_trues.extend(y_true.cpu().detach().numpy())

            running_loss += [loss.item()]

        return np.mean(running_loss), np.array(y_preds), np.array(y_trues)


    def test(self, test_loader, fn_save_to_dir, fold_i):
        logging.info("Start evaluation...")

        with torch.no_grad():
            test_loss, y_preds, y_trues = self.validate(test_loader)

        coverage = coverage_error(y_trues, y_preds)

        y_preds_01 = np.where(y_preds >= 0.5, 1, 0)
        micro_f1 = f1_score(y_pred=y_preds_01, y_true=y_trues, average='micro', zero_division=0)
        macro_f1 = f1_score(y_pred=y_preds_01, y_true=y_trues, average='macro', zero_division=0)

        result_log = {
            "prediction": y_preds.tolist(),
            "gold": y_trues.tolist(),
            "loss": test_loss,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "coverage_error": coverage
        }

        print(result_log)

        logging.info("Results are stored in {}/results.json.".format(fn_save_to_dir))

        with open(os.path.join(fn_save_to_dir, "results_fold_{}.json".format(fold_i)), "w") as f:
            json.dump(result_log, f, indent=2)


