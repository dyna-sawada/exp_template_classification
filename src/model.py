

import os
import json
import logging
from tqdm import tqdm
import pickle
import glob

import numpy as np

#from sklearn.metrics import recall_score, precision_score, f1_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import coverage_error, roc_auc_score, average_precision_score, label_ranking_loss
from evaluation_metrics import f1_threshold_score
from evaluation_metrics import pr_auc_scores_average, roc_auc_scores_average
from evaluation_metrics import one_error_score, coverage_score, ranking_loss_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, y_pred, y_true):
        
        ## version1.0
        cross_entropy_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        cross_entropy_loss = cross_entropy_loss_fn(y_pred, y_true)
        y_pred = torch.sigmoid(y_pred)
        p_t = torch.where(y_true >= 0.5, y_pred, 1-y_pred)
        
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)

        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)

        focal_cross_entropy_loss = focal_cross_entropy_loss.mean()        
        """
        ## version2.0
        l = y_pred.reshape(-1)
        t = y_true.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        focal_cross_entropy_loss = logp * ((1-p)**self._gamma)
        focal_cross_entropy_loss = 25*focal_cross_entropy_loss.mean()
        """
        return focal_cross_entropy_loss



class DummyLR():
    def __init__(self):
        pass
    def step(self):
        pass



class TorchTemplateClassifier(nn.Module):
    def __init__(self, args, device):
        super(TorchTemplateClassifier, self).__init__()

        self.args = args
        self.device = device

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
        self.fc1 = nn.Linear(self.hDim, 25)
        self.fc2 = nn.Linear(25, 25)
        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()



    def forward(self, input_id, attention_mask, *sp_token_positions):
        outputs = self.docenc(input_id, attention_mask)
        all_emb = outputs.last_hidden_state
        
        if self.args.encoder_out == 'cls':
            ## get [CLS] embedding
            #out = all_emb[:, 0, :]
            out = torch.mean(all_emb, 1)
        
        elif self.args.encoder_out == 'fb':
            ## get [FB] embedding
            sp_token_positions = sp_token_positions[0]
            n_batch, _n_seq_length, n_hidden = all_emb.size()[0], all_emb.size()[1], all_emb.size()[2]
            fb_outs = torch.empty(0, n_hidden).to(self.device)
            for i, stp in enumerate(sp_token_positions):
                fb_out = torch.empty(0, n_hidden).to(self.device)
                for s in stp:
                    if s[0] == 0 and s[1] == 0:
                        break
                    fb_emb = all_emb[i, s[0]:s[1], :]
                    fb_out = torch.cat((fb_out, fb_emb), dim=0)
                    
                    #print(all_emb[i, :, :])

                fb_out = torch.mean(fb_out, 0)
                fb_out = fb_out.unsqueeze(0)
                fb_outs = torch.cat((fb_outs, fb_out), 0)
                
        
            assert fb_outs.size()[0] == n_batch
            out = fb_outs

        #out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.sig(out)

        return out



class TemplateClassifier():
    def __init__(self, args, device):
        self.args = args
        self.device =device
        self.classifier = TorchTemplateClassifier(self.args, self.device).to(self.device)

        if args.encoder == 'bert-large':
            self.tok = AutoTokenizer.from_pretrained('bert-large-cased')
        elif args.encoder == 'bert-base':
            self.tok = AutoTokenizer.from_pretrained('bert-base-cased')
        elif args.encoder == 'roberta-large':
            self.tok = AutoTokenizer.from_pretrained('roberta-large')
        elif args.encoder == 'roberta-base':
            self.tok = AutoTokenizer.from_pretrained('roberta-base')
        
        ## add new tokens to pre-trained model.
        sp_tokens = [
                '<PM>', '</PM>',
                '<LO>', '</LO>',
                '<FB>', '</FB>',
                '<CLAIM>',
                '<PREMISE>',
                '<EXAMPLE>',
                '<STANCE>'
            ]
        self.tok.add_tokens(sp_tokens, special_tokens=True)
        self.classifier.docenc.resize_token_embeddings(len(self.tok))

        if self.args.loss_fn == 'focal_loss':
            self.loss_fn = FocalLoss(gamma=self.args.gamma, alpha=self.args.alpha)
        elif self.args.loss_fn == 'BCE_loss':
            self.loss_fn = nn.BCEWithLogitsLoss()
        #self.sig = nn.Sigmoid()

    
    @staticmethod
    def from_pretrained(fn_model_dir, args, device, fold_i):

        fn_model = os.path.join(fn_model_dir, "best_model_fold_{}.pt".format(fold_i))
        logging.info("Loading model from {}...".format(fn_model))

        m = TemplateClassifier(args, device)
        if torch.cuda.is_available():
            m.classifier.load_state_dict(torch.load(fn_model))
        else:
            m.classifier.load_state_dict(torch.load(fn_model, map_location=torch.device('cpu')))
        
        m.classifier = m.classifier.to(device)

        return m


    def get_tokenizer(self):
        return self.tok


    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            fn_save_to_dir,
            fold_i
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

            t_total = ((len(train_loader) + self.args.batch_size - 1) // self.args.batch_size) * self.args.epochs       # check
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
        y_val_preds, y_val_trues = [], []
        #_f1_micros, _f1_macros, coverages = [], [], []
        pr_averages, roc_averages = [], []

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
            y_val_preds.append(y_val_pred.tolist())
            y_val_trues.append(y_val_true.tolist())

            logging.info("Loss Train: {:.3f} Valid: {:.3f}".format(train_loss, val_loss))


            #coverage = coverage_score(y_val_true, y_val_pred)
            #coverages.append(coverage)

            _pr_scores, pr_average = pr_auc_scores_average(y_val_true, y_val_pred, average='weighted')
            _roc_scores, roc_average = roc_auc_scores_average(y_val_true, y_val_pred, average='weighted')
            
            pr_averages.append(pr_average)
            roc_averages.append(roc_average)


            logging.info(
                "Valid\tPR AUC score: {:.3f}\tROC AUC score: {:.3f}".format(
                    pr_average, roc_average
                    )
                )


            if best_val_mse > val_loss:
                logging.info("Best validation loss!")
                best_model = pickle.dumps(self.classifier.state_dict())
                best_val_mse = val_loss


            # Writing to the log file.
            with open(os.path.join(
                                fn_save_to_dir, 
                                "train_log_fold_{}.json".format(fold_i)),
                                "w"
                                ) as f:

                train_log = {
                    "prediction": y_val_preds,
                    "gold": y_val_trues,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    #"f1_micro": f1_micros,
                    #"f1_macro": f1_macros,
                    "pr_averages": pr_averages,
                    "roc_averages": roc_averages
                    #"coverage_error": coverages
                    #"best_epoch": epoch_i
                }

                logging.info("Updating train log file.")
                json.dump(train_log, f, indent=2)

        if best_model is not None:
            logging.info("Saving the best model to {}/best_model_fold_{}.".format(fn_save_to_dir, fold_i))
            torch.save(
                    pickle.loads(best_model), 
                    os.path.join(fn_save_to_dir, "best_model_fold_{}.pt".format(fold_i))
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

        #torch.save(train_loader, './out_test/train_loader_{}.pt'.format(self.args.encoder_out))
        for batch in tqdm(train_loader):
            if self.args.encoder_out == 'cls':
                input_id, attention_mask, y_true = (d.to(self.device) for d in batch)
                y_pred = self.classifier(input_id, attention_mask)
            elif self.args.encoder_out == 'fb':
                input_id, attention_mask, sp_token_position, y_true = (d.to(self.device) for d in batch)
                y_pred = self.classifier(input_id, attention_mask, sp_token_position)

            loss = self.loss_fn(y_pred, y_true) / self.args.grad_accum

            y_pred = torch.sigmoid(y_pred)
            y_preds.extend(y_pred.cpu().detach().numpy())
            y_trues.extend(y_true.cpu().detach().numpy())

            running_loss += [loss.item()]
            grad_accum_steps += 1

            loss.backward()

            if grad_accum_steps % self.args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        y_preds, y_trues = np.array(y_preds), np.array(y_trues)

        logging.info("Prediction sample:\n{}".format(y_preds))
        logging.info("Target sample:\n{}".format(y_trues.astype(np.int)))
        
        
        #coverage = coverage_score(y_trues, y_preds)
        _pr_scores, pr_average = pr_auc_scores_average(y_trues, y_preds, average='weighted')
        _roc_scores, roc_average = roc_auc_scores_average(y_trues, y_preds, average='weighted')

        logging.info(
            "Train\tPR AUC score: {:.3f}\tROC AUC score: {:.3f}".format(
                pr_average, roc_average
                )
            )
        

        return np.mean(running_loss)


    def validate(self, valid_loader: torch.utils.data.DataLoader):
        running_loss = []

        self.classifier.eval()
        self.classifier.docenc.eval()

        y_preds, y_trues = [], []

        for batch in tqdm(valid_loader):
            if self.args.encoder_out == 'cls':
                input_id, attention_mask, y_true = (d.to(self.device) for d in batch)
                y_pred = self.classifier(input_id, attention_mask)
            elif self.args.encoder_out == 'fb':
                input_id, attention_mask, sp_token_position, y_true = (d.to(self.device) for d in batch)
                y_pred = self.classifier(input_id, attention_mask, sp_token_position)

            loss = self.loss_fn(y_pred, y_true)

            y_pred = torch.sigmoid(y_pred)
            y_preds.extend(y_pred.cpu().detach().numpy())
            y_trues.extend(y_true.cpu().detach().numpy())

            running_loss += [loss.item()]

        return np.mean(running_loss), np.array(y_preds), np.array(y_trues)


    def test(self, test_loader, fn_save_to_dir, fold_i):
        logging.info("Start evaluation...")

        with torch.no_grad():
            test_loss, y_preds, y_trues = self.validate(test_loader)

        pr_scores, pr_average = pr_auc_scores_average(y_trues, y_preds, average='weighted')
        roc_scores, roc_average = roc_auc_scores_average(y_trues, y_preds, average='weighted')
        pr_result = {
            "scores": pr_scores.tolist(),
            "average": pr_average
        }
        roc_result = {
            "scores": roc_scores.tolist(),
            "average": roc_average
        }

        _one_err_scores, one_err_average = one_error_score(y_trues, y_preds) 
        coverage = coverage_score(y_trues, y_preds)
        rank_loss = ranking_loss_score(y_trues, y_preds)

        result_log = {
            "prediction": y_preds.tolist(),
            "gold": y_trues.tolist(),
            "loss": test_loss,
            #"micro_f1": micro_f1,
            #"macro_f1": macro_f1,
            "PR": pr_result,
            "ROC": roc_result,
            "one_error": one_err_average,
            "coverage_error": coverage,
            "rank_loss": rank_loss,
        }

        print(result_log)

        logging.info("Results are stored in {}/results.json.".format(fn_save_to_dir))

        with open(os.path.join(fn_save_to_dir, "results_fold_{}.json".format(fold_i)), "w") as f:
            json.dump(result_log, f, indent=2)


