

import argparse
from os import openpty
import nltk
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import json
import glob
import random

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Subset
import torch.nn.functional as F

import tensorflow as tf

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import coverage_error
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, precision_recall_curve
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score

from transformers import AutoTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel
#from data import TemplateIdsDataset
from model import TorchTemplateClassifier





"""
a = "They said that the death penalty is inhumane because it requires a person to kill another person. However, technically speaking, executions do not need to be carried out by human beings. It is quite possible to devise a method of execution that is fully automated, so no person has to deliver the deadly punishment directly. Even if there were something in the current legal code that would prevent such an automated execution, there is no reason that the law surrounding capital punishment couldn't be rewritten to keep up with technology. Although executions at this time are still carried out using human executioners, methods are already put in place to prevent the perception that one particular person has put another to death. For instance, multiple prison officials simultaneously pull multiple levers, not knowing which lever results in the delivery of an electrical charge, in the case of death by electrocution. Thus, the death penalty can't be said to be inhumane for the stated reason."
b = nltk.sent_tokenize(a)
print(b)
c = {
      "adus": {
        "adu0": {
          "adu_type": '',
          "sent_idx": [
            0
          ]
        },
        "adu1": {
          "adu_type": "Claim",
          "sent_idx": [
            1
          ]
        },
        "adu2": {
          "adu_type": "Premise",
          "sent_idx": [
            2,
            3
          ]
        },
        "adu3": {
          "adu_type": "Premise",
          "sent_idx": [
            4
          ]
        },
        "adu4": {
          "adu_type": "Example",
          "sent_idx": [
            5
          ]
        },
        "adu5": {
          "adu_type": "Stance",
          "sent_idx": [
            6
          ]
        }
      }
    }

ref_id = [0,1]
ref_info_list = [0] * len(b)
for r_id in ref_id:
    ref_info_list[r_id] = 1
print(ref_info_list)

adu_info_list = [''] * len(b)
for d in c['adus'].values():
    adu_type = d['adu_type']
    sent_idx = d['sent_idx']
    for s_idx in sent_idx:
        adu_info_list[s_idx] = adu_type
print(adu_info_list)

for i, _sent in enumerate(b):
    if adu_info_list[i] == 'Claim':
        b[i] = '</CLAIM>' + b[i] + ' </CLAIM>'
print(b)

x = b
for i, _sent in enumerate(b):
    if ref_info_list[i] == 1:
        x[i] = 'hoge' + b[i] + '/hoge'
print(x)


"""

"""
cls_dataloader = torch.load('./out_test/train_loader_cls.pt')
fb_dataloader = torch.load('./out_test/train_loader_fb.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained('roberta-base')
tok = AutoTokenizer.from_pretrained('roberta-base')
docenc = AutoModel.from_pretrained('roberta-base')
sp_tokens = ['<PM>', '</PM>', '<LO>', '</LO>', '<FB>', '</FB>']
tok.add_tokens(sp_tokens, special_tokens=True)
docenc.resize_token_embeddings(len(tok))
docenc.to(device)


for i, (c, f) in enumerate(zip(cls_dataloader, fb_dataloader)):

    assert torch.equal(c[0], f[0])
    assert torch.equal(c[1], f[1])
    assert torch.equal(c[2], f[3])

    c_outputs = docenc(c[0].to(device), c[1].to(device))
    c_outputs_2 = docenc(c[0].to(device), c[1].to(device))
    f_outputs = docenc(f[0].to(device), f[1].to(device))
    c_all_emb = c_outputs.last_hidden_state
    c_all_emb_2 = c_outputs_2.last_hidden_state
    f_all_emb = f_outputs.last_hidden_state
    print(c_all_emb)
    print(f_all_emb)

    assert torch.equal(c_all_emb, f_all_emb)
"""

"""
a = [(0,1), (2,3), (4,5)]
b = torch.tensor(a)
#print(b)

c = [(0,1)]
d = torch.tensor(c)
#print(d)

#print(b.unsqueeze(0).size())
#print(d.unsqueeze(0).size())

n_batch = 2
n_seq_length = 512
n_hidden = 768

e = torch.rand(n_batch, n_seq_length, n_hidden)
print(e.size())
cls_emb = e[:, 0, :]
print(cls_emb)
print(cls_emb.size())

index_position = torch.tensor(
    [
        [
            [102, 110],
            [150, 160],
            [180, 190],
            [0, 0],
            [0, 0]
        ],
        [
            [99, 100],
            [120, 135],
            [160, 167],
            [180, 188],
            [0, 0]
        ]
    ]
)

print("---")


fb_emb = torch.empty(0, n_hidden)

for b, postions in enumerate(index_position):
    fb_emb_ = torch.empty(0, n_hidden)
    
    for p in postions:
        if p[0] == 0 and p[1] == 0:
            break

        emb = e[b, p[0]:p[1], :]
        print(emb)
        print(emb.size())
        fb_emb_ = torch.cat((fb_emb_, emb), dim=0)

    print(fb_emb_)
    print(fb_emb_.size())
    fb_emb_ = torch.mean(fb_emb_, 0)
    fb_emb_ = fb_emb_.unsqueeze(0)
    print(fb_emb_)
    print(fb_emb_.size())
    fb_emb = torch.cat((fb_emb, fb_emb_), 0)


print(fb_emb)
print(fb_emb.size())
"""

"""
y_true = np.array(
                [
                    [0,0,1,1],
                    [0,0,1,0],
                    [0,1,0,0]
                ]
                )

y_pred = np.array(
                [
                    [0.1, 0.4, 0.35, 0.6],
                    [0.2, 0.3, 0.2, 0.9],
                    [0.6, 0.4, 0.1, 0.4]
                ]
                )

for y_t, y_p in zip(y_true, y_pred):
    pr, rc, th = precision_recall_curve(y_t, y_p)
    print(pr)
    print(rc)
    print(th)


#y_true = torch.load('y_val_true_0_0.pt')
#y_pred = torch.load('y_val_pred_0_0.pt')

#np.set_printoptions(threshold=np.inf)
#print(y_true)
map_0 = average_precision_score(y_true, y_pred, average='micro')
map_1 = average_precision_score(y_true, y_pred, average='macro')
map_2 = average_precision_score(y_true, y_pred, average='weighted')
map_3 = average_precision_score(y_true, y_pred, average='samples')
print(map_0)
print(map_1)
print(map_2)
print(map_3)
print(roc_auc_score(y_true, y_pred, average='micro'))

#auc = roc_auc_score(y_true, y_scores, average='micro')
#print(auc)
#lg_loss = log_loss(y_true, y_scores)
#print(lg_loss)
"""

"""
sort_id = np.arange(3)
print(sort_id)
np.random.shuffle(sort_id)
print(sort_id)
y_true_r = y_true[sort_id]
y_scores_r = y_scores[sort_id]
print(y_true_r)
print(y_scores_r)
"""

"""
group = ['DP_LO_2_11', 'DP_LO_2_24', 'DP_LO_2_11', 'DP_LO_2_11', 'DP_LO_2_11',
         'DP_LO_2_11', 'DP_LO_2_18', 'DP_LO_2_18', 'DP_LO_2_18', 'DP_LO_2_18',
         'DP_LO_2_24', 'DP_LO_2_24', 'DP_LO_2_11']
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]
y = ['a', 'a', 'b', 'c', 'a', 'c', 'b', 'b', 'c', 'a', 'b', 'b', 'c']



temp_id_gold = json.load(open('./work/temp_id_gold.json'))
lo_ids = list(temp_id_gold.keys())
group_id = [lo_ids.index(a) for a in group]
group_id = np.array(group_id)

gkf = GroupKFold(n_splits=3).split(x, y, groups=group_id)
for tr, te in gkf:
    print("%s %s" % (tr, te))
    print(group_id[tr])

#tensor_group = torch.tensor(group)
#print(tensor_group)

a = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)
index = [0, 1]
print(a[index])

"""

"""
全てのデータを事前準備（主に，tokenize&encode）
lo_ids, input_ids, mask_ids, label_ids (tensor) をそれぞれ入手
　lo_ids を int に変換．dictを使用する．

GKFの作成． 
GKF = GroupKFoldn_split=args.iter_size).split(input_ids, groups=lo_ids)

index の取得
for tr_vl_index, te_index in GKF:
    input_ids / mask_ids / label_ids / group_ids

→ dataset, dataloaderの作成


それぞれの情報を取得
tr_val_input_ids, test_input_ids = input_ids[tr_val_index], input_ids[te_index]
tr_val_mask_ids, test_mask_ids = mask_ids[tr_val_index], mask_ids[te_index]
tr_val_label_ids, test_label_ids = label_ids[tr_val_index], input_ids[te_index]
tr_val_group_ids = group_ids[tr_val_index]

tr_val_dataset = torch.dataset(tr_val_input_ids, tr_val_mask_ids, tr_val_label_ids)
te_dataset = torch.dataset(...)

"""



"""
temp_id_gold = json.load(open('./work/temp_id_gold.json'))
len_ids = len(temp_id_gold)
split_id = int(len_ids / 5 * 4)
lo_ids = list(temp_id_gold.keys())
print(len_ids)
print(lo_ids)
random.shuffle(lo_ids)
print()
#print(lo_ids)
lo_ids_tr = lo_ids[:split_id]
lo_ids_te = lo_ids[split_id:]
print(len(lo_ids_tr), len(lo_ids_te))
print(lo_ids_tr)
print(lo_ids_te)
assert len(lo_ids_tr) + len(lo_ids_te) == len_ids
"""

"""
class FocalLoss():
    def __init__(self, gamma=2, alpha=0.25):
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, y_pred, y_true):
        cross_entropy_loss_fn = torch.nn.BCELoss()
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


class FocalLoss_(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss_, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.float()

        # BCELossWithLogits
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


target = np.array(
        [[1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0,],
        [0, 0, 0, 0, 0, 1]]
        )
#target = target.to(torch.float)

sample = np.array(
        [[0.5068503,  0.4909574,  0.48088843, 0.56281924, 0.5019796,  0.55661233],
         [0.5004162,  0.5388744,  0.5154122, 0.6002497,  0.5186469,  0.52400404],
         [0.5173367,  0.53379077, 0.48255765, 0.57839257, 0.49632147, 0.5143831 ],
         [0.5200508,  0.540166,   0.4624398, 0.5662759,  0.5013316,  0.55133134]]
        )

loss = coverage_error(target, sample)
print(loss)




loss_fn = FocalLoss_()

target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
output = torch.full([10, 64], 1.5)  # A prediction (logit)
pos_weight = torch.ones([64])  # All weights are equal to 1
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(output, target)  # -log(sigmoid(1.5))
print(target.shape, output.shape)



te_data = json.load(open('./work/temp_id_gold.json'))

ref_ids = [1,2]
MAX_SEQ_LEN = 512

for lo_id, _text_label_dict in te_data.items():
    pm_speech = te_data[lo_id]['pm_speech']
    lo_speech = te_data[lo_id]['lo_speech']
    for _fb_unit_id, temp_data_dict in _text_label_dict['temp_data'].items():
        ref_id = temp_data_dict['ref_id']
        label = temp_data_dict['temp_id']

        print(ref_id)
        print(label)



# tokenizer setting
sp_tokens = ['<PM>', '</PM>', '<LO>', '</LO>', '<FB>', '</FB>']
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer.add_tokens(sp_tokens, special_tokens=True)

lo_texts = lo_speech.split('.')
lo_speech = ''
for i, lo_text in enumerate(lo_texts):
    if i == len(lo_texts) - 1:
        break

    if i in ref_ids:
        lo_text = ' <FB>' + lo_text + '. </FB>'
    else:
        lo_text = lo_text + '.'
    print("{}\t{}\n".format(i, lo_text))
    lo_speech += lo_text

lo_speech = ' <LO> ' + lo_speech + ' </LO>'
#print(lo_speech)


pm_texts = pm_speech.split('.')

i = 0
tokens = ['t'] * 512
while len(tokens) >= 510:
    speeches = '<PM> ' + '.'.join(pm_texts[i:]) + ' </PM>' + lo_speech
    tokens = tokenizer.tokenize(speeches)
    i += 1

print("iteration:{}\nn_tokens:{}\nspeeches:{}".format(i-1, len(tokens), speeches))

speech_ids = tokenizer.encode_plus(
                                speeches,
                                add_special_tokens=True,
                                max_length=MAX_SEQ_LEN,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt'
                                )
print(speech_ids)
print(speech_ids['input_ids'].size())
"""


"""
y_val_true = np.array(
        [[1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0,],
        [0, 0, 0, 0, 0, 1]]
        )

sample = np.array(
        [[0.5068503,  0.4909574,  0.48088843, 0.56281924, 0.5019796,  0.55661233],
         [0.5004162,  0.5388744,  0.5154122, 0.6002497,  0.5186469,  0.52400404],
         [0.5173367,  0.53379077, 0.48255765, 0.57839257, 0.49632147, 0.5143831 ],
         [0.5200508,  0.540166,   0.4624398, 0.5662759,  0.5013316,  0.55133134]]
        )

#print(sample)
y_val_pred = np.where(sample >= 0.5, 1, 0)
#print(y_val_pred)


micro_f1 = f1_score(y_pred=y_val_pred, y_true=y_val_true, average='micro')
macro_f1 = f1_score(y_pred=y_val_pred, y_true=y_val_true, average='macro')

print("MicroF1:{:.3f}\tMacroF1:{:.3f}".format(
                                    micro_f1, macro_f1
                                    )
    )

"""


"""
model_dir = './out_test/iter_0/*.json'
training_loss_files = glob.glob(model_dir)
print(sorted(training_loss_files))

valid_losses = []
for t_l_f in sorted(training_loss_files):
    t_l = json.load(open(t_l_f))
    valid_loss = t_l['val_losses']
    valid_losses.append(valid_loss[0])
    print(valid_loss)

best_fold_i = valid_losses.index(min(valid_losses))
print(best_fold_i)
print('best_model_fold_{}.pt'.format(best_fold_i))
"""


"""
def fit(
    xy_train: torch.utils.data.dataset.TensorDataset,
    xy_val: torch.utils.data.dataset.TensorDataset):
    print(xy_train)
    print(xy_val)



parser = argparse.ArgumentParser()
parser.add_argument(
    '-enc', '--encoder', default='bert-base',
    help='Encoder'
)
args = parser.parse_args()

data_dir = './work/temp_id_gold.json'
data_dict = json.load(open(data_dir))

MAX_TOKEN_LEN = 512
batch_size = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

data_set = TemplateIdsDataset(
            data_dict,
            tokenizer,
            MAX_TOKEN_LEN
        )



m = TorchTemplateClassifier(args)
loss_fn = nn.BCELoss()


_, _, input_ids, attention_masks, labels = data_set.preprocess_dataset()
dataset = data_set.make_tensor_dataset()
#print(dataset)



for i in range(5):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_valid_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    #print(train_dataset)
    t_x = [train_valid_dataset[i][0] for i, _ in enumerate(train_valid_dataset)]
    #print(len(t_x))
    
    kf = KFold(n_splits=5)
    for _fold, (train_index, valid_index) in enumerate(kf.split(t_x)):
        #print(train_index)
        #print(valid_index)

        train_dataset = Subset(train_valid_dataset, train_index)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dataset   = Subset(train_valid_dataset, valid_index)
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

        #fit(train_dataset, valid_dataset)
        #fit(train_dataloader, valid_dataloader)
        print(len(valid_index))
        count = 0
        for batch in valid_dataloader:
            #print(batch)
            id, attention, y_true = (d for d in batch)
            y_pred = m(id, attention)
            
            loss = loss_fn(y_pred, y_true)
            print("count:{}\tloss:{}".format(count, loss))
            count += 1
            
        break
    break


"""

"""

# データセットクラスの作成
dataset = TensorDataset(input_ids, attention_masks, labels)

# 90%地点のIDを取得
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# データセットを分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('訓練データ数：{}'.format(train_size))
print('検証データ数:　{} '.format(val_size))


# 90%地点のIDを取得
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# データセットを分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('訓練データ数：{}'.format(train_size))
print('検証データ数:　{} '.format(val_size))


# データローダーの作成
batch_size = 16

# 訓練データローダー
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
            batch_size = batch_size
        )

# 検証データローダー
validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), # 順番にデータを取得してバッチ化
            batch_size = batch_size
        )

"""