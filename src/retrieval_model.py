

import json
import pickle
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score




def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def return_arg_str_sent_ids(arg_str_idx:list, ref_id:list):
    and_count = [
        len(set(as_idx) & set(ref_id)) for as_idx in arg_str_idx
        ]
    assert len(and_count) == len(arg_str_idx)
    arg_id = and_count.index(max(and_count))
    return arg_str_idx[arg_id]


def convert_one_hot_from_temp_id(temp_ids:list, temp_id_info:dict):
    one_hot_temp_ids = [0] * len(temp_id_info)
    for t_id in temp_ids:
        one_hot_id = temp_id_info[t_id]['position']
        one_hot_temp_ids[one_hot_id] = 1
    return one_hot_temp_ids


def cal_pre_rec_f1(y_true:list, y_pred:list):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return pre, rec, f1


def cal_average(sample_list:list):
    s = sum(sample_list)
    l = len(sample_list)
    return s / l *100





## args setting
parser = argparse.ArgumentParser()
parser.add_argument(
    '-st', '--sim-type', default='target', choices=['target', 'they_target', 'argstr_target', 'random', 'majority']
)
parser.add_argument(
    '-seed', '--random-seed', default=0,
    help='random seed.'
)
parser.add_argument(
    '-d', '--show-details', action='store_true',
    help='show matching details.'
)
args = parser.parse_args()





## fixed seed
seed = args.random_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


## set embeddings on each target sentences
temp_id_gold_dir = './work/temp_id_gold.json'
temp_id_info_dir = './work/temp_id_info.json'
sbert_embeddings_dir = './work/sbert_embeddings.pickle'

temp_id_gold = json.load(open(temp_id_gold_dir))
temp_id_info = json.load(open(temp_id_info_dir))
with open(sbert_embeddings_dir, 'rb') as f:
    sbert_embeddings = pickle.load(f)

test_lo_ids_dir = './data/test_lo_ids.txt'
with open(test_lo_ids_dir, encoding='utf-8')as f:
    test_lo_ids = [lo_id for lo_id in f.read().splitlines()]


tr_sent_emb_datas = []
te_sent_emb_datas = []
for lo_id, data_dict in tqdm(temp_id_gold.items()):
    pm_speech = data_dict['speech']['pm_speech']['speech']
    lo_speech = data_dict['speech']['lo_speech']['speech']
    lo_sentences = data_dict['speech']['lo_speech']['sentences']
    arg_str_idx = data_dict['argument_structure']['argument_structure_sent_idx']

    temp_data = data_dict['temp_data']
    for t_data in temp_data.values():
        ref_id = t_data['ref_id']
        arg_id = return_arg_str_sent_ids(arg_str_idx, ref_id)
        fb_comments = t_data['feedback_comments']
        temp_id_names = [fb_c['template_number'] for fb_c in fb_comments]
        temp_id_onehot = convert_one_hot_from_temp_id(temp_id_names, temp_id_info)
        they_embedding = sbert_embeddings[lo_id]['lo_speech']['embeddings'][0]
        target_embedding = sbert_embeddings[lo_id]['lo_speech']['embeddings'][[ref_id]]
        target_embedding = torch.mean(target_embedding, 0)
        arg_str_embedding = sbert_embeddings[lo_id]['lo_speech']['embeddings'][[arg_id]]
        arg_str_embedding = torch.mean(arg_str_embedding, 0)

        sent_emb_info = {
            'lo_id':lo_id,
            'pm_speech': pm_speech,
            'lo_speech': lo_speech,
            'lo_sentences': lo_sentences,
            'ref_id': ref_id,
            'feedback_comments': fb_comments,
            'temp_id': temp_id_onehot,
            'they_embedding': they_embedding,
            'arg_str_embedding': arg_str_embedding,
            'target_embedding': target_embedding
        }

        if lo_id in test_lo_ids:
            te_sent_emb_datas.append(sent_emb_info)
        else:
            tr_sent_emb_datas.append(sent_emb_info)


best_sims = []
match_ids = []
for te_sent_emb_data in te_sent_emb_datas:
    te_target_emb = te_sent_emb_data["target_embedding"]

    if args.sim_type == 'they_target':
        te_they_emb = te_sent_emb_data["they_embedding"]
    
    if args.sim_type == 'argstr_target':
        te_argstr_emb = te_sent_emb_data["arg_str_embedding"]


    sims = []
    for tr_sent_emb_data in tr_sent_emb_datas:
        tr_target_emb = tr_sent_emb_data["target_embedding"]

        sim = cos_sim(te_target_emb.numpy(), tr_target_emb.numpy())

        if np.isnan(sim):
            sim = 0
        if args.sim_type == 'they_target':
            tr_they_emb = tr_sent_emb_data["they_embedding"]
            they_sim = cos_sim(te_they_emb.numpy(), tr_they_emb.numpy())
            sim = sim * they_sim

        if args.sim_type == 'argstr_target':
            tr_argstr_emb = tr_sent_emb_data["arg_str_embedding"]
            argstr_sim = cos_sim(te_argstr_emb.numpy(), tr_argstr_emb.numpy())
            sim = sim * argstr_sim

        sims.append(sim)

    sims = torch.tensor(sims)
    sorted_id = torch.argsort(sims, descending=True).tolist()
    match_id = sorted_id[0]
    match_ids.append(match_id)
    best_sim = sims[match_id].item()
    best_sims.append(best_sim)


assert len(te_sent_emb_datas) == len(match_ids)

if args.sim_type == 'random':
    temp_id_os, temp_id_ms = [], []
    for te_sent_emb_data in te_sent_emb_datas:
        lo_id_o = te_sent_emb_data['lo_id']
        temp_id_o = te_sent_emb_data['temp_id']
        temp_id_m = [random.randint(0, 1) for _ in range(len(temp_id_info))]
        temp_id_os.append(temp_id_o)
        temp_id_ms.append(temp_id_m)

    p_macro = precision_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
    r_macro = recall_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
    f_macro = f1_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
    print(p_macro, r_macro, f_macro)
    exit()

if args.sim_type == 'majority':
    temp_id_os, temp_id_ms = [], []
    for te_sent_emb_data in te_sent_emb_datas:
        lo_id_o = te_sent_emb_data['lo_id']
        temp_id_o = te_sent_emb_data['temp_id']
        temp_id_m = [0]*25
        temp_id_m[9] = 1
        precision, recall, f1 = cal_pre_rec_f1(temp_id_o, temp_id_m)
        temp_id_os.append(temp_id_o)
        temp_id_ms.append(temp_id_m)
        
    p_macro = precision_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
    r_macro = recall_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
    f_macro = f1_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
    print(p_macro, r_macro, f_macro)
    exit()


data_array2d = []
temp_id_os = []
temp_id_ms = []
for i, (te_sent_emb_data, m_id, similarity) in enumerate(zip(te_sent_emb_datas, match_ids, best_sims)):
    original_data = te_sent_emb_data
    matching_data = tr_sent_emb_datas[m_id]

    lo_id_o = original_data['lo_id']
    pm_speech_o = original_data['pm_speech']
    lo_speech_o = original_data['lo_speech']
    ref_id_o = original_data['ref_id']
    target_sent_o = [original_data['lo_sentences'][i] for i in ref_id_o]
    target_sent_o = ' '.join(target_sent_o)
    fb_comments_o = original_data['feedback_comments']
    temp_id_o = original_data['temp_id']

    lo_id_m = matching_data['lo_id']
    pm_speech_m = matching_data['pm_speech']
    lo_speech_m = matching_data['lo_speech']
    ref_id_m = matching_data['ref_id']
    target_sent_m = [matching_data['lo_sentences'][i] for i in ref_id_m]
    target_sent_m = ' '.join(target_sent_m)
    fb_comments_m = matching_data['feedback_comments']
    temp_id_m = matching_data['temp_id']

    precision, recall, f1 = cal_pre_rec_f1(temp_id_o, temp_id_m)

    print('### sample {}'.format(i))
    print(
        'original: {}\tmatching: {}\tsimilarity:{:.4f}\nTEMPID\tPrecision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}'.format(
            lo_id_o, lo_id_m, similarity, precision, recall, f1
        )
    )
    if args.show_details:
        print('--- lo speech ---')
        print(lo_speech_o)
        print('--- target sent ---')
        print(target_sent_o)
        print('--- prediction ---')
        print(fb_comments_m)
        print('--- gold ---')
        print(fb_comments_o)
    print()
    
    data_array2d.append(
        [
            lo_id_o, pm_speech_o, lo_speech_o, target_sent_o, fb_comments_o,
            similarity, lo_id_m, pm_speech_m, lo_speech_m, target_sent_m, fb_comments_m,
            precision, recall, f1
        ]
    )
    temp_id_os.append(temp_id_o)
    temp_id_ms.append(temp_id_m)

p_macro = precision_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
r_macro = recall_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
f_macro = f1_score(temp_id_os, temp_id_ms, average='micro', zero_division=0)
print(p_macro, r_macro, f_macro)


df_sample = pd.DataFrame(data_array2d)
#df_sample.to_excel('./trash/data/retrieval_model_sample.xlsx', header=False, index=False)
