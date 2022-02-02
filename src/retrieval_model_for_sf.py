

import re
import json
import pickle
from tqdm import tqdm
import random
import itertools
import statistics
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.functional as F



def cal_cos_sim(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    sims = torch.squeeze(sim_mt)
    ind = int(torch.argmax(sims))
    best_sim = int(torch.max(sims))
    return ind, best_sim, sims


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

def get_motion(LOID):
    m = re.search('(.*)_LO_(.*)_(.*)', LOID)
    motion = m.group(1)
    return motion



## args setting
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='pair', choices=['pair', 'all', 'majority', 'random_pair', 'random_all']
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


## set embeddings
temp_id_gold_dir = './work/temp_id_gold.json'
temp_id_info_dir = './data/temp_id_info.json'
test_lo_ids_dir = './data/test_lo_ids.txt'
slot_knowledges_dir = './work/slot_knowledges.json'
sbert_embeddings_ts_dir = './work/sbert_embeddings_for_ts.pickle'

temp_id_gold = json.load(open(temp_id_gold_dir))
temp_id_info = json.load(open(temp_id_info_dir))
slot_knowledges = json.load(open(slot_knowledges_dir))
with open(test_lo_ids_dir, encoding='utf-8')as f:
    test_lo_ids = [lo_id for lo_id in f.read().splitlines()]
with open(sbert_embeddings_ts_dir, 'rb') as f:
    sbert_embeddings_ts = pickle.load(f)

if args.model == 'pair':
    sbert_embeddings_sf_dir = './work/sbert_embeddings_for_sf_pair.pickle'
    with open(sbert_embeddings_sf_dir, 'rb') as f:
        sbert_embeddings_sf = pickle.load(f)
elif args.model == 'all':
    sbert_embeddings_sf_dir = './work/sbert_embeddings_for_sf_all.pickle'
    with open(sbert_embeddings_sf_dir, 'rb') as f:
        sbert_embeddings_sf = pickle.load(f)



te_sent_emb_datas = []
for lo_id, data_dict in tqdm(temp_id_gold.items()):
    if lo_id not in test_lo_ids:
        continue
    pm_speech = data_dict['speech']['pm_speech']['speech']
    lo_speech = data_dict['speech']['lo_speech']['speech']
    lo_sentences = data_dict['speech']['lo_speech']['sentences']
    arg_str_idx = data_dict['argument_structure']['argument_structure_sent_idx']

    temp_data = data_dict['temp_data']
    for t_data in temp_data.values():
        ref_id = t_data['ref_id']
        if len(ref_id) == 0:
            ref_id = list(range(len(lo_sentences)))
        arg_id = return_arg_str_sent_ids(arg_str_idx, ref_id)
        fb_comments = t_data['feedback_comments']
        temp_id_names = [fb_c['template_number'] for fb_c in fb_comments]
        temp_id_onehot = convert_one_hot_from_temp_id(temp_id_names, temp_id_info)
        they_embedding = sbert_embeddings_ts[lo_id]['lo_speech']['embeddings'][0]
        target_embedding = sbert_embeddings_ts[lo_id]['lo_speech']['embeddings'][[ref_id]]
        target_embedding = torch.mean(target_embedding, 0)
        arg_str_embedding = sbert_embeddings_ts[lo_id]['lo_speech']['embeddings'][[arg_id]]
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

        te_sent_emb_datas.append(sent_emb_info)



data_array2d = []
for te_sent_emb_data in te_sent_emb_datas:
    lo_id_o = te_sent_emb_data['lo_id']
    motion = get_motion(lo_id_o)
    pm_speech_o = te_sent_emb_data['pm_speech']
    lo_speech_o = te_sent_emb_data['lo_speech']
    ref_id_o = te_sent_emb_data['ref_id']
    target_sent_o = [te_sent_emb_data['lo_sentences'][i] for i in ref_id_o]
    target_sent_o = ' '.join(target_sent_o)
    te_feedback_comments = te_sent_emb_data['feedback_comments']

    te_target_emb = te_sent_emb_data['target_embedding']
    te_target_emb = torch.unsqueeze(te_target_emb, dim=0)

    for i, te_fb_comment in enumerate(te_feedback_comments):
        temp_id = te_fb_comment['template_number']
        if temp_id == '999':
            continue

        split_index = slot_knowledges[temp_id]['split_index']

        fb_comment_o = te_fb_comment['original_comment']
        fixed_comment_o = te_fb_comment['fixed_comment_jp']
        temp_comment_o = te_fb_comment['template_comment_jp']

        if args.model == 'pair':
            if motion == 'DP':
                tr_temp_emb = sbert_embeddings_sf[temp_id]['embeddings'][:split_index]
            elif motion == 'HW':
                tr_temp_emb = sbert_embeddings_sf[temp_id]['embeddings'][split_index:]
            ind, best_sim, sims = cal_cos_sim(te_target_emb, tr_temp_emb)
            temp_comment_m_en = sbert_embeddings_sf[temp_id]['temp_comments'][ind]
        
            slot1_ind = ind
            slot2_ind = ind
            slot3_ind = ind

        elif args.model == 'all':
            slot1_en_o, slot2_en_o, slot3_en_o = slot_knowledges[temp_id]['slot1']['en'], slot_knowledges[temp_id]['slot2']['en'], slot_knowledges[temp_id]['slot3']['en']
            slot1_en, slot2_en, slot3_en = slot1_en_o[:split_index], slot2_en_o[:split_index], slot3_en_o[:split_index]
            sorted_slot1_en, sorted_slot2_en, sorted_slot3_en = sorted(list(set(slot1_en))), sorted(list(set(slot2_en))), sorted(list(set(slot3_en)))
            slot_all = list(itertools.product(sorted_slot1_en, sorted_slot2_en, sorted_slot3_en))
            
            if motion == 'DP':
                tr_temp_emb = sbert_embeddings_sf[temp_id]['embeddings'][:len(slot_all)]
            elif motion == 'HW':
                tr_temp_emb = sbert_embeddings_sf[temp_id]['embeddings'][len(slot_all):]
            
            ind, best_sim, sims = cal_cos_sim(te_target_emb, tr_temp_emb)
            temp_comment_m_en = sbert_embeddings_sf[temp_id]['temp_comments'][ind]

            if motion == 'HW':
                slot1_en, slot2_en, slot3_en = slot1_en_o[split_index:], slot2_en_o[split_index:], slot3_en_o[split_index:]
                sorted_slot1_en, sorted_slot2_en, sorted_slot3_en = sorted(list(set(slot1_en))), sorted(list(set(slot2_en))), sorted(list(set(slot3_en)))
                slot_all = list(itertools.product(sorted_slot1_en, sorted_slot2_en, sorted_slot3_en))        

            slot1_ind = slot1_en.index(slot_all[ind][0])
            slot2_ind = slot2_en.index(slot_all[ind][1])
            slot3_ind = slot3_en.index(slot_all[ind][2])

        elif args.model == 'random_pair':
            slot1s = slot_knowledges[temp_id]['slot1']['jp']
            if motion == 'DP':
                slot1s_m = slot1s[:split_index]
            elif motion == 'HW':
                slot1s_m = slot1s[split_index:]
            ind = random.randrange(0, len(slot1s_m))
            slot1_ind = ind
            slot2_ind = ind
            slot3_ind = ind
        
        elif args.model == 'random_all':
            slot1s = slot_knowledges[temp_id]['slot1']['jp']
            if motion == 'DP':
                slot1s_m = slot1s[:split_index]
            elif motion == 'HW':
                slot1s_m = slot1s[split_index:]
            slot1_ind = random.randrange(0, len(slot1s_m))
            slot2_ind = random.randrange(0, len(slot1s_m))
            slot3_ind = random.randrange(0, len(slot1s_m))

        elif args.model == 'majority':
            slot1s = slot_knowledges[temp_id]['slot1']['jp']
            slot2s = slot_knowledges[temp_id]['slot2']['jp']
            slot3s = slot_knowledges[temp_id]['slot3']['jp']
            if motion == 'DP':
                slot1s_m = slot1s[:split_index]
                slot2s_m = slot2s[:split_index]
                slot3s_m = slot3s[:split_index]
            elif motion == 'HW':
                slot1s_m = slot1s[split_index:]
                slot2s_m = slot2s[split_index:]
                slot3s_m = slot3s[split_index:]

            slot1_ind = slot1s.index(statistics.mode(slot1s_m))
            slot2_ind = slot2s.index(statistics.mode(slot2s_m))
            slot3_ind = slot3s.index(statistics.mode(slot3s_m))

        temp_comment_m_jp = slot_knowledges[temp_id]['temp_text']['jp'].translate(
            str.maketrans({
                    'X': slot_knowledges[temp_id]['slot1']['jp'][slot1_ind],
                    'Y': slot_knowledges[temp_id]['slot2']['jp'][slot2_ind],
                    'Z': slot_knowledges[temp_id]['slot3']['jp'][slot3_ind]
            })
        )


        if args.show_details:
            print('--- lo speech ---')
            print(lo_speech_o)
            print('--- target sent ---')
            print(target_sent_o)
            print('--- temp id ---')
            print(temp_id)
            print('--- gold ---')
            print(temp_comment_o)
            print('--- prediction ---')
            print(temp_comment_m_jp)
            print()

        data_array2d.append(
            [lo_id_o, pm_speech_o, lo_speech_o, target_sent_o, temp_id, temp_comment_o, temp_comment_m_jp]
        )


df_sample = pd.DataFrame(data_array2d)
if args.model == 'pair':
    df_sample.to_excel('./out_test/retrieval_model_out_sf_pair.xlsx', header=False, index=False)
elif args.model == 'all':
    df_sample.to_excel('./out_test/retrieval_model_out_sf_all.xlsx', header=False, index=False)
elif args.model == 'random_pair':
    df_sample.to_excel('./out_test/retrieval_model_out_sf_rp.xlsx', header=False, index=False)
elif args.model == 'random_all':
    df_sample.to_excel('./out_test/retrieval_model_out_sf_ra.xlsx', header=False, index=False)
elif args.model == 'majority':
    df_sample.to_excel('./out_test/retrieval_model_out_sf_ma.xlsx', header=False, index=False)
