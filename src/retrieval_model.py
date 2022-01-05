

import json
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.functional as F




def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


## args setting
parser = argparse.ArgumentParser()
parser.add_argument(
    '-st', '--sim-type', default='target', choices=['target', 'they_target']
)
parser.add_argument(
    '-seed', '--random-seed', default=0,
    help='random seed.'
)
args = parser.parse_args()


## fixed seed
seed = args.random_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


## set embeddings on each target sentences
temp_id_gold_dir = './work/temp_id_gold.json'
sbert_embeddings_dir = './work/sbert_embeddings.pickle'

temp_id_gold = json.load(open(temp_id_gold_dir))
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

    temp_data = data_dict['temp_data']
    for t_data in temp_data.values():
        ref_id = t_data['ref_id']
        fb_comments = t_data['feedback_comments']
        they_embedding = sbert_embeddings[lo_id]['lo_speech']['embeddings'][0]
        target_embedding = sbert_embeddings[lo_id]['lo_speech']['embeddings'][[ref_id]]
        target_embedding = torch.mean(target_embedding, 0)

        sent_emb_info = {
            'lo_id':lo_id,
            'pm_speech': pm_speech,
            'lo_speech': lo_speech,
            'lo_sentences': lo_sentences,
            'ref_id': ref_id,
            'feedback_comments': fb_comments,
            'they_embedding': they_embedding,
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

        sims.append(sim)

    sims = torch.tensor(sims)
    sorted_id = torch.argsort(sims, descending=True).tolist()
    match_id = sorted_id[0]
    match_ids.append(match_id)
    best_sim = sims[match_id].item()
    best_sims.append(best_sim)


assert len(te_sent_emb_datas) == len(match_ids)


data_array2d = []
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

    lo_id_m = matching_data['lo_id']
    pm_speech_m = matching_data['pm_speech']
    lo_speech_m = matching_data['lo_speech']
    ref_id_m = matching_data['ref_id']
    target_sent_m = [matching_data['lo_sentences'][i] for i in ref_id_m]
    target_sent_m = ' '.join(target_sent_m)
    fb_comments_m = matching_data['feedback_comments']

    print('### sample {}'.format(i))
    print('original: {}\tmatching: {}\tsimilarity:{:.4f}'.format(lo_id_o, lo_id_m, similarity))
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
            similarity, lo_id_m, pm_speech_m, lo_speech_m, target_sent_m, fb_comments_m
        ]
    )


df_sample = pd.DataFrame(data_array2d)
#df_sample.to_excel('./trash/data/retrieval_model_sample.xlsx', header=False, index=False)
