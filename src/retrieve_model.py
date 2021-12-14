

import json
import pickle
from torch.utils import data
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
#from sklearn.metrics.pairwise import cosine_similarity


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


temp_id_gold_dir = './work/temp_id_gold.json'
sbert_embeddings_dir = './work/sbert_embeddings.pickle'

temp_id_gold = json.load(open(temp_id_gold_dir))
with open(sbert_embeddings_dir, 'rb') as f:
    sbert_embeddings = pickle.load(f)


sent_emb_datas = []

for lo_id, data_dict in tqdm(temp_id_gold.items()):
    pm_speech = data_dict['speech']['pm_speech']['speech']
    lo_speech = data_dict['speech']['lo_speech']['speech']
    lo_sentences = data_dict['speech']['lo_speech']['sentences']

    temp_data = data_dict['temp_data']
    for t_data in temp_data.values():
        ref_id = t_data['ref_id']
        fb_comments = t_data['feedback_comments']
        embedding = sbert_embeddings[lo_id]['lo_speech']['embeddings'][[ref_id]]
        embedding = torch.mean(embedding, 0)

        sent_emb_info = {
            'lo_id':lo_id,
            'pm_speech': pm_speech,
            'lo_speech': lo_speech,
            'lo_sentences': lo_sentences,
            'ref_id': ref_id,
            'feedback_comments': fb_comments,
            'embedding': embedding
        }

        sent_emb_datas.append(sent_emb_info)


#print(len(sent_emb_datas))

sample_index = np.random.randint(0, len(sent_emb_datas), 20)
#print(sample_index)
match_index = []
best_sims = []
for sample_id in sample_index:
    sample_emb = sent_emb_datas[sample_id]['embedding']
    sample_lo_id = sent_emb_datas[sample_id]['lo_id']
    sample_ref_id = sent_emb_datas[sample_id]['ref_id']

    sims = []
    for sent_emb_data in sent_emb_datas:
        sample_emb_2 = sent_emb_data['embedding']
        sample_lo_id_2 = sent_emb_data['lo_id']
        sample_ref_id_2 = sent_emb_data['ref_id']
        if sample_lo_id == sample_lo_id_2 and sample_ref_id == sample_ref_id_2:
            continue
        #sim = F.cosine_similarity(sample_emb, sample_emb_2, dim=0)
        sim = cos_sim(sample_emb.numpy(), sample_emb_2.numpy())
        if np.isnan(sim):
            sim = 0
        if sample_lo_id == sample_lo_id_2:
            sim = 0
        sims.append(sim)


    sims = torch.tensor(sims)
    sorted_id = torch.argsort(sims, descending=True).tolist()
    #print(sims)
    #print(sorted_id)
    match_id = sorted_id[0]
    match_index.append(match_id)

    best_sim = sims[match_id].item()
    best_sims.append(best_sim)

assert len(sample_index) == len(match_index)


data_array2d = []
for s_id, m_id, similarity in zip(sample_index, match_index, best_sims):
    original_data = sent_emb_datas[s_id]
    matching_data = sent_emb_datas[m_id]

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

    print('### sample')
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
