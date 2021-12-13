
import re
import json
import torch
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pprint



d_data_dir = './work/temp_id_gold.json'
d_data = json.load(open(d_data_dir))


model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')


pm_id_list = [
                'HW1', 'HW2', 'HW3', 'HW4', 'HW5',
                'DP1', 'DP2', 'DP3', 'DP4', 'DP5'
            ]

flag_pm_id = [
                False, False, False, False, False,
                False, False, False, False, False
            ]

pm_emb_list = [0] * 10


used_lo_ids = list(d_data.keys())
sent_emb_dict = {}
for l in used_lo_ids:
    sent_emb_dict[l] = {
            'pm_speech':{
                    'speech':'hoge',
                    'sentences':'hoge',
                    'embeddings':'hoge'
                    },
            'lo_speech':{
                    'speech':'hoge',
                    'sentences':'hoge',
                    'embeddings':'hoge'
                }
            }


for lo_id, datas in tqdm(d_data.items()):
    
    m = re.search('(.*)_LO_([0-9])_(.*)', lo_id)
    motion, pm_id, _lo_id = m.group(1), m.group(2), m.group(3)
    
    PM_ID = motion + pm_id
    pm_index = pm_id_list.index(PM_ID)

    pm_speech = datas['speech']['pm_speech']['speech']
    pm_sentences = datas['speech']['pm_speech']['sentences']
    lo_speech = datas['speech']['lo_speech']['speech']
    lo_sentences = datas['speech']['lo_speech']['sentences']

    if flag_pm_id[pm_index] == False:
        pm_sent_embs = model.encode(pm_sentences, convert_to_tensor=True)
        pm_emb_list[pm_index] = pm_sent_embs
        flag_pm_id[pm_index] == True


    lo_embeddings = model.encode(lo_sentences, convert_to_tensor=True)
    pm_embeddings = pm_emb_list[pm_index]


    sent_emb_dict[lo_id]['pm_speech']['speech'] = pm_speech
    sent_emb_dict[lo_id]['pm_speech']['sentences'] = pm_sentences
    sent_emb_dict[lo_id]['pm_speech']['embeddings'] = pm_embeddings
    sent_emb_dict[lo_id]['lo_speech']['speech'] = lo_speech
    sent_emb_dict[lo_id]['lo_speech']['sentences'] = lo_sentences
    sent_emb_dict[lo_id]['lo_speech']['embeddings'] = lo_embeddings



pprint.pprint(sent_emb_dict, indent=2)
with open('./work/sbert_embeddings.pickle', mode='wb') as f:
    pickle.dump(sent_emb_dict,f)


