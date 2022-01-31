
import argparse
import json
import pickle
from tqdm import tqdm
import itertools
import pprint
from sentence_transformers import SentenceTransformer


##############################################################################
# convert template-sentence to embeddings with SentenceBert for slot filling #
##############################################################################


def exclude_duplication(sample:list):
    return sorted(list(set(sample)))


parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--combination', default='pair', choices=['pair', 'all']
)
args = parser.parse_args()

t_data_dir = './work/slot_knowledges.json'
t_data = json.load(open(t_data_dir))

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

used_temp_ids = sorted(list(t_data.keys()))
sent_emb_dict = {}
for t_id in used_temp_ids:
    sent_emb_dict[t_id] = {
            'temp_text': t_data[t_id]['temp_text']['en'],
            'temp_comments': [],
            'embeddings': 'foo'
            }


for t_id, slot_infos in t_data.items():
    temp_text = slot_infos['temp_text']['en']
    slot1 = slot_infos['slot1']['en']
    slot2 = slot_infos['slot2']['en']
    slot3 = slot_infos['slot3']['en']

    if args.combination == 'pair':
        for s1, s2, s3 in zip(slot1, slot2, slot3):
            template_comment = temp_text.translate(
                str.maketrans({
                    '{': '',
                    '}': '',
                    'X': s1,
                    'Y': s2,
                    'Z': s3
                })
            )
            sent_emb_dict[t_id]['temp_comments'].append(template_comment)

    elif args.combination == 'all':
        slot1 = exclude_duplication(slot1)
        slot2 = exclude_duplication(slot2)
        slot3 = exclude_duplication(slot3)
        slot_all = list(itertools.product(slot1, slot2, slot3))
        for s_all in slot_all:
            template_comment = temp_text.translate(
                str.maketrans({
                    '{': '',
                    '}': '',
                    'X': s_all[0],
                    'Y': s_all[1],
                    'Z': s_all[2]
                })
            )
            sent_emb_dict[t_id]['temp_comments'].append(template_comment)
        


for t_id in tqdm(used_temp_ids):
    temp_comments = sent_emb_dict[t_id]['temp_comments']
    temp_embeddings = model.encode(temp_comments, convert_to_tensor=True)
    sent_emb_dict[t_id]['embeddings'] = temp_embeddings


pprint.pprint(sent_emb_dict, indent=2)
if args.combination == 'pair':
    with open('./work/sbert_embeddings_for_sf_pair.pickle', mode='wb') as f:
        pickle.dump(sent_emb_dict, f)
elif args.combination == 'all':
    with open('./work/sbert_embeddings_for_sf_all.pickle', mode='wb') as f:
        pickle.dump(sent_emb_dict, f)

