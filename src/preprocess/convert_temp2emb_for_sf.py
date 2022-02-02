
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
    split_index = slot_infos['split_index']

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
        slot1_dp, slot2_dp, slot3_dp = slot1[:split_index], slot2[:split_index], slot3[:split_index]
        slot1_hw, slot2_hw, slot3_hw = slot1[split_index:], slot2[split_index:], slot3[split_index:]
        slot1_dp, slot2_dp, slot3_dp = exclude_duplication(slot1_dp), exclude_duplication(slot2_dp), exclude_duplication(slot3_dp)
        slot1_hw, slot2_hw, slot3_hw = exclude_duplication(slot1_hw), exclude_duplication(slot2_hw), exclude_duplication(slot3_hw)

        slot_all_dp = list(itertools.product(slot1_dp, slot2_dp, slot3_dp))
        slot_all_hw = list(itertools.product(slot1_hw, slot2_hw, slot3_hw))

        for s_all_dp in slot_all_dp:
            template_comment = temp_text.translate(
                str.maketrans({
                    '{': '',
                    '}': '',
                    'X': s_all_dp[0],
                    'Y': s_all_dp[1],
                    'Z': s_all_dp[2]
                })
            )
            sent_emb_dict[t_id]['temp_comments'].append(template_comment)
        
        for s_all_hw in slot_all_hw:
            template_comment = temp_text.translate(
                str.maketrans({
                    '{': '',
                    '}': '',
                    'X': s_all_hw[0],
                    'Y': s_all_hw[1],
                    'Z': s_all_hw[2]
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

