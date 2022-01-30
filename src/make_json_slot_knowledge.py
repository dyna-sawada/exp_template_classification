
import json


temp_id_gold_dir = './work/temp_id_gold.json'
temp_id_gold = json.load(open(temp_id_gold_dir))

temp_id_info_dir = './data/temp_id_info.json'
temp_id_info = json.load(open(temp_id_info_dir))

test_lo_ids_dir = './data/test_lo_ids.txt'
with open(test_lo_ids_dir, encoding='utf-8')as f:
    test_lo_ids = [lo_id for lo_id in f.read().splitlines()]


slot_knowledge_dict = {}
for template_number, temp_info_dict in temp_id_info.items():
    if template_number == '999':
        continue
    temp_text_jp = temp_info_dict['temp_text_jp']
    temp_text_en = temp_info_dict['temp_text_en']
    slot_knowledge_dict[template_number] = {
        'temp_text': {
            'jp': temp_text_jp,
            'en': temp_text_en
        },
        'slot1': {
            'jp': [],
            'en': []
        },
        'slot2': {
            'jp': [],
            'en': []
        },
        'slot3': {
            'jp': [],
            'en': []
        }
    }


for lo_id, debate_info_dict in temp_id_gold.items():

    if lo_id in test_lo_ids:
        continue

    temp_data_dict = debate_info_dict['temp_data']

    for _i, fb_info_dict in temp_data_dict.items():
        feedback_comments = fb_info_dict['feedback_comments']

        for fb_comment in feedback_comments:
            template_number = fb_comment['template_number']

            if template_number == '999':
                continue
            
            slot1_jp = fb_comment['slot1_jp']
            slot1_en = fb_comment['slot1_en']
            slot2_jp = fb_comment['slot2_jp']
            slot2_en = fb_comment['slot2_en']
            slot3_jp = fb_comment['slot3_jp']
            slot3_en = fb_comment['slot3_en']

            slot_knowledge_dict[template_number]['slot1']['jp'].append(slot1_jp)
            slot_knowledge_dict[template_number]['slot1']['en'].append(slot1_en)
            slot_knowledge_dict[template_number]['slot2']['jp'].append(slot2_jp)
            slot_knowledge_dict[template_number]['slot2']['en'].append(slot2_en)
            slot_knowledge_dict[template_number]['slot3']['jp'].append(slot3_jp)
            slot_knowledge_dict[template_number]['slot3']['en'].append(slot3_en)

            assert len(slot_knowledge_dict[template_number]['slot1']['jp']) == len(slot_knowledge_dict[template_number]['slot2']['jp'])
            assert len(slot_knowledge_dict[template_number]['slot1']['jp']) == len(slot_knowledge_dict[template_number]['slot3']['jp'])
            assert len(slot_knowledge_dict[template_number]['slot1']['en']) == len(slot_knowledge_dict[template_number]['slot2']['en'])
            assert len(slot_knowledge_dict[template_number]['slot1']['en']) == len(slot_knowledge_dict[template_number]['slot3']['en'])
            assert len(slot_knowledge_dict[template_number]['slot1']['jp']) == len(slot_knowledge_dict[template_number]['slot1']['en'])


with open('./work/slot_knowledges.json', mode='wt', encoding='utf=8')as f:
    json.dump(slot_knowledge_dict, f, indent=2, ensure_ascii=False)
    