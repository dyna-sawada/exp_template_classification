
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
    temp_text = temp_info_dict['temp_text']
    slot_knowledge_dict[template_number] = {
        'temp_text': temp_text,
        'slot1': [],
        'slot2': [],
        'slot3': []
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
            
            slot1 = fb_comment['slot1']
            slot2 = fb_comment['slot2']
            slot3 = fb_comment['slot3']

            slot_knowledge_dict[template_number]['slot1'].append(slot1)
            slot_knowledge_dict[template_number]['slot2'].append(slot2)
            slot_knowledge_dict[template_number]['slot3'].append(slot3)

            assert len(slot_knowledge_dict[template_number]['slot1']) == len(slot_knowledge_dict[template_number]['slot2'])
            assert len(slot_knowledge_dict[template_number]['slot1']) == len(slot_knowledge_dict[template_number]['slot3'])



with open('./work/slot_knowledges.json', mode='wt', encoding='utf=8')as f:
    json.dump(slot_knowledge_dict, f, indent=2, ensure_ascii=False)
    