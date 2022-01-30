

import json
import pprint
import nltk



def main():
    
    ANNOTATION_DATA_PATH = './data/diagnostic_comments.json'
    TEMPLATE_INFO_DATA_PATH = './data/temp_id_info.json'

    annotation_data = json.load(open(ANNOTATION_DATA_PATH))
    temp_id_info = json.load(open(TEMPLATE_INFO_DATA_PATH))

    used_temp_ids_gold = set(list(temp_id_info.keys()))

    used_temp_ids_sample = []
    for _lo_id, anno_data in annotation_data.items():

        diagnostic_comments = anno_data['diagnostic_comments']

        for d_comment in diagnostic_comments:
            template_annotation = d_comment['template_annotation']
            temp_id_sample = template_annotation[0]['template_number']
            if temp_id_sample != None:
                used_temp_ids_sample.append(temp_id_sample)
    used_temp_ids_sample = set(used_temp_ids_sample)
    
    assert used_temp_ids_gold == used_temp_ids_sample, "Template ID is not match."
    

    used_lo_ids = list(annotation_data.keys())
    
    temp_id_gold = {}
    for l in used_lo_ids:
        temp_id_gold[l] = {
            'speech':{},
            'argument_structure': 'hoge',
            'temp_data': {}
        }

    for lo_id, anno_data in annotation_data.items():
        pm_speech = anno_data['speech']['pm_speech']
        pm_sentences = nltk.sent_tokenize(pm_speech)
        pm_speech_info = {
            'speech':pm_speech,
            'sentences':pm_sentences
        }
        lo_speech_info = anno_data['speech']['lo_speech']
        argument_structure = anno_data['argument_structure']
        diagnostic_comments = anno_data['diagnostic_comments']

        temp_id_gold[lo_id]['speech']['pm_speech'] = pm_speech_info
        temp_id_gold[lo_id]['speech']['lo_speech'] = lo_speech_info
        temp_id_gold[lo_id]['argument_structure'] = argument_structure


        for d_comments in diagnostic_comments:
            p_id = 0

            for _i in range(len(diagnostic_comments)):

                is_valid = d_comments['template_annotation'][0]['is_valid']
                is_understandable = d_comments['template_annotation'][0]['is_understandable']
                is_not_duplicated = d_comments['template_annotation'][0]['is_not_duplicated']

                if is_valid == False or is_understandable == False or is_not_duplicated == False:
                    continue

                original_comment = d_comments['original_comment']
                fixed_comment_jp = d_comments['template_annotation'][0]['fixed_comment_jp']
                fixed_comment_en = d_comments['template_annotation'][0]['fixed_comment_en']
                template_comment_jp = d_comments['template_annotation'][0]['template_comment_jp']
                template_comment_en = d_comments['template_annotation'][0]['template_comment_en']
                template_number = d_comments['template_annotation'][0]['template_number']
                #template_text = temp_id_info[template_number]['temp_text']
                slot1_jp = d_comments['template_annotation'][0]['slot1_jp']
                slot1_en = d_comments['template_annotation'][0]['slot1_en']
                slot2_jp = d_comments['template_annotation'][0]['slot2_jp']
                slot2_en = d_comments['template_annotation'][0]['slot2_en']
                slot3_jp = d_comments['template_annotation'][0]['slot3_jp']
                slot3_en = d_comments['template_annotation'][0]['slot3_en']
                temp_id = d_comments['template_annotation'][0]['template_number']
                ref_id = d_comments['target_sent_idx']

                if temp_id is None:
                    continue

                if p_id not in temp_id_gold[lo_id]['temp_data'].keys():
                    temp_id_gold[lo_id]['temp_data'][p_id] = {
                                                            'ref_id': ref_id,
                                                            'temp_id': [0] * len(used_temp_ids_gold),
                                                            'feedback_comments':[]
                                                            }
                    temp_position = temp_id_info[temp_id]['position']
                    temp_id_gold[lo_id]['temp_data'][p_id]['temp_id'][temp_position] = 1

                    temp_id_gold[lo_id]['temp_data'][p_id]['feedback_comments'].append(
                        {
                            'original_comment': original_comment,
                            'fixed_comment_jp': fixed_comment_jp,
                            'fixed_comment_en': fixed_comment_en,
                            'template_comment_jp': template_comment_jp,
                            'template_comment_en': template_comment_en,
                            'template_number': template_number,
                            'slot1_jp': slot1_jp,
                            'slot1_en': slot1_en,
                            'slot2_jp': slot2_jp,
                            'slot2_en': slot2_en,
                            'slot3_jp': slot3_jp,
                            'slot3_en': slot3_en
                        }
                    )
                                    
                else:
                    if ref_id == temp_id_gold[lo_id]['temp_data'][p_id]['ref_id']:
                        temp_position = temp_id_info[temp_id]['position']
                        if temp_id_gold[lo_id]['temp_data'][p_id]['temp_id'][temp_position] == 0:
                            temp_id_gold[lo_id]['temp_data'][p_id]['temp_id'][temp_position] = 1

                            temp_id_gold[lo_id]['temp_data'][p_id]['feedback_comments'].append(
                                {
                                    'original_comment': original_comment,
                                    'fixed_comment_jp': fixed_comment_jp,
                                    'fixed_comment_en': fixed_comment_en,
                                    'template_comment_jp': template_comment_jp,
                                    'template_comment_en': template_comment_en,
                                    'template_number': template_number,
                                    'slot1_jp': slot1_jp,
                                    'slot1_en': slot1_en,
                                    'slot2_jp': slot2_jp,
                                    'slot2_en': slot2_en,
                                    'slot3_jp': slot3_jp,
                                    'slot3_en': slot3_en
                                }
                            )

                    else:
                        p_id += 1


    pprint.pprint(temp_id_gold, indent=2)
    with open('./work/temp_id_gold.json', mode='wt', encoding='utf=8')as f:
        json.dump(temp_id_gold, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    main()


