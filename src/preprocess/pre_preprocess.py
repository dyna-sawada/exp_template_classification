

import pandas as pd
import json
import pprint




def embed_temp_id_gold(df_data, temp_id_gold, tempid2position):

    max_count = max(
                    list(
                        df_data[lo_id_column].value_counts()
                        )
                    )

    for _i, row in df_data.iterrows():

        lo_id = row[lo_id_column]
        pm_speech = row[pm_speech_column]
        lo_speech = row[lo_speech_column]
        ref_id = row[fb_reference_column]
        temp_id = row[temp_id_column]
        
        if ref_id != '*':
            ref_id = [int(r_id) for r_id in ref_id.split(',')]

        temp_id_gold[lo_id]['pm_speech'] = pm_speech
        temp_id_gold[lo_id]['lo_speech'] = lo_speech

        p_id = 0
        for _i in range(max_count):
            if p_id not in temp_id_gold[lo_id]['temp_data'].keys():
                temp_id_gold[lo_id]['temp_data'][p_id] = {
                                                        'temp_id': [0] * len(used_temp_ids),
                                                        'ref_id': ref_id
                                                        }
                temp_position = tempid2position[temp_id]['position']
                temp_id_gold[lo_id]['temp_data'][p_id]['temp_id'][temp_position] = 1
                                
            else:
                if ref_id == temp_id_gold[lo_id]['temp_data'][p_id]['ref_id']:
                    temp_position = tempid2position[temp_id]['position']
                    temp_id_gold[lo_id]['temp_data'][p_id]['temp_id'][temp_position] = 1
                else:
                    p_id += 1
    
    return temp_id_gold



def main():
    
    ###  setting  ###
    ANNOTATION_DATA_PATH = './data/テンプレートアノテーション_20211020_整形後.xlsx'

    df_data = pd.read_excel(
        ANNOTATION_DATA_PATH,
        sheet_name='Sheet1',
        index_col=None
    )
    df_label_info = pd.read_excel(
        ANNOTATION_DATA_PATH,
        sheet_name='Sheet2',
        index_col=None
    )

    df_data = df_data.fillna('*')
    df_label_info = df_label_info.fillna('*')

    ## column name settings
    global lo_id_column
    global _component_id_column
    global _judge_id_column
    global pm_speech_column
    global lo_speech_column
    global _fb_id_column
    global fb_reference_column
    global temp_id_column
    
    lo_id_column = 'LO_ID'
    _component_id_column = 'COMPONENT_ID'
    _judge_id_column = 'JUDGE_ID'
    pm_speech_column = 'PM_SPEECH'
    lo_speech_column = 'LO_SPEECH'
    _fb_id_column = 'FB_ID'
    fb_reference_column = 'FB_REFERENCE_ID'
    temp_id_column = 'TEMPLATE_ID'
    
    
    ## アノテーションで使用した全てのLO_ID
    used_lo_ids = set([id for id in df_data[lo_id_column].to_list()])
    used_lo_ids = sorted(list(used_lo_ids)) 

    ## アノテーションで使用した全てのTEMPLATE_ID
    global used_temp_ids
    used_temp_ids = set(df_data[temp_id_column])
    
    ## temp_id から template_id_gold のindex へ変換するもの {temp_id : temp_id_gold_INDEX, temp_text}
    tempid2position = {}
    p_count = 0
    for _index, row in df_label_info.iterrows():
        if row['ID'] == '*':
            continue
        if row['ID'] in used_temp_ids:
            tempid2position[int(row['ID'])] = {'position': p_count, 'temp_text': row['TEMPLATE']}
            p_count += 1
        if row['ID'] == 999:
            break
    
    assert len(tempid2position) == len(used_temp_ids)
    #print(tempid2position)

    with open('./work/temp_id_info.json', mode='wt', encoding='utf=8')as f:
        json.dump(tempid2position, f, ensure_ascii=False, indent=2)    


    temp_id_gold = {}
    for l in used_lo_ids:
        temp_id_gold[l] = { 
                            'pm_speech': 'hoge',
                            'lo_speech': 'hoge', 
                            'temp_data': {}
                        }


    temp_id_gold = embed_temp_id_gold(df_data, temp_id_gold, tempid2position)

    """
    ## テンプレート適用がないLOIDは削除
    non_list = []
    for lo_id, dic in temp_id_gold.items():
        flag_count = dic['temp_id'].count(1)
        if flag_count == 0:
            non_list.append(lo_id)
    for i in non_list:
        temp_id_gold.pop(i, None)
    """
    
    
    #pprint.pprint(temp_id_gold, indent=2)
    with open('./work/temp_id_gold.json', mode='wt', encoding='utf=8')as f:
        json.dump(temp_id_gold, f, indent=2)
    



if __name__ == '__main__':
    main()


