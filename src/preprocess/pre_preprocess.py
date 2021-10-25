

import re
import pandas as pd
import json
import glob
from collections import defaultdict




def component_id2lo_id(component_id):
    if component_id == '*':
        return '*'

    m = re.search('(.*)_LO_([0-9])_(.*)_([0-9])_(.*)', component_id)
    motion, pm_id, lo_id, _comp_id, _judge_id = \
        m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    LO_ID = motion + '_LO_' + pm_id + '_' + lo_id

    return LO_ID


def embed_temp_id_gold(df_anno_motion, temp_id_gold, tempid2position):
    for i, row in df_anno_motion.iterrows():
        component_id = row[component_id_column]
        _fb_id = row[fb_id_column]
        fb_comment = row[fb_comment_column]
        temp_id = row[temp_id_column]
        validity_1 = row[validity_1_column]
        validity_2 = row[validity_2_column]
        validity_3 = row[validity_3_column]
        anno_done = row[anno_done_column]

        ## アノテーション結果に関係しない行はスキップ．
        if i <= 1 or component_id == '*':
            continue
        if validity_1 == 1 or validity_2 == 1 or validity_3 == 1:
            continue

        ## アノテーション済みだけどテンプレ選択していないものは「999.該当なし」に設定
        if anno_done != '*' and fb_comment != 'なし' and validity_1 != 1 and validity_2 != 1 and validity_3 != 1 and temp_id == '*':
            temp_id = 999
        
        if temp_id == '*':
            continue
        
        LO_ID = component_id2lo_id(component_id)

        temp_position = tempid2position[temp_id]['position']
        temp_id_gold[LO_ID]['temp_id'][temp_position] = 1
    
    return temp_id_gold




def main():
    
    ###  setting  ###
    SPEECH_DATA_PATH = '../data/speech_text/debate_data.json'
    speech_dict = json.load(open(SPEECH_DATA_PATH))

    ANNOTATION_DATA_PATH = './data/テンプレートアノテーション_20211020.xlsx'

    df_HW = pd.read_excel(
        ANNOTATION_DATA_PATH,
        sheet_name='★統合-HW-割り振り',
        index_col=None,
        header=2
    )
    df_DP = pd.read_excel(
        ANNOTATION_DATA_PATH,
        sheet_name='★統合-DP-割り振り',
        index_col=None,
        header=2
    )
    df_label = pd.read_excel(
        ANNOTATION_DATA_PATH,
        sheet_name='テンプレート',
        index_col=None
    )

    df_HW = df_HW.fillna('*')
    df_DP = df_DP.fillna('*')
    df_label = df_label.fillna('*')

    ## column name settings
    global component_id_column
    global temp_id_column
    global fb_id_column
    global fb_comment_column
    global validity_1_column
    global validity_2_column
    global validity_3_column
    global anno_done_column
    component_id_column = '問題番号'
    temp_id_column = '\n各フィードバックコメントがどのテンプレートに分類されるかを分類したうえで診断部へのテンプレートの適用（まれに複数ある場合や、どちらにも当てはまらない場合もあり→「備考」欄に記入）\n＋各々のXとYを考える。（M～O列）'
    fb_id_column = 'フィードバックコメントの番号'
    fb_comment_column = 'フィードバックコメント '
    validity_1_column = '1. フィードバックコメントの妥当性の判断'     # FBコメントが妥当でない
    validity_2_column = 'Unnamed: 12'                           # FBコメントの指摘している内容がわからない
    validity_3_column = 'Unnamed: 13'                           # FBコメントの重複
    anno_done_column = '実施済みの場合（1）'
    

    ## アノテーションで使用した全てのLO_ID
    used_hw_ids = [component_id2lo_id(id) for id in df_HW[component_id_column].to_list()]
    used_dp_ids = [component_id2lo_id(id) for id in df_DP[component_id_column].to_list()]
    used_lo_ids = set(used_hw_ids) | set(used_dp_ids)
    used_lo_ids.remove('*')
    used_lo_ids = sorted(list(used_lo_ids))    
    #print(len(used_lo_ids))    -> 165


    ## アノテーションで使用した全てのTEMPLATE_ID
    results_hw = df_HW[temp_id_column]
    results_dp = df_DP[temp_id_column]  
    
    used_temp_ids = set(results_hw) | set(results_dp)
    used_temp_ids.remove('*')
    #used_temp_ids.remove(120.0)     # 応急処置
    print(used_temp_ids)
    print(len(used_temp_ids))
    

    ## LO_ID / LO_SPEECH / TEMPLATE_ID をまとめたデータ
    temp_id_gold = {}
    for l in used_lo_ids:
        lo_speech = speech_dict[l]['speech']
        temp_id_gold[l] = {
                            'lo_speech': lo_speech, 
                            'temp_id': [0] * len(used_temp_ids)
                        }


    temp_label = {}
    for _index, row in df_label.iterrows():
        if row['ID'] == '*':
            continue
        temp_label[int(row['ID'])] = row['テンプレート']
        if row['ID'] == 999:
            break
    
    ## temp_id から template_id_gold のindex へ変換するもの {temp_id : temp_id_gold_INDEX, temp_text}
    tempid2position = {}
    p_count = 0
    for _index, row in df_label.iterrows():
        if row['ID'] == '*':
            continue
        if row['ID'] in used_temp_ids:
            tempid2position[int(row['ID'])] = {'position': p_count, 'temp_text': row['テンプレート']}
            p_count += 1
        if row['ID'] == 999:
            break
    
    assert len(tempid2position) == len(used_temp_ids)

    with open('./work/temp_id_info.json', mode='wt', encoding='utf=8')as f:
        json.dump(tempid2position, f, ensure_ascii=False, indent=2)    


    temp_id_gold = embed_temp_id_gold(df_HW, temp_id_gold, tempid2position)
    temp_id_gold = embed_temp_id_gold(df_DP, temp_id_gold, tempid2position)


    ## テンプレート適用がないLOIDは削除
    non_list = []
    for lo_id, dic in temp_id_gold.items():
        flag_count = dic['temp_id'].count(1)
        if flag_count == 0:
            non_list.append(lo_id)
    for i in non_list:
        temp_id_gold.pop(i, None)


    with open('./work/temp_id_gold.json', mode='wt', encoding='utf=8')as f:
        json.dump(temp_id_gold, f, indent=2)





if __name__ == '__main__':
    main()


