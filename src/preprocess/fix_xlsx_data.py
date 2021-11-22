


import json
import pandas as pd
import re


def component_id2lo_id(component_id):
    if component_id == '*':
        return '*'

    m = re.search('(.*)_LO_([0-9])_(.*)_([0-9])_(.*)', component_id)
    motion, pm_id, lo_id, comp_id, judge_id = \
        m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    LO_ID = motion + '_LO_' + pm_id + '_' + lo_id

    return LO_ID, comp_id, judge_id


def embed_fix_data_lists(df_anno_motion: pd.DataFrame, fix_data_lists: list):

    ## column name settings
    component_id_column = '問題番号'
    pm_en_speech_column = 'スピーチ'
    pm_jp_speech_column = 'スピーチ_DeepLの翻訳結果'
    evaluation_target_column = '評価対象'
    lo_jp_speech_column = '評価対象_DeepLの翻訳結果'
    fb_id_column = 'フィードバックコメントの番号'
    fb_reference_column = '対象文'
    temp_id_column = '\n各フィードバックコメントがどのテンプレートに分類されるかを分類したうえで診断部へのテンプレートの適用（まれに複数ある場合や、どちらにも当てはまらない場合もあり→「備考」欄に記入）\n＋各々のXとYを考える。（M～O列）'
    fb_comment_column = 'フィードバックコメント '
    fb_comment_v2_column = '修正結果'
    x_c1_column = 'X/C1'
    y_c2_column = 'Y/C2'
    z_column = 'Z'
    validity_1_column = '1. フィードバックコメントの妥当性の判断'     # FBコメントが妥当でない
    validity_2_column = 'Unnamed: 12'                           # FBコメントの指摘している内容がわからない
    validity_3_column = 'Unnamed: 13'                           # FBコメントの重複
    anno_done_column = '実施済みの場合（1）'


    for i, row in df_anno_motion.iterrows():
        component_id = row[component_id_column]
        pm_en_speech = row[pm_en_speech_column]
        pm_jp_speech = row[pm_jp_speech_column]
        eval_target = row[evaluation_target_column]
        lo_jp_speech = row[lo_jp_speech_column]
        fb_id = row[fb_id_column]
        fb_reference_pre = row[fb_reference_column]
        temp_id = row[temp_id_column]
        fb_comment = row[fb_comment_column]
        fb_comment_v2 = row[fb_comment_v2_column]
        x_c1 = row[x_c1_column]
        y_c2 = row[y_c2_column]
        z = row[z_column]
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
        
        if fb_comment_v2 == '*':
            fb_comment_v2 = ''
        if x_c1 == '*':
            x_c1 = ''
        if y_c2 == '*':
            y_c2 = ''
        if z == '*':
            z = ''

        eval_target = eval_target.split('\n')
        fb_reference_pre = [int(fr) if fr != "*" else fr for fr in fb_reference_pre.split(',')]
        fb_reference = []
        for i, evl_tgt in enumerate(eval_target):
            #print(evl_tgt)
            try:
                evl_id = int(evl_tgt[0])
                if evl_id in fb_reference_pre:
                    fb_reference.append(str(i))
            except:
                continue
        fb_reference = ', '.join(fb_reference)

        LO_ID, comp_id, judge_id = component_id2lo_id(component_id)
        lo_en_speech = speech_dict[LO_ID]['speech']

        data_list = [
                    LO_ID, comp_id, judge_id, pm_en_speech, pm_jp_speech,
                    lo_en_speech, eval_target, lo_jp_speech, fb_id,
                    fb_reference, temp_id, fb_comment, fb_comment_v2, x_c1, y_c2, z                    
                    ]
        fix_data_lists.append(data_list)

    return fix_data_lists



def main():

    ###  setting  ###
    global speech_dict
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


    fix_data_lists = []
    fix_data_lists = embed_fix_data_lists(df_HW, fix_data_lists)
    fix_data_lists = embed_fix_data_lists(df_DP, fix_data_lists)

    header_list_1 = [
                    'LO_ID', 'COMPONENT_ID', 'JUDGE_ID', 'PM_SPEECH', 'PM_SPEECH_JP', 
                    'LO_SPEECH', 'LO_COMPONENT_SPEECH', 'LO_SPEECH_JP', 'FB_ID',
                    'FB_REFERENCE_ID', 'TEMPLATE_ID', 'FEEDBACK_COMMENT',
                    'FEEDBACK_COMMENT_RE', 'X/C1', 'Y/C2', 'Z'
                    ]
    df_data = pd.DataFrame(fix_data_lists, columns=header_list_1)
    #print(len(df_data))
    #print(df_data)


    header_list_2 = ['ID', 'TEMPLATE']

    used_temp_ids = set(df_data['TEMPLATE_ID'])
    temp_infos = []
    for _i, row in df_label.iterrows():
        if row['ID'] in used_temp_ids:
            temp_info = [
                            row['ID'],
                            row['テンプレート']
                         ]
            temp_infos.append(temp_info)

        if row['ID'] == 999.0:
            break


    df_temp_info = pd.DataFrame(temp_infos, columns=header_list_2)

    #df_new.to_excel('./data/テンプレートアノテーション_20211020_整形後.xlsx', header=header_list, index=False, encoding='utf_8_sig')
    
    with pd.ExcelWriter(
                        './data/テンプレートアノテーション_20211020_整形後.xlsx',
                        mode='w',
                        ) as writer:
        df_data.to_excel(writer, sheet_name='Sheet1', index=False, header=header_list_1)
        df_temp_info.to_excel(writer, sheet_name='Sheet2', index=False, header=header_list_2)
        writer.save()
    


if __name__ == '__main__':
    main()