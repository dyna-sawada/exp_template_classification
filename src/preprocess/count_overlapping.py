

import pandas as pd
import re

from preprocess import embed_temp_id_gold


def embed_result_from_df(df:pd.DataFrame, results:list):
    for i, row in df.iterrows():
        result = []
        component_id = row[component_id_column]
        _fb_id = row[fb_id_column]
        fb_reference = row[fb_reference_column]
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

        m = re.search('(.*)_LO_([0-9])_(.*)_([0-9])_(.*)', component_id)
        motion, pm_id, lo_id, comp_id, _judge_id = \
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        component_id = motion + '_LO_' + pm_id + '_' + lo_id + '_' + comp_id

        result.append(component_id)
        result.append(fb_reference)
        result.append(temp_id)
        results.append(result)

    return results



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
component_id_column = '問題番号'
temp_id_column = '\n各フィードバックコメントがどのテンプレートに分類されるかを分類したうえで診断部へのテンプレートの適用（まれに複数ある場合や、どちらにも当てはまらない場合もあり→「備考」欄に記入）\n＋各々のXとYを考える。（M～O列）'
fb_id_column = 'フィードバックコメントの番号'
fb_reference_column = '対象文'
fb_comment_column = 'フィードバックコメント '
validity_1_column = '1. フィードバックコメントの妥当性の判断'     # FBコメントが妥当でない
validity_2_column = 'Unnamed: 12'                           # FBコメントの指摘している内容がわからない
validity_3_column = 'Unnamed: 13'                           # FBコメントの重複
anno_done_column = '実施済みの場合（1）'


results = []
results = embed_result_from_df(df_HW, results)
results = embed_result_from_df(df_DP, results)
print(len(results))
df_results = pd.DataFrame(results)
#print(df_results)

#df_results.to_csv('./data/temp_id_results.csv', header=False, index=False)
