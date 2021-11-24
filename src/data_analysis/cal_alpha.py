


import pandas as pd
import numpy as np
import nltk
from nltk.metrics import agreement, masi_distance, jaccard_distance
from nltk.metrics.agreement import AnnotationTask



def main():
    DATA_PATH = './data/テンプレートアノテーション_20211020_整形後.xlsx'
    df_data = pd.read_excel(DATA_PATH)
    df_data = df_data.fillna('*')

    judge_ids = ['judgeA', 'judgeB', 'judgeC', 'judgeD']
    df_judge_group = df_data.groupby('JUDGE_ID')

    task_data = []
    for j_id in judge_ids:
        df_j_g = df_judge_group.get_group(j_id)
        lo_ids = sorted(
            list(
                set(
                    df_j_g['LO_ID'].tolist()
                )
            )
        )
        df_j_g_group = df_j_g.groupby('LO_ID')

        for lo_id in lo_ids:
            df_l_j_g = df_j_g_group.get_group(lo_id)
            ref_ids = list(
                set(
                    df_l_j_g['FB_REFERENCE_ID'].tolist()
                )
            )
            df_l_j_g_group = df_l_j_g.groupby('FB_REFERENCE_ID')

            for ref_id in ref_ids:
                df_l_j_r_g = df_l_j_g_group.get_group(ref_id)

                temp_list = []
                for _ind, row in df_l_j_r_g.iterrows():
                    temp_id = row[10]
                    temp_list.append(temp_id)

                #print(temp_list)

                t_data = (
                    f'{j_id}',
                    f'{lo_id}' + '_' + f'{ref_id}',
                    frozenset(temp_list)
                )
        
                task_data.append(t_data)

    print(task_data)


    """
    task_data = [
                ('coder1','video0',frozenset(['food','sports'])),
                ('coder1','video1',frozenset(['sports'])),
                ('coder3','video1',frozenset(['sports', 'education'])),
                ('coder2','video2',frozenset(['education', 'family'])),
                ('coder3','video2',frozenset(['family', 'food']))
                ]
    """

    jaccard_task = AnnotationTask(data=task_data,distance = jaccard_distance)
    masi_task = AnnotationTask(data=task_data,distance = masi_distance)

    #print(f"Fleiss's Kappa using Jaccard: {jaccard_task.multi_kappa()}")
    #print(f"Fleiss's Kappa using MASI: {masi_task.multi_kappa()}")
    print(f"Krippendorff's Alpha using Jaccard: {jaccard_task.alpha()}")
    print(f"Krippendorff's Alpha using MASI: {masi_task.alpha()}")




if __name__ == '__main__':
    main()

