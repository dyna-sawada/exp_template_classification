


import pandas as pd
import numpy as np



DATA_PATH = './data/テンプレートアノテーション_20211020_整形後.xlsx'
df_data = pd.read_excel(DATA_PATH)
lo_ids = sorted(
            list(
                set(
                    df_data['LO_ID'].tolist()
                    )
                )
            )

df_lo_group = df_data.groupby('LO_ID')

flag_list = [0] * len(df_data.index)


count_all = 0
for lo_id in lo_ids:

    df_l_g = df_lo_group.get_group(lo_id)
    
    duplicated_df = df_l_g[df_l_g.duplicated(subset=['FB_REFERENCE_ID'], keep=False)]

    inds = duplicated_df.index.values

    for i in range(len(inds)):

        if i == len(inds) - 1:
            break

        j = i + 1
        
        while j != len(inds):
            ref_id_1 = duplicated_df.at[inds[i], 'FB_REFERENCE_ID']
            ref_id_2 = duplicated_df.at[inds[j], 'FB_REFERENCE_ID']
            if ref_id_1 == ref_id_2:
                temp_id_1 = duplicated_df.at[inds[i], 'TEMPLATE_ID']
                temp_id_2 = duplicated_df.at[inds[j], 'TEMPLATE_ID']
        
                print(
                    "ind1:{}\tind2:{}\tref:{}\ttemp1:{}\ttemp2:{}".format(
                        i, j, ref_id_1, temp_id_1, temp_id_2
                        )
                    )

                if temp_id_1 == temp_id_2:
                    flag_list[inds[i]] = 1
                    flag_list[inds[j]] = 1

            j += 1

    count_all += len(inds)


print(flag_list)
print(sum(flag_list), count_all, sum(flag_list)/count_all * 100)



