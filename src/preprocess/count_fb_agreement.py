


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

for lo_id in lo_ids:

    df_l_g = df_lo_group.get_group(lo_id)
    
    duplicated_df = df_l_g[df_l_g.duplicated(subset=['TEMPLATE_ID'], keep=False)]

    inds = duplicated_df.index.values

    for i in range(len(inds)):

        if i == len(inds) - 1:
            break

        j = i + 1
        
        while j != len(inds):
            temp_id_1 = duplicated_df.at[inds[i], 'TEMPLATE_ID']
            temp_id_2 = duplicated_df.at[inds[j], 'TEMPLATE_ID']
            if temp_id_1 == temp_id_2:
                ref_id_1 = duplicated_df.at[inds[i], 'FB_REFERENCE_ID']
                ref_id_2 = duplicated_df.at[inds[j], 'FB_REFERENCE_ID']
                print(
                    "ind1:{}\tind2:{}\ttemp:{}\tref1:{}\tref2:{}".format(
                        i, j, temp_id_1, ref_id_1, ref_id_2
                        )
                    )

                if ref_id_1 in ref_id_2 or ref_id_2 in ref_id_1:
                    flag_list[inds[i]] = 1
                    flag_list[inds[j]] = 1

            j += 1

print(flag_list)
print(sum(flag_list), sum(flag_list)/len(flag_list)*100)



