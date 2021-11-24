


import pandas as pd
import numpy as np
import json


def main():
    
    DATA_PATH = './data/diagnostic_comments.json'
    annotation_datas = json.load(open(DATA_PATH))
    
    flag_list = [0] * 1265

    count_all = 0
    count_duplicated_all = 0
    inc = 0
    for lo_id, anno_data in annotation_datas.items():
        #print(lo_id)
        #print(anno_data)
        diagnostic_comments = anno_data['diagnostic_comments']
        target_sent_idx = [d_comment['target_sent_idx'] for d_comment in diagnostic_comments]
        #print(target_sent_idx)
        
        indexes_2d = []
        for t_s_idx in target_sent_idx:
            indexes = [i for i, x in enumerate(target_sent_idx) if x == t_s_idx]
            if len(indexes) >= 2:
                indexes_2d.append(indexes)
        

        for i in range(len(indexes_2d)):

            if i == len(indexes_2d) - 1:
                break

            j = i + 1
            
            while j != len(indexes_2d):
                ref_id_1 = diagnostic_comments[i]['target_sent_idx']
                ref_id_2 = diagnostic_comments[j]['target_sent_idx']
                if ref_id_1 == ref_id_2:
                    temp_id_1 = diagnostic_comments[i]['template_annotation'][0]['template_number']
                    temp_id_2 = diagnostic_comments[j]['template_annotation'][0]['template_number']
            

                    #print(
                    #    "ind1:{}\tind2:{}\tref:{}\ttemp1:{}\ttemp2:{}".format(
                    #        i, j, ref_id_1, temp_id_1, temp_id_2
                    #        )
                    #    )
                    

                    if temp_id_1 == temp_id_2:
                        flag_list[i + inc] = 1
                        flag_list[j + inc] = 1

                j += 1
            
        inc += len(diagnostic_comments)
            
        
        count_all += len(diagnostic_comments)
        count_duplicated_all += len(indexes_2d)

        #print(indexes_2d)
    
    
    print(
        "ALL: {}\tDuplicated: {}\tN: {}\tRatio: {:.2f}".format(
            count_all, count_duplicated_all, sum(flag_list), sum(flag_list)/count_duplicated_all * 100
            )
        )




if __name__ == '__main__':
    main()

