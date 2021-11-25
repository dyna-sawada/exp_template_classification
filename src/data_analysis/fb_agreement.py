


import pandas as pd
import numpy as np
import json


def main():
    
    DATA_PATH = './data/diagnostic_comments.json'
    annotation_datas = json.load(open(DATA_PATH))
    
    n_all = 0
    in_appropriate_index = []
    for lo_id, anno_data in annotation_datas.items():
        diagnostic_comments = anno_data['diagnostic_comments']

        for d_idx, d_comment in enumerate(diagnostic_comments):
            template_annotation = d_comment['template_annotation'][0]
            is_valid = template_annotation['is_valid']
            is_understandable = template_annotation['is_understandable']
            is_not_duplicated = template_annotation['is_not_duplicated']

            if is_valid == False or is_understandable == False or is_not_duplicated == False:
                in_appropriate_index.append(d_idx+n_all)

        n_all += len(diagnostic_comments)


    print(n_all)
    print(in_appropriate_index)

    flag_list = [0] * n_all

    
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
        indexes_2d = list(set(list(map(tuple, indexes_2d))))
        
        if len(indexes_2d) < 1:
            inc += len(diagnostic_comments)
            continue

        print(indexes_2d)

        for indexes_dup in indexes_2d:
            for i, _index_dup in enumerate(indexes_dup):
                if i == len(indexes_dup):
                    break
                j = i + 1
                
                while j != len(indexes_dup):

                    ref_id_1 = diagnostic_comments[i]['target_sent_idx']
                    ref_id_2 = diagnostic_comments[j]['target_sent_idx']
                    
                    if ref_id_1 == ref_id_2:
                        temp_id_1 = diagnostic_comments[i]['template_annotation'][0]['template_number']
                        temp_id_2 = diagnostic_comments[j]['template_annotation'][0]['template_number']

                        print(
                            "ind1:{}\tind2:{}\tref:{}\ttemp1:{}\ttemp2:{}".format(
                                i, j, ref_id_1, temp_id_1, temp_id_2
                                )
                            )

                        if temp_id_1 == temp_id_2:
                            flag_list[i + inc] = 1
                            flag_list[j + inc] = 1
                    j += 1
        
        inc += len(diagnostic_comments)

        count_duplicated_all += len(indexes_2d)

        #print(indexes_2d)
    for i_a_ind in in_appropriate_index:
        flag_list[i_a_ind] = 0
    print(flag_list)

    count_duplicated_all = count_duplicated_all - len(in_appropriate_index)
    n_all = n_all - len(in_appropriate_index)
    
    print(
        "ALL: {}\tDuplicated: {}\tN: {}\tRatio: {:.2f}".format(
            n_all, count_duplicated_all, sum(flag_list), sum(flag_list)/count_duplicated_all * 100
            )
        )
    



if __name__ == '__main__':
    main()

