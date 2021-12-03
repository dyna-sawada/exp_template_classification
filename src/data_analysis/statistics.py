

import json
import numpy as np
import argparse



def main():
    gold_data_dir = './work/temp_id_gold.json'
    temp_id_gold = json.load(open(gold_data_dir))

    info_data_dir = './work/temp_id_info.json'
    temp_id_info = json.load(open(info_data_dir))


    anno_result = []
    for ind, val in temp_id_gold.items():
        for i, v in val['temp_data'].items():
            flag_count = v['temp_id'].count(1)
            assert flag_count != 0

            anno_result.append(v['temp_id'])


    n_lo_id = len(anno_result)
    n_temp_id = len(anno_result[0])

    np_anno_result = np.array(anno_result)
    n_pos_each_temp_id = np.count_nonzero(np_anno_result > 0.5, axis=0)
    n_neg_each_temp_id = np.count_nonzero(np_anno_result < 0.5, axis=0)
    total_n_temp_id = np.sum(np_anno_result)


    print('N_batch: {}\tN_total_temp_id: {}'.format(n_lo_id, n_temp_id))
    #print(n_each_temp_id)
    assert n_temp_id == len(n_pos_each_temp_id)
    assert n_temp_id == len(n_neg_each_temp_id)

    print("temp_id\tn_positive\tn_negative\tratio_macro\tratio_micro\ttemp_text")
    for i, (n_p, n_n) in enumerate(zip(n_pos_each_temp_id, n_neg_each_temp_id)):
        for temp_id, dic in temp_id_info.items():
            if dic['position'] == i:
                temp_text = dic['temp_text']

                print('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{}'
                        .format(
                            temp_id,
                            n_p,
                            n_n,
                            n_p / (n_p + n_n) * 100,
                            n_p/total_n_temp_id*100,
                            temp_text
                        )
                    )




if __name__ == '__main__':
    main()


