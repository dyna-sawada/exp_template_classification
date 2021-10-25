

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
    n_each_temp_id = np.sum(np_anno_result, axis=0)
    total_n_temp_id = np.sum(np_anno_result)


    print('N_batch: {}\tN_total_temp_id: {}'.format(n_lo_id, n_temp_id))
    #print(n_each_temp_id)
    assert n_temp_id == len(n_each_temp_id)

    print("temp_id\tn_temp_id\tratio\ttemp_text")
    for i, n in enumerate(n_each_temp_id):
        for temp_id, dic in temp_id_info.items():
            if dic['position'] == i:
                temp_text = dic['temp_text']

                print('{}\t{}\t{:.2f}%\t{}'
                        .format(
                            temp_id,
                            n,
                            n/total_n_temp_id*100,
                            temp_text
                        )
                    )




if __name__ == '__main__':
    main()


