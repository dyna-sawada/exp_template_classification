

import json
import numpy as np
import argparse



def main():
    gold_data_dir = './work/temp_id_gold.json'
    temp_id_gold = json.load(open(gold_data_dir))

    info_data_dir = './data/temp_id_info.json'
    temp_id_info = json.load(open(info_data_dir))


    anno_result_multi = []
    temp_positions = []
    n_result = 0
    for ind, val in temp_id_gold.items():
        for i, v in val['temp_data'].items():
            flag_count = v['temp_id'].count(1)
            assert flag_count != 0

            if flag_count >= 2:
                anno_result_multi.append(v['temp_id'])
                
                temp_position = [i for i, x in enumerate(v['temp_id']) if x == 1]
                temp_positions.append(temp_position)

            n_result += 1

    assert len(anno_result_multi) == len(temp_positions)

    n_result_multi = len(anno_result_multi)

    print(
        "N_all: {}\tN_multi: {}\tRatio: {:.2f} %".format(
            n_result,
            n_result_multi,
            n_result_multi/n_result*100
            )
        )
    print()

    print("Details")
    for i, t_p in enumerate(temp_positions):
        print("sample: {}".format(i))

        for p in t_p:
            for _temp_id, dic in temp_id_info.items():
                if dic['position'] == p:
                    print("{}\t{}".format(p, dic['temp_text']))
        print()




if __name__ == '__main__':
    main()
