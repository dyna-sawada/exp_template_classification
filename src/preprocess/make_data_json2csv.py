

import re
import json
import csv


def get_motion_pm_lo_id(LO_ID):
    #LO_ID = motion + '_LO_' + pm_id + '_' + lo_id

    if LO_ID == '*':
        return '*'

    m = re.search('(.*)_LO_([0-9])_(.*)', LO_ID)
    motion, pm_id, lo_id = \
        m.group(1), m.group(2), m.group(3)
    
    return motion, pm_id, lo_id


def main():
    data_file = './work/temp_id_gold.json'
    data_json = json.load(open(data_file))

    data_csv = []
    data_id = 0
    for LO_ID, d_j in data_json.items():
        
        motion, pm_id, lo_id = get_motion_pm_lo_id(LO_ID)
        #print(motion)
        pm_speech = d_j['pm_speech']
        lo_speech = d_j['lo_speech']

        for _i, d in d_j['temp_data'].items():
            temp_id = d['temp_id']
            ref_id = d['ref_id']

            d_c = [data_id, motion, pm_id, lo_id, pm_speech, lo_speech, ref_id, temp_id]
            data_csv.append(d_c)
            data_id += 1


    assert len(data_csv) == 762

    print(data_csv)
    with open('./work/temp_id_gold.csv', 'w')as f:
        writer = csv.writer(f)
        writer.writerows(data_csv)



if __name__ == '__main__':
    main()