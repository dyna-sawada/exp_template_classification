
import json
import pandas as pd



fb_data_dir = './data/diagnostic_comments.json'
temp_datas = json.load(open(fb_data_dir))




data_2d_anno1, data_2d_anno2 = [], []
for lo_id, temp_data in temp_datas.items():
    diagnostic_comments = temp_data["diagnostic_comments"]

    for d_comment in diagnostic_comments:
        template_annotation = d_comment["template_annotation"]

        if len(template_annotation) <= 1:
            continue
        
        assert len(template_annotation) == 2


        for t_annotation in template_annotation:
            template_number = t_annotation["template_number"]
            
            #if template_number == None:
            #    continue

            pm_speech = temp_data["speech"]["pm_speech"]
            lo_speech = temp_data["speech"]["lo_speech"]["speech"]
            lo_sentences = temp_data["speech"]["lo_speech"]["sentences"]

            target_sent_idx = d_comment["target_sent_idx"]
            target_speech = ' '.join(
                                    [lo_sentences[t_s_idx] for t_s_idx in target_sent_idx]
                                    )
            original_comment = d_comment["original_comment"]
            fixed_comment = t_annotation["fixed_comment_jp"]
            annotator = t_annotation["annotator"]
            template_comment = t_annotation["template_comment_jp"]

            data = [
                lo_id,
                pm_speech,
                lo_speech,
                target_speech,
                original_comment,
                fixed_comment,
                template_number,
                template_comment
            ]

            if annotator == 'annotatorA':
                data_2d_anno1.append(data)
            else:
                data_2d_anno2.append(data)



print('N\t{}\t{}'.format(len(data_2d_anno1), len(data_2d_anno2)))



df_anno1 = pd.DataFrame(data_2d_anno1)
df_anno2 = pd.DataFrame(data_2d_anno2)
df_anno1.to_excel('./work/for_redundancy_samples.xlsx', header=None, index=None)
df_anno2.to_excel('./work/for_redundancy_samples2.xlsx', header=None, index=None)

