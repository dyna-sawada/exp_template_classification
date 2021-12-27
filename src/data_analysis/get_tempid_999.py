
import json
import pandas as pd



fb_data_dir = './data/diagnostic_comments.json'
temp_datas = json.load(open(fb_data_dir))




data_2d = []
for lo_id, temp_data in temp_datas.items():
    diagnostic_comments = temp_data["diagnostic_comments"]

    for d_comment in diagnostic_comments:
        template_annotation = d_comment["template_annotation"]

        for t_annotation in template_annotation:
            template_number = t_annotation["template_number"]
            
            if template_number == None:
                continue

            if int(template_number) == 999:
                pm_speech = temp_data["speech"]["pm_speech"]
                lo_speech = temp_data["speech"]["lo_speech"]["speech"]
                lo_sentences = temp_data["speech"]["lo_speech"]["sentences"]

                target_sent_idx = d_comment["target_sent_idx"]
                target_speech = ' '.join(
                                        [lo_sentences[t_s_idx] for t_s_idx in target_sent_idx]
                                        )
                original_comment = d_comment["original_comment"]
                fixed_comment = t_annotation["fixed_comment"]
                annotator = t_annotation["annotator"]

                data = [
                    annotator,
                    lo_id,
                    pm_speech,
                    lo_speech,
                    target_speech,
                    original_comment,
                    fixed_comment
                ]
                data_2d.append(data)




print(len(data_2d))
print(data_2d[0])

df = pd.DataFrame(data_2d)
#df.to_excel('./work/temp_id_999_samples.xlsx', header=None, index=None)

