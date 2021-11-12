


import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

#from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
#from transformers import Trainer, TrainerArguments



class TemplateIdsDataset(Dataset):
    
    def __init__(self, data: dict, tokenizer, max_token_len: int=510):
        
        self.data = data
        self.tok = tokenizer
        self.max_token_len = max_token_len


    def __len__(self):
        return len(self.data)


    def preprocess_dataset(self):
        data_ids = []
        lo_ids = []
        _lo_speeches = []
        input_ids = []
        attention_masks = []
        labels = []
        used_lo_ids = list(self.data.keys())

        # tokenizer setting
        sp_tokens = ['<PM>', '</PM>', '<LO>', '</LO>', '<FB>', '</FB>']
        self.tok.add_tokens(sp_tokens, special_tokens=True)

        data_id = 0
        for lo_id_name, lo_id_dict in self.data.items():

            for _fb_unit_id, temp_data_dict in lo_id_dict['temp_data'].items():

                lo_id = used_lo_ids.index(lo_id_name)
                pm_speech = self.data[lo_id_name]['pm_speech']
                lo_speech = self.data[lo_id_name]['lo_speech']
                ref_id = temp_data_dict['ref_id']
                label = temp_data_dict['temp_id']

                """ 
                ## PM speech + LO speech + Reference texts
                if ref_id == '*':
                    lo_speech = '<FB> ' + lo_speech + ' </FB>'
                else:    
                    lo_texts = lo_speech.split('.')
                    lo_speech = ''
                    for i, lo_text in enumerate(lo_texts):
                        if i == len(lo_texts) - 1:
                            break

                        if i in ref_id:
                            lo_text = ' <FB>' + lo_text + '. </FB>'
                        else:
                            lo_text = lo_text + '.'
                        #print("{}\t{}\n".format(i, lo_text))
                        lo_speech += lo_text

                lo_speech = ' <LO> ' + lo_speech + ' </LO>'


                pm_texts = pm_speech.split('.')

                j = 0
                input_tokens = ['foo'] * self.max_token_len
                while len(input_tokens) >= self.max_token_len - 3:
                    input_speech = '<PM> ' + '.'.join(pm_texts[j:]) + ' </PM>' + lo_speech
                    input_tokens = self.tok.tokenize(input_speech)
                    j += 1
                """

                ## Only Reference texts
                if ref_id == '*':
                    input_speech = lo_speech
                else:
                    lo_texts = lo_speech.split('.')
                    input_speech = ''
                    for i, lo_text in enumerate(lo_texts):
                        if i == len(lo_texts) - 1:
                            break
                        if i in ref_id:
                            input_speech += lo_text

                #print(input_speech)
                #print()

                encoding = self.tok(
                            input_speech,
                            padding = 'max_length',
                            #max_length = self.max_token_len,
                            return_token_type_ids = False,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                            )
                
                
                lo_ids.append(lo_id)
                _lo_speeches.append(lo_speech)
                input_ids.append(encoding['input_ids'])
                attention_masks.append(encoding['attention_mask'])
                labels.append(label)
                data_ids.append(data_id)

                data_id += 1

        data_ids = torch.tensor(data_ids)
        lo_ids = torch.tensor(lo_ids)
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.FloatTensor(labels)

        return data_ids, lo_ids, _lo_speeches, input_ids, attention_masks, labels
    

    def _make_tensor_dataset(self):
        _lo_ids, _lo_speeches, input_ids, attention_masks, labels = self.preprocess_dataset()

        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset

