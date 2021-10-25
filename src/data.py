


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
        _lo_ids = []
        _lo_speeches = []
        input_ids = []
        attention_masks = []
        labels = []
        
        # tokenizer setting
        sp_tokens = ['<PM>', '</PM>', '<LO>', '</LO>', '<FB>', '</FB>']
        self.tok.add_tokens(sp_tokens, special_tokens=True)

        for lo_id, lo_id_dict in self.data.items():
            for _fb_unit_id, temp_data_dict in lo_id_dict['temp_data'].items():
                pm_speech = self.data[lo_id]['pm_speech']
                lo_speech = self.data[lo_id]['lo_speech']
                ref_id = temp_data_dict['ref_id']
                label = temp_data_dict['temp_id']

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


                encoding = self.tok(
                            input_speech,
                            padding = 'max_length',
                            #max_length = self.max_token_len,
                            return_token_type_ids = False,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                            )
                
                
                _lo_ids.append(lo_id)
                _lo_speeches.append(lo_speech)
                input_ids.append(encoding['input_ids'])
                attention_masks.append(encoding['attention_mask'])
                labels.append(label)
            
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.FloatTensor(labels)

        return _lo_ids, _lo_speeches, input_ids, attention_masks, labels
    

    def make_tensor_dataset(self):
        _lo_ids, _lo_speeches, input_ids, attention_masks, labels = self.preprocess_dataset()

        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset

