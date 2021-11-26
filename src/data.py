


from numpy import logical_xor
import tqdm
import nltk

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, dataset, random_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

#from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
#from transformers import Trainer, TrainerArguments



class TemplateIdsDataset(Dataset):
    
    def __init__(self, args, data: dict, tokenizer, max_token_len: int=510):
        
        self.args = args
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
        sp_token_positions = []

        # tokenizer setting
        sp_tokens = [
                        '<PM>', '</PM>',
                        '<LO>', '</LO>',
                        '<FB>', '</FB>',
                        '<CLAIM>', '<CLAIM>',
                        '<PREMISE>', '</PREMISE>',
                        '<EXAMPLE>', '</EXAMPLE>',
                        '<STANCE>', '</STANCE>'
                    ]

        self.tok.add_tokens(sp_tokens, special_tokens=True)
        sp_tokens_ids = self.tok.convert_tokens_to_ids(sp_tokens)


        data_id = 0
        for lo_id_name, lo_id_dict in self.data.items():
            lo_id = used_lo_ids.index(lo_id_name)
            pm_speech = self.data[lo_id_name]['pm_speech']
            lo_speech = self.data[lo_id_name]['lo_speech']

            pm_texts = nltk.sent_tokenize(pm_speech)
            lo_texts = nltk.sent_tokenize(lo_speech)
            len_lo_speech = len(lo_texts)


            if self.args.argument_structure:
                arg_structure = self.data[lo_id_name]['argument_structure']
                adus = arg_structure['adus']
                adu_info_list = [''] * len_lo_speech
                
                for adu in adus.values():
                    adu_type = adu['adu_type']
                    sent_idx = adu['sent_idx']
                    for s_idx in sent_idx:
                        adu_info_list[s_idx] = adu_type

                for lo_txt_id in range(len_lo_speech):
                    if adu_info_list[lo_txt_id] == 'Claim':
                        lo_texts[lo_txt_id] = ' <CLAIM> ' + lo_texts[lo_txt_id] + ' </CLAIM> '
                    elif adu_info_list[lo_txt_id] == 'Premise':
                        lo_texts[lo_txt_id] = ' <PREMISE> ' + lo_texts[lo_txt_id] + ' </PREMISE> '
                    elif adu_info_list[lo_txt_id] == 'Example':
                        lo_texts[lo_txt_id] = ' <EXAMPLE> ' + lo_texts[lo_txt_id] + ' </EXAMPLE> '
                    elif adu_info_list[lo_txt_id] == 'Stance':
                        lo_texts[lo_txt_id] = ' <STANCE> ' + lo_texts[lo_txt_id] + ' </STANCE> '
                
            
            for temp_data_dict in lo_id_dict['temp_data'].values():
                ref_id = temp_data_dict['ref_id']
                label = temp_data_dict['temp_id']

                lo_texts_with_fb = [l_t for l_t in lo_texts]
                ref_info_list = [0] * len_lo_speech

                if ref_id != '*':
                    for r_id in ref_id:
                        ref_info_list[r_id] = 1
                    
                    for lo_txt_id in range(len_lo_speech):
                        if ref_info_list[lo_txt_id] == 1:
                            lo_texts_with_fb[lo_txt_id] = ' <FB> ' + lo_texts_with_fb[lo_txt_id] + ' </FB> '
                            
                    lo_speech_with_sp = ' '.join(lo_texts_with_fb)

                else:
                    lo_speech_with_sp = ' '.join(lo_texts_with_fb)
                    lo_speech_with_sp = ' <FB> ' + lo_speech_with_sp + ' </FB> '

                lo_speech_with_sp = ' <LO> ' + lo_speech_with_sp + ' </LO>'

                j = 0
                input_tokens = ['foo'] * self.max_token_len
                while len(input_tokens) >= self.max_token_len - 3:
                    input_speech = '<PM> ' + ' '.join(pm_texts[j:]) + ' </PM>' + lo_speech_with_sp
                    input_tokens = self.tok.tokenize(input_speech)
                    j += 1


                #print(input_speech)
                #print()

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
                """


                encoding = self.tok(
                            input_speech,
                            padding = 'max_length',
                            #max_length = self.max_token_len,
                            return_token_type_ids = False,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                            )
                input_id = encoding['input_ids']
                attention_mask = encoding['attention_mask']
                
                lo_ids.append(lo_id)
                _lo_speeches.append(lo_speech)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
                data_ids.append(data_id)


                if self.args.encoder_out == 'fb':
                    start_fb_positions = (input_id == sp_tokens_ids[-2]).nonzero(as_tuple=True)[1]
                    end_fb_positions = (input_id == sp_tokens_ids[-1]).nonzero(as_tuple=True)[1]
                    
                    sp_token_position = torch.cat((start_fb_positions.unsqueeze(0), end_fb_positions.unsqueeze(0)), 0)
                    sp_token_position = torch.t(sp_token_position)

                    sp_token_positions.append(sp_token_position)


                data_id += 1


        data_ids = torch.tensor(data_ids)
        lo_ids = torch.tensor(lo_ids)
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.FloatTensor(labels)


        if self.args.encoder_out == 'fb':
            sp_token_positions = pad_sequence(sp_token_positions, batch_first=True, padding_value=0)
        else:
            sp_token_positions = torch.zeros(data_ids.size()[0], 7, 2)
            

        return data_ids, lo_ids, _lo_speeches, input_ids, attention_masks, sp_token_positions, labels
    


    def _make_tensor_dataset(self):
        _data_ids, _lo_ids, _lo_speeches, input_ids, attention_masks, _sp_token_positions, labels = self.preprocess_dataset()
        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset



    def save_dataset(self):
        data_ids, lo_ids, _lo_speeches, input_ids, attention_masks, sp_token_positions, labels = \
            self.preprocess_dataset()
        
        D = TensorDataset(data_ids, lo_ids, input_ids, attention_masks, sp_token_positions, labels)
        torch.save(
            D,
            '{}/datasets_{}.pt'.format(
                self.args.model_dir,
                self.args.encoder_out
            )
        )
