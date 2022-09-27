import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from transformers import BertTokenizer
import json
import numpy as np


class SpaCEDataset(Dataset):
    def __init__(self, file, device, mode):
        self.tokenizer = BertTokenizer.from_pretrained(config.model_file,
                                                       do_lower_case=True)
        self.device = device
        self.mode = mode  # train, test, dev
        self.dataset = self.preprocess(file)

    def preprocess(self, file):
        with open(file, mode='r+',
                  encoding='utf8') as file:
            raw_data = json.load(file)

        data_set = []
        for item in raw_data:
            tokens = self.tokenizer.convert_tokens_to_ids(
                [char for char in item['context']]
            )
            if self.mode == 'dev':
                qid = item['qid']
                data_set.append([tokens, qid])
            else:
                labels_ids = [0] * len(item['context'])
                for reason in item['reasons']:
                    r_type = reason['type']
                    for fragment in reason['fragments']:
                        f_role = fragment['role']
                        rules = [
                            rule
                            for rule in config.tag_dic_list
                            if rule['role'] == f_role
                        ]
                        for i in range(len(fragment['idxes'])):
                            if i == 0:
                                labels_ids[fragment['idxes'][i]] = rules[0]['tag']
                            else:
                                labels_ids[fragment['idxes'][i]] = rules[1]['tag']
                if self.mode == 'train':
                    data_set.append([tokens, labels_ids])
                elif self.mode == 'test':
                    qid = item['qid']
                    data_set.append([tokens,
                                     labels_ids, qid])
                else:
                    raise AttributeError(
                        'The input mode: {}, is wrong.'.format(self.mode)
                    )

        return data_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def collate_fn(self, batch):
        """
        这个函数实现两个功能：\n
        1.padding：将batch中的句子padding成最长的长度\n
        2.转化为pytorch的tensor\n
        """
        sentences = [np.array(s[0]) for s in batch]
        batch_data = pad_sequence([torch.from_numpy(s)
                                   for s in sentences],
                                  batch_first=True,
                                  padding_value=self.tokenizer.pad_token_id)
        batch_data = batch_data.long()
        batch_data = batch_data.to(config.device)
        if self.mode == 'dev':
            qids = [s[-1] for s in batch]
            return [batch_data, qids]
        else:
            labels = [np.array(s[1]) for s in batch]
            batch_labels = pad_sequence([torch.from_numpy(s)
                                         for s in labels],
                                        batch_first=True,
                                        padding_value=0)
            batch_labels = batch_labels.long()
            batch_labels = batch_labels.to(config.device)
            if self.mode == 'train':
                return [batch_data, batch_labels]
            elif self.mode == 'test':
                qids = [s[2] for s in batch]
                return [batch_data, batch_labels, qids]
            else:
                raise AttributeError(
                    'The input mode: {}, is wrong.'.format(self.mode)
                )


if __name__ == '__main__':
    import random
    space = SpaCEDataset(config.test_data,
                         config.device,
                         'dev')
    with open(config.test_data, 'r+',
              encoding='utf8') as file:
        t_data = json.load(file)

    n = random.randint(0, len(t_data))
    print(t_data[n]['context'])
    print(t_data[n]['reasons'])
    for i in range(len(space.dataset[n])):
        print(space.dataset[n][i], sep='\n')
