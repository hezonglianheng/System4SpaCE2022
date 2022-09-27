import torch
from torch.utils.data import Dataset
import config
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class SpaCEDataset(Dataset):
    def __init__(self, file, type, device,
                 tokenizer, mode):
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device
        if type == 'A':
            self.tags = config.tags4a
        elif type == 'B':
            self.tags = config.tags4b
        elif type == 'C':
            self.tags = config.tags4c
        else:
            raise AttributeError(
                'the input type: {}, is wrong.'.format(type)
            )
        self.dataset = self.preprocess(file)

    def preprocess(self, data_file):
        with open(data_file, 'r+',
                  encoding='utf8') as file:
            original_data = json.load(file)

        data_set = []
        for item in original_data:
            tokens_ids = self.tokenizer.convert_tokens_to_ids(
                [char for char in item['context']]
            )
            qids = item['qid']
            if self.mode == 'dev':
                data_set.append([tokens_ids, qids])
            else:
                tokens_labels = [0] * len(item['context'])
                for reason in item['reasons']:
                    for f in reason['fragments']:
                        f_tags = self.tags[f['role']]
                        for i in range(len(f['idxes'])):
                            if i == 0:
                                tokens_labels[f['idxes'][i]] = f_tags[0]
                            else:
                                tokens_labels[f['idxes'][i]] = f_tags[1]
                if self.mode == 'train':
                    data_set.append([tokens_ids, tokens_labels])
                elif self.mode == 'test':
                    data_set.append(
                        [tokens_ids, tokens_labels, qids]
                    )
                else:
                    raise AttributeError(
                        'the input mode: {}, is wrong.'.format(self.mode)
                    )
        return data_set

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        本函数实现两个功能：\n
        将batch padding到一致的长度\n
        转化为torch的tensor类
        """
        batch_sentence = pad_sequence(
            [
                torch.from_numpy(np.array(s[0]))
                for s in batch
            ],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        batch_sentence = batch_sentence.long().to(config.device)
        if self.mode == 'dev':
            batch_qids = [s[-1] for s in batch]
            return [batch_sentence, batch_qids]
        else:
            batch_labels = pad_sequence(
                [
                    torch.from_numpy(np.array(s[1]))
                    for s in batch
                ],
                batch_first=True,
                padding_value=0
            )
            batch_labels = batch_labels.long().to(config.device)
            if self.mode == 'train':
                return [batch_sentence, batch_labels]
            elif self.mode == 'test':
                batch_qids = [s[-1] for s in batch]
                return [batch_sentence, batch_labels, batch_qids]


if __name__ == '__main__':
    from transformers import BertTokenizer
    import utils
    utils.data_split()
    tokenizer = BertTokenizer.from_pretrained(
        config.model_dir
    )
    space = SpaCEDataset(
        config.train4a, 'A', config.device,
        tokenizer, 'train'
    )
    made_batch = [space.dataset[i] for i in range(16)]
    print(space.collate_fn(made_batch))
