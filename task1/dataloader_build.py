import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from transformers import BertTokenizer
import json
import numpy as np


class SpaCEDataSet(Dataset):
    def __init__(self, file_path: str, device,
                 mode: str = 'train'):
        self.tokenizer = BertTokenizer.from_pretrained(config.model_file,
                                                       do_lower_case=True)
        self.mode = mode  # 3 modes: train, test, dev
        self.dataset = self._get_dataset(file_path)
        self.device = device

    def _get_dataset(self, file_path):

        with open(file_path, mode='r+', encoding='utf8') \
                as file:
            raw_data = json.load(file)

        data_set = []
        for item in raw_data:
            tokens = self.tokenizer.encode(item['context'])
            if self.mode == 'train' or self.mode == 'test':
                data_set.append([tokens, item['judge']])
            elif self.mode == 'dev':
                data_set.append([tokens])
            else:
                raise ValueError('Your mode is wrong. We only have 3 modes: "train", "test" and "dev".')

        return data_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        context = self.dataset[index][0]
        if self.mode == 'train' or self.mode == 'test':
            judge = self.dataset[index][1]
            return [context, judge]
        elif self.mode == 'dev':
            return [context]

    def collate_fn(self, batch):
        """
        这个函数实现两个功能：
        1.padding：将batch中的句子padding成最长的长度
        2.转化为pytorch的tensor
        """
        sentences = [np.array(s[0]) for s in batch]
        batch_data = pad_sequence([torch.from_numpy(s) for s in sentences],
                                  batch_first=True,
                                  padding_value=self.tokenizer.pad_token_id)
        batch_data = batch_data.long()
        batch_data = batch_data.to(config.device)
        if self.mode == 'train' or self.mode == 'test':
            labels = [s[1] for s in batch]
            batch_label = torch.tensor(labels)
            batch_label = batch_label.to(config.device)
            return [batch_data, batch_label]
        elif self.mode == 'dev':
            return [batch_data]
        else:
            raise ValueError('Your mode is wrong.We only have 3 mode:"train", "test" and "dev".')


if __name__ == '__main__':
    dataset = SpaCEDataSet(config.test_data, config.device,
                           mode='test')
    print(dataset.dataset)
