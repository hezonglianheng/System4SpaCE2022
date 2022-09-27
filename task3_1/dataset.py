from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import config
import torch
import numpy as np


class Dataset4Step1(Dataset):
    def __init__(self, file, device, tokenizer, mode):
        self.device = device
        self.tokenizer = tokenizer
        self.mode = mode
        self.tags = config.tags4step1
        self.dataset = self.get_dataset(file)

    def get_dataset(self, file):
        with open(file, 'r+', encoding='utf8') as f:
            data = json.load(f)

        dataset = []
        for item in data:
            token_ids = self.tokenizer.convert_tokens_to_ids(
                [char for char in item['context']]
            )
            if self.mode == 'dev':
                qid = item['qid']
                dataset.append([token_ids, qid])
            elif self.mode == 'train' or self.mode == 'test':
                s_tags = [0] * len(item['context'])
                for output in item['outputs']:
                    for i in range(len(output[0]['idxes'])):
                        if i == 0:
                            s_tags[output[0]['idxes'][i]] = self.tags[0]
                        else:
                            s_tags[output[0]['idxes'][i]] = self.tags[1]
                dataset.append([token_ids, s_tags])
            else:
                raise AttributeError(
                    'the mode: {}, is wrong.'.format(self.mode)
                )
        return dataset

    def __getitem__(self, item: int):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        sentence = [np.array(s[0]) for s in batch]
        batch_data = pad_sequence(
            [torch.from_numpy(s) for s in sentence],
            batch_first=True,
            padding_value=0
        )
        batch_data = batch_data.long().to(self.device)
        if self.mode == 'train' or self.mode == 'test':
            tags = [np.array(s[1]) for s in batch]
            batch_tags = pad_sequence(
                [torch.from_numpy(t) for t in tags],
                batch_first=True,
                padding_value=0
            )
            batch_tags = batch_tags.long().to(self.device)
            return [batch_data, batch_tags]
        elif self.mode == 'dev':
            qids = [s[1] for s in batch]
            return [batch_data, qids]
        else:
            raise AttributeError(
                'the mode: {}, is wrong!'.format(self.mode)
            )


class Dataset4Step2(Dataset):
    def __init__(self, file, device, tokenizer, mode):
        self.device = device
        self.tokenizer = tokenizer
        self.mode = mode
        self.tags = config.tags4step2
        self.dataset = self.get_dataset(file)

    def get_dataset(self, file):
        dataset = []
        with open(file, 'r+',
                  encoding='utf8') as f:
            data = json.load(f)

        for item in data:
            token_ids = self.tokenizer.convert_tokens_to_ids(
                [char for char in item['context']]
            )
            for output in item['outputs']:
                curr_s_id = self.tokenizer.convert_tokens_to_ids(
                    [char for char in output[0]['text']]
                )
                # 保障S元素的完整性
                if len(token_ids) + len(curr_s_id) > 511:
                    item_id = token_ids[
                              :511-len(curr_s_id)
                              ] + [
                        self.tokenizer.sep_token_id
                    ] + curr_s_id
                else:
                    item_id = token_ids + [
                        self.tokenizer.sep_token_id
                    ] + curr_s_id
                if self.mode == 'dev':
                    dataset.append([item_id, item['qid']])
                elif self.mode == 'train' or self.mode == 'test':
                    item_labels = [0] * len(item_id)
                    # 标注
                    for i in range(len(self.tags)):
                        if output[i] is not None:
                            if len(self.tags[i]) == 1:  # 单纯标注
                                for idx in output[i]['idxes']:
                                    item_labels[idx] = self.tags[i][0]
                            elif len(self.tags[i]) == 0:  # 标签信息不标注
                                continue
                            else:  # 揉入标签信息的标注
                                if i == 1:
                                    for idx in output[i]['idxes']:
                                        if output[17] is None:
                                            item_labels[idx] = self.tags[i][0]
                                        elif output[17] == '远':
                                            item_labels[idx] = self.tags[i][1]
                                        elif output[17] == '近':
                                            item_labels[idx] = self.tags[i][2]
                                        elif output[17] == '变远':
                                            item_labels[idx] = self.tags[i][3]
                                        elif output[17] == '变近':
                                            item_labels[idx] = self.tags[i][4]
                                elif i == 2:
                                    for idx in output[i]['idxes']:
                                        if output[6] is None:
                                            item_labels[idx] = self.tags[i][0]
                                        elif output[6] == '说话时':
                                            item_labels[idx] = self.tags[i][1]
                                        elif output[6] == '过去':
                                            item_labels[idx] = self.tags[i][2]
                                        elif output[6] == '将来':
                                            item_labels[idx] = self.tags[i][3]
                                elif i == 5:
                                    for idx in output[i]['idxes']:
                                        if output[6] is None:
                                            item_labels[idx] = self.tags[i][0]
                                        elif output[6] == '之时':
                                            item_labels[idx] = self.tags[i][1]
                                        elif output[6] == '之前':
                                            item_labels[idx] = self.tags[i][2]
                                        elif output[6] == '之后':
                                            item_labels[idx] = self.tags[i][3]
                                else:
                                    raise AttributeError('tags error!')
                    if self.mode == 'train':
                        dataset.append([item_id, item_labels])
                    else:
                        dataset.append([item_id, item_labels, item['qid']])
        return dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        batch_s = [np.array(s[0]) for s in batch]
        batch_s = [torch.from_numpy(s) for s in batch_s]
        batch_s = pad_sequence(
            batch_s,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        batch_s = batch_s.long().to(self.device)
        if self.mode == 'dev':
            qids = [s[1] for s in batch]
            return [batch_s, qids]
        else:
            batch_labels = [np.array(s[1]) for s in batch]
            batch_labels = [torch.from_numpy(s) for s in batch_labels]
            batch_labels = pad_sequence(
                batch_labels,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            batch_labels = batch_labels.long().to(self.device)
            if self.mode == 'train':
                return [batch_s, batch_labels]
            elif self.mode == 'test':
                qids = [s[-1] for s in batch]
                return [batch_s, batch_labels, qids]


class Dataset4Step3(Dataset):
    def __init__(self, file, device, tokenizer, mode):
        self.device = device
        self.tokenizer = tokenizer
        self.mode = mode
        self.dataset = self.get_dataset(file)

    def get_dataset(self, file):
        with open(file, 'r+', encoding='utf8') as f:
            data = json.load(f)

        dataset = []
        for item in data:
            token_ids = self.tokenizer.convert_tokens_to_ids(
                [char for char in item['context']]
            )
            for output in item['outputs']:
                output_token = []
                for i in range(len(output)):
                    if i == 3 or i == 6 or i == 17:
                        continue
                    else:
                        if output[i] is not None:
                            char_list = [c for c in output[i]['text']]
                            output_token.extend(char_list)
                output_token_ids = self.tokenizer.convert_tokens_to_ids(
                    output_token
                )
                if len(token_ids) + len(output_token_ids) > 511:
                    curr_ids = token_ids[:511 - len(output_token_ids)] + \
                               [self.tokenizer.sep_token_id] + \
                               output_token_ids
                else:
                    curr_ids = token_ids + \
                               [self.tokenizer.sep_token_id] + \
                               output_token_ids
                if output[3] == '假':
                    truth = 0
                else:
                    truth = 1
                if self.mode == 'train':
                    dataset.append([curr_ids, truth])
                elif self.mode == 'test':
                    qid = item['qid']
                    dataset.append([curr_ids, truth, qid])
                elif self.mode == 'dev':
                    dataset.append([curr_ids])

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def collate_fn(self, batch):
        batch_ids = [np.array(s[0]) for s in batch]
        batch_ids = pad_sequence(
            [torch.from_numpy(s) for s in batch_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        batch_ids = batch_ids.long().to(self.device)
        if self.mode == 'train':
            batch_truths = [s[1] for s in batch]
            batch_truths = torch.tensor(batch_truths)
            batch_truths = batch_truths.to(self.device)
            return [batch_ids, batch_truths]
        elif self.mode == 'test':
            batch_truths = [s[1] for s in batch]
            batch_truths = torch.tensor(batch_truths)
            batch_truths = batch_truths.to(self.device)
            return [batch_ids, batch_truths]
        elif self.mode == 'dev':
            return [batch_ids]
        else:
            raise AttributeError(
                'the type: {}, is wrong.'.format(self.mode)
            )


if __name__ == '__main__':
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        config.model_dir
    )
    space = Dataset4Step3(
        config.step2_result_file, config.device,
        tokenizer, 'dev'
    )
    print(len(space))
    print(space.collate_fn(space.dataset))

    with open(config.step2_result_file, 'r+',
              encoding='utf8') as file:
        data = json.load(file)
    num = 0
    for item in data:
        num += len(item['outputs'])
    print(num)
