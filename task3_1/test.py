import logging
import torch
from tqdm import tqdm
import numpy as np
import json
import config
from transformers import BertForTokenClassification, BertForSequenceClassification


def test4step1(dataloader):
    if config.step1_model_dir is None:
        logging.info('no model be built!')
        return None
    else:
        dev_model = BertForTokenClassification.from_pretrained(
            config.step1_model_dir
        )
        dev_model.to(config.device)
        logging.info(
            'model built from {}.'.format(config.step1_model_dir)
        )

    qids = []
    pred_labels = []
    dev_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_data, batch_qids = batch
            batch_mask = batch_data.gt(0)
            outputs = dev_model(batch_data, batch_mask)
            batch_outputs = outputs.logits
            batch_outputs = batch_outputs.detach().cpu().numpy()
            batch_pred_labels = np.argmax(batch_outputs, axis=-1)
            qids.extend(batch_qids)
            for i in range(len(batch_pred_labels)):
                pred_labels.append(batch_pred_labels[i])

    assert len(qids) == len(pred_labels)

    with open(config.dev_file, 'r+',
              encoding='utf8') as file:
        dev_data = json.load(file)

    result = []

    for i in range(len(qids)):
        for item in dev_data:
            if item['qid'] == qids[i]:
                dev_outputs_idx = []
                new_item = {
                    'qid': item['qid'],
                    'context': item['context'],
                    'outputs': []
                }
                for t in range(len(pred_labels[i])):
                    if pred_labels[i][t] == 1:
                        dev_outputs_idx.append([t])
                    elif pred_labels[i][t] == 2 \
                            and len(dev_outputs_idx) > 0:
                        dev_outputs_idx[-1].append(t)
                for idxes in dev_outputs_idx:
                    text = ''
                    for idx in idxes:
                        text += item['context'][idx]
                    new_item['outputs'].append(
                        [{
                            'text': text,
                            'idxes': idxes
                        }]
                    )
                result.append(new_item)

    with open(config.step1_result_file, 'w+',
              encoding='utf8') as file:
        json.dump(result, file, ensure_ascii=False)


def test4step2(dataloader):
    if config.step2_model_dir is None:
        logging.info('no model!')
        return
    else:
        dev_model = BertForTokenClassification.from_pretrained(
            config.step2_model_dir
        )
        dev_model.to(config.device)
        logging.info(
            'model built from {}'.format(config.step2_model_dir)
        )

    qids = []
    pred_labels = []
    dev_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_s, batch_qids = batch
            batch_mask = batch_s.gt(0)
            outputs = dev_model(batch_s, batch_mask)
            batch_outputs = outputs.logits
            batch_outputs = batch_outputs.detach().cpu().numpy()
            batch_pred_lables = np.argmax(
                batch_outputs, axis=-1
            )
            qids.extend(batch_qids)
            for i in range(len(batch_pred_lables)):
                pred_labels.append(batch_pred_lables[i])

    assert len(qids) == len(pred_labels)

    with open(config.step1_result_file, 'r+',
              encoding='utf8') as f:
        data = json.load(f)

    data = [
        {
            'qid': item['qid'],
            'context': item['context'],
            'outputs': []
        }
        for item in data
    ]

    for item in data:
        for i in range(len(qids)):
            if item['qid'] == qids[i]:
                curr_output = [None] * len(config.tags4step2)
                for ch_idx in range(len(item['context'])):
                    for out_idx in range(len(config.tags4step2)):
                        if pred_labels[i][ch_idx] in config.tags4step2[out_idx]:
                            if curr_output[out_idx] is None:
                                curr_output[out_idx] = {
                                    'text': item['context'][ch_idx],
                                    'idxes': [ch_idx]
                                }
                            else:
                                curr_output[out_idx]['text'] += item['context'][ch_idx]
                                curr_output[out_idx]['idxes'].append(ch_idx)
                            if out_idx == 1:
                                if pred_labels[i][ch_idx] == 3:
                                    curr_output[18] = '远'
                                elif pred_labels[i][ch_idx] == 4:
                                    curr_output[18] = '近'
                                elif pred_labels[i][ch_idx] == 5:
                                    curr_output[18] = '变远'
                                elif pred_labels[i][ch_idx] == 6:
                                    curr_output[18] = '变近'
                            elif out_idx == 2:
                                if pred_labels[i][ch_idx] == 8:
                                    curr_output[6] = '说话时'
                                elif pred_labels[i][ch_idx] == 9:
                                    curr_output[6] = '过去'
                                elif pred_labels[i][ch_idx] == 10:
                                    curr_output[6] = '将来'
                            elif out_idx == 5:
                                if pred_labels[i][ch_idx] == 13:
                                    curr_output[6] = '之时'
                                elif pred_labels[i][ch_idx] == 14:
                                    curr_output[6] = '之前'
                                elif pred_labels[i][ch_idx] == 15:
                                    curr_output[6] = '之后'
                item['outputs'].append(curr_output)

    with open(config.step2_result_file, 'w+',
              encoding='utf8') as file:
        json.dump(data, file, ensure_ascii=False)


def test4step3(dataloader):
    if config.step3_model_dir is None:
        logging.info('no model be built!')
        return None
    else:
        dev_model = BertForSequenceClassification.from_pretrained(
            config.step3_model_dir
        )
        dev_model.to(config.device)
        logging.info(
            'model loaded from {}'.format(
                config.step3_model_dir
            )
        )

    pred_tags = []
    dev_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_ids = batch[0]
            batch_mask = batch_ids.gt(0)
            outputs = dev_model(batch_ids, batch_mask)
            batch_outputs = outputs.logits
            # dimension: (batch_size, config.num_labels)
            batch_outputs = batch_outputs.detach().cpu().numpy()
            batch_pred_labels = np.argmax(
                batch_outputs, axis=-1
            )
            for label in batch_pred_labels:
                pred_tags.append(label)

    with open(config.step2_result_file, 'r+',
              encoding='utf8') as f:
        data = json.load(f)

    all_output = 0
    for item in data:
        all_output += len(item['outputs'])

    assert all_output == len(pred_tags)

    count = 0
    ans_data = []
    for item in data:
        for output in item['outputs']:
            if pred_tags[count] == 0:
                output[3] = '假'
            count += 1
        ans_data.append(item)

    with open(config.step3_result_file, 'w+',
              encoding='utf8') as file:
        json.dump(ans_data, file, ensure_ascii=False)