from tqdm import tqdm
import torch
import config
from dataset_build import SpaCEDataset
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
import logging
import numpy as np
import json
import metrics


def put_answer(dev_file, pred_tags, qids):
    with open(dev_file, 'r+',
              encoding='utf8') as dev:
        dev_data = json.load(dev)

    dev_answer = []
    for i in range(len(qids)):
        pred_reasons = metrics.reasons_translate(
            pred_tags[i]
        )
        for reason in pred_reasons:
            for role in reason['fragments']:
                role['text'] = ''
        for item in dev_data:
            if item['qid'] == qids[i]:
                for reason in pred_reasons:
                    for role in reason['fragments']:
                        for idx in role['idxes']:
                            if idx < len(item['context']):
                                role['text'] += item['context'][idx]
                dev_answer.append(
                    {
                        'qid': item['qid'],
                        'context': item['context'],
                        'reasons': pred_reasons
                    }
                )

    with open(config.dev_result_file, 'w+',
              encoding='utf8') as result:
        json.dump(dev_answer, result, ensure_ascii=False)


def test(dev_file, model_dir):
    dev_dataset = SpaCEDataset(dev_file, config.device,
                               'dev')
    logging.info('dataset is built.')

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn
    )
    logging.info('dataloader is built.')

    if model_dir is not None:
        dev_model = BertForTokenClassification.from_pretrained(
            model_dir
        )
        dev_model.to(config.device)
        logging.info('model loaded from {}'.format(model_dir))
    else:
        logging.info('no model to trained')
        return None

    dev_model.eval()
    pred_tags = []
    qids = []
    with torch.no_grad():
        for sample in tqdm(dev_dataloader):
            sample_data, sample_qids = sample
            sample_mask = sample_data.gt(0)
            output = dev_model(sample_data, sample_mask)
            sample_output = output.logits
            sample_output = sample_output.detach().cpu().numpy()
            sample_pred_tags = np.argmax(sample_output, axis=-1)
            qids.extend(sample_qids)
            for i in range(len(sample_pred_tags)):
                pred_tags.append(sample_pred_tags[i])

    assert len(pred_tags) == len(qids)

    put_answer(dev_file, pred_tags, qids)

