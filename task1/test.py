import torch
from tqdm import tqdm
from dataloader_build import SpaCEDataSet
from torch.utils.data import DataLoader
import config
from transformers import BertForSequenceClassification
import logging
import numpy as np
import json


def test(dev_file, model_dir):
    dev_dataset = SpaCEDataSet(dev_file,
                               config.device,
                               mode='dev')
    logging.info('Dataset is built.')

    dev_loader = DataLoader(dev_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=dev_dataset.collate_fn)
    logging.info('Dataloader is built.')

    if model_dir is not None:
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(config.device)
        logging.info('Model loaded from {}.'.format(model_dir))
    else:
        logging.info('No model to test.')
        return None

    model.eval()
    predict_tags = []
    with torch.no_grad():
        for sample in tqdm(dev_loader):
            sample_data = sample[0]
            sample_musk = sample_data.gt(0)
            outputs = model(sample_data, sample_musk)
            sample_output = outputs.logits  # shape:(batch_size, num_labels)
            sample_output = sample_output.detach().cpu().numpy()
            predict_tags.extend(np.argmax(sample_output, axis=-1))

    with open(config.dev_data, mode='r',
              encoding='utf8') as dev_file:
        dev_items = json.load(dev_file)

    for index, item in enumerate(dev_items):
        item['judge'] = int(predict_tags[index])

    with open(config.result_file, mode='w',
              encoding='utf8') as result_file:
        json.dump(dev_items, result_file,
                  ensure_ascii=False)
