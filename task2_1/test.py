import config
from dataset import SpaCEDataset
import logging
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
import torch
from tqdm import tqdm
import numpy as np


def main(dev_file, model_dir, tokenizer, _type):
    dev_dataset = SpaCEDataset(
        dev_file, _type, config.device,
        tokenizer, 'dev'
    )
    logging.info(
        'dataset is built.'
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn
    )
    logging.info(
        'dataloader is built.'
    )

    if model_dir is not None:
        dev_model = BertForTokenClassification.from_pretrained(
            model_dir
        )
        dev_model.to(config.device)
        logging.info('model loaded from {}.'.format(model_dir))
    else:
        logging.info('no model!')
        return None

    dev_model.eval()
    pred_labels = []
    qids = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            batch_sentence, batch_qids = batch
            batch_mask = batch_sentence.gt(0)
            outputs = dev_model(
                batch_sentence, batch_mask,
            )
            batch_outputs = outputs.logits
            batch_outputs = batch_outputs.detach().cpu().numpy()
            batch_pred_labels = np.argmax(batch_outputs, axis=-1)
            qids.extend(batch_qids)
            for i in range(len(batch_pred_labels)):
                pred_labels.append(batch_pred_labels[i])

    assert len(qids) == len(pred_labels)
    return qids, pred_labels
