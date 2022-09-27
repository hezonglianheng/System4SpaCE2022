import config
from utils import split_data, get_logger
from dataset_build import SpaCEDataset
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import logging
import warnings
import train
import test
import torch

warnings.filterwarnings('ignore')


def run_dev():
    get_logger(config.log_file)
    test.test(config.dev_data, config.result_dir)


def run_train():
    split_data()
    get_logger(config.log_file)
    logging.info('We will use {} to train.'.
                 format(config.device))

    train_dataset = SpaCEDataset(
        config.train_data,
        config.device,
        'train'
    )
    test_dataset = SpaCEDataset(
        config.test_data,
        config.device,
        'test'
    )
    logging.info('Datasets are built!')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    logging.info('Dataloaders are built!')

    model = BertForTokenClassification.from_pretrained(
        config.model_file,
        num_labels=23
    )
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        correct_bias=False
    )
    train_step_each_epoch = len(train_dataset) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_step_each_epoch,
        num_training_steps=train_step_each_epoch*config.batch_size
    )
    logging.info('Training starts!')
    train.train(
        train_loader, test_loader, model,
        optimizer, scheduler
    )


if __name__ == '__main__':
    run_train()
    run_dev()
