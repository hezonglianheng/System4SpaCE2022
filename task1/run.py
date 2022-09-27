from torch.utils.data import DataLoader
import config
from utils import split_data, get_logger
from dataloader_build import SpaCEDataSet
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import logging
import warnings
import train
import test
import torch

warnings.filterwarnings('ignore')


def run():
    split_data()
    get_logger(config.log_file)
    logging.info('We will use {} to train.'.format(config.device))

    train_dataset = SpaCEDataSet(config.train_data,
                                 config.device)
    test_dataset = SpaCEDataSet(config.test_data,
                                config.device,
                                mode='test')
    logging.info('Datasets are built.')

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             collate_fn=test_dataset.collate_fn)
    logging.info('Dataloaders are built.')

    model = BertForSequenceClassification.from_pretrained(config.model_file,
                                                          num_labels=2)
    model.to(config.device)
    # print(next(model.parameters()).device)
    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate,
                      correct_bias=False)
    train_step_each_epoch = len(train_dataset) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_step_each_epoch,
                                                num_training_steps=train_step_each_epoch * config.epoch_num)
    train.train(train_loader, test_loader, model, optimizer, scheduler)


def run_test():
    get_logger(config.log_file)
    test.test(config.dev_data, config.result_dir)
    logging.info('Test finished.')


if __name__ == '__main__':
    run()
    run_test()
