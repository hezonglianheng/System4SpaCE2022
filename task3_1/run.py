import logging
import warnings
import config
from utils import get_logger, split_data
from dataset import Dataset4Step1, Dataset4Step2, Dataset4Step3
import train41
import train42
import train43
import test
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertForTokenClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

warnings.filterwarnings('ignore')


def dev_step1(tokenizer):
    dev_dataset = Dataset4Step1(
        config.dev_file,
        config.device,
        tokenizer,
        'dev'
    )
    logging.info('dataset for step1 dev built.')
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        collate_fn=dev_dataset.collate_fn
    )
    logging.info('dataloader for step1 dev built.')
    test.test4step1(dev_dataloader)


def train_step1(tokenizer):
    train_dataset = Dataset4Step1(
        config.train_file,
        config.device,
        tokenizer,
        'train'
    )
    test_dataset = Dataset4Step1(
        config.test_file,
        config.device,
        tokenizer,
        'test'
    )
    logging.info(
        'datasets for step 1 train built.'
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    logging.info(
        'dataloaders for step 1 train built.'
    )
    model = BertForTokenClassification.from_pretrained(
        config.model_dir,
        num_labels=len(train_dataset.tags) + 1
    )
    model.to(config.device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        correct_bias=False
    )
    num_warm_up_steps = len(train_dataset) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warm_up_steps,
        num_training_steps=num_warm_up_steps * config.batch_size
    )
    logging.info(
        'train for step 1 starts.'
    )
    train41.main(train_dataloader, test_dataloader,
                 model, optimizer, scheduler)


def dev_step2(tokenizer):
    dev_dataset = Dataset4Step2(
        config.step1_result_file,
        config.device,
        tokenizer,
        'dev'
    )
    logging.info('dataset for step2 dev built!')
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn
    )
    logging.info('dataset for step2 dev built.')
    test.test4step2(dev_dataloader)


def train_step2(tokenizer):
    train_dataset = Dataset4Step2(
        config.train_file,
        config.device,
        tokenizer,
        'train'
    )
    test_dataset = Dataset4Step2(
        config.test_file,
        config.device,
        tokenizer,
        'test'
    )
    logging.info('datasets for step2 are built.')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    logging.info('dataloaders for step2 are built.')
    model = BertForTokenClassification.from_pretrained(
        config.model_dir,
        num_labels=26
    )
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        correct_bias=False
    )
    warm_up_steps = len(train_dataset) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps=warm_up_steps * config.batch_size
    )
    logging.info('train for step2 starts.')
    train42.main(
        train_loader, test_loader, model,
        optimizer, scheduler
    )


def dev_step3(tokenizer):
    dev_dataset = Dataset4Step3(
        config.step2_result_file, config.device,
        tokenizer, 'dev'
    )
    logging.info('dataset for step3 dev built.')

    dataloader = DataLoader(
        dev_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=dev_dataset.collate_fn
    )
    logging.info('dataloader for step3 dev built.')
    test.test4step3(dataloader)


def train_step3(tokenizer):
    train_dataset = Dataset4Step3(
        config.train_file, config.device,
        tokenizer, 'train'
    )
    test_dataset = Dataset4Step3(
        config.test_file, config.device,
        tokenizer, 'test'
    )
    logging.info('datasets for step3 train built.')

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=test_dataset.collate_fn
    )
    logging.info('dataloaders for step3 train built.')

    model = BertForSequenceClassification.from_pretrained(
        config.model_dir, num_labels=2
    )
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        correct_bias=False
    )
    num_warmup_steps = len(train_dataset) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_warmup_steps * config.batch_size
    )
    train43.main(
        train_loader, test_loader, model,
        optimizer, scheduler
    )


def run():
    get_logger(config.log_file)
    split_data()
    logging.info(
        'we will use {} to train.'.format(config.device)
    )
    tokenizer = BertTokenizer.from_pretrained(
        config.model_dir
    )
    train_step1(tokenizer)
    dev_step1(tokenizer)
    train_step2(tokenizer)
    dev_step2(tokenizer)
    train_step3(tokenizer)
    dev_step3(tokenizer)


if __name__ == '__main__':
    run()
