from torch.utils.data import DataLoader
import config
import json
import train
import test
import metrics
from dataset import SpaCEDataset
import logging
import warnings
from utils import get_logger, data_split
from transformers import BertTokenizer, BertForTokenClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")


def run_train(_type, tokenizer):
    if _type == 'A':
        train_dataset = SpaCEDataset(
            config.train4a, _type, config.device,
            tokenizer, 'train'
        )
        test_dataset = SpaCEDataset(
            config.test4a, _type, config.device,
            tokenizer, 'test'
        )
        num_type = 5
    elif _type == 'B':
        train_dataset = SpaCEDataset(
            config.train4b, _type, config.device,
            tokenizer, 'train'
        )
        test_dataset = SpaCEDataset(
            config.test4b, _type, config.device,
            tokenizer, 'test'
        )
        num_type = 13
    elif _type == 'C':
        train_dataset = SpaCEDataset(
            config.train4c, _type, config.device,
            tokenizer, 'train'
        )
        test_dataset = SpaCEDataset(
            config.test4c, _type, config.device,
            tokenizer, 'test'
        )
        num_type = 7
    else:
        raise ValueError(
            'the input type: {}, is wrong.'.format(_type)
        )
    logging.info('datasets are built.')

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
    logging.info('dataloaders are built.')

    model = BertForTokenClassification.from_pretrained(
        config.model_dir,
        num_labels=num_type
    )
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        correct_bias=False
    )
    num_warmup_steps = len(train_dataset) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_warmup_steps * config.batch_size
    )
    logging.info(
        'training for type {} starts.'.format(_type)
    )
    train.main(
        train_loader, test_loader, model,
        optimizer, scheduler, _type
    )


def run_dev(tokenizer):
    logging.info('test starts.')
    res4a = test.main(
        config.dev_file, config.model4a_dir,
        tokenizer, 'A'
    )
    logging.info('get result for type A.')
    res4b = test.main(
        config.dev_file, config.model4b_dir,
        tokenizer, 'B'
    )
    logging.info('get result for type B.')
    res4c = test.main(
        config.dev_file, config.model4c_dir,
        tokenizer, 'C'
    )
    logging.info('get result for type C.')

    with open(config.dev_file, 'r+',
              encoding='utf8') as file:
        dev_data = json.load(file)

    dev_ans = []

    for item in dev_data:
        pred_ans = []
        if res4a is not None:
            for i in range(len(res4a[0])):
                if res4a[0][i] == item['qid']:
                    pred_ans += metrics.reasons_translate(
                        res4a[1][i], 'A'
                    )

        if res4b is not None:
            for i in range(len(res4b[0])):
                if res4b[0][i] == item['qid']:
                    pred_ans += metrics.reasons_translate(
                        res4b[1][i], 'B'
                    )

        if res4c is not None:
            for i in range(len(res4c[0])):
                if res4c[0][i] == item['qid']:
                    pred_ans += metrics.reasons_translate(
                        res4c[1][i], 'C'
                    )

        for ans in pred_ans:
            for f in ans['fragments']:
                text = str.join([
                    item['context'][i]
                    for i in f['idxes']
                ])
                f['text'] = text

        dev_ans.append(
            {
                'qid': item['qid'],
                'context': item['context'],
                'reasons': pred_ans
            }
        )

    with open(config.dev_result_file, 'w+',
              encoding='utf8') as file:
        json.dump(dev_ans, file, ensure_ascii=False)


def run():
    get_logger(config.log_file)
    data_split()
    logging.info(
        'we will use {} to train.'.format(config.device)
    )
    tokenizer = BertTokenizer.from_pretrained(
        config.model_dir
    )
    run_train('A', tokenizer)
    run_train('B', tokenizer)
    run_train('C', tokenizer)
    run_dev(tokenizer)


if __name__ == '__main__':
    run()
