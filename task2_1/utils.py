import logging
from sklearn.model_selection import train_test_split
import config
import json


def get_logger(file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        s_handler = logging.StreamHandler()
        s_formatter = logging.Formatter(
            '{asctime}: {levelname}: {message}',
            style='{'
        )
        s_handler.setFormatter(s_formatter)

        logger.addHandler(s_handler)
        f_handler = logging.FileHandler(file)
        f_formatter = logging.Formatter(
            '{asctime}: {levelname}: {message}',
            style='{'
        )
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)


def data_split():
    with open(config.origin_train_file, 'r+',
              encoding='utf8') as origin:
        origin_data = json.load(origin)

    origin4a = []
    origin4b = []
    origin4c = []
    for item in origin_data:
        item4a = []
        item4b = []
        item4c = []
        for r in item['reasons']:
            if r['type'] == 'A':
                item4a.append(
                    {
                        'qid': item['qid'],
                        'context': item['context'],
                        'reasons': [r]
                    }
                )
            elif r['type'] == 'B':
                item4b.append(
                    {
                        'qid': item['qid'],
                        'context': item['context'],
                        'reasons': [r]
                    }
                )
            elif r['type'] == 'C':
                item4c.append(
                    {
                        'qid': item['qid'],
                        'context': item['context'],
                        'reasons': [r]
                    }
                )
        origin4a.extend(item4a)
        origin4b.extend(item4b)
        origin4c.extend(item4c)

    train4a, test4a = train_test_split(
        origin4a,
        test_size=config.test_size,
        shuffle=True
    )
    train4b, test4b = train_test_split(
        origin4b,
        test_size=config.test_size,
        shuffle=True
    )
    train4c, test4c = train_test_split(
        origin4c,
        test_size=config.test_size,
        shuffle=True
    )

    with open(config.train4a, 'w+',
              encoding='utf8') as file:
        json.dump(train4a, file, ensure_ascii=False)

    with open(config.test4a, 'w+',
              encoding='utf8') as file:
        json.dump(test4a, file, ensure_ascii=False)

    with open(config.train4b, 'w+',
              encoding='utf8') as file:
        json.dump(train4b, file, ensure_ascii=False)

    with open(config.test4b, 'w+',
              encoding='utf8') as file:
        json.dump(test4b, file, ensure_ascii=False)

    with open(config.train4c, 'w+',
              encoding='utf8') as file:
        json.dump(train4c, file, ensure_ascii=False)

    with open(config.test4c, 'w+',
              encoding='utf8') as file:
        json.dump(test4c, file, ensure_ascii=False)


if __name__ == '__main__':
    data_split()
