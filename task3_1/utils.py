import logging
from sklearn.model_selection import train_test_split
import json
import config


def get_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    s_handler = logging.StreamHandler()
    s_formatter = logging.Formatter(
        '{asctime}: {levelname}: {message}',
        style='{'
    )
    s_handler.setFormatter(s_formatter)

    f_handler = logging.FileHandler(log_file)
    f_formatter = logging.Formatter(
        '{asctime}: {levelname}: {message}',
        style='{'
    )
    f_handler.setFormatter(f_formatter)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)


def split_data():
    with open(config.origin_train_file, 'r+',
              encoding='utf8') as origin:
        origin_data = json.load(origin)

    train_data, test_data = train_test_split(
        origin_data,
        test_size=config.test_size
    )

    with open(config.train_file, 'w+',
              encoding='utf8') as file:
        json.dump(train_data, file, ensure_ascii=False)

    with open(config.test_file, 'w+',
              encoding='utf8') as file:
        json.dump(test_data, file, ensure_ascii=False)


if __name__ == '__main__':
    split_data()
