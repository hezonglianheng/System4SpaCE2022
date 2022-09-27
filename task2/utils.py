from sklearn.model_selection import train_test_split
import logging
import json
import config


def get_logger(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        s_handler = logging.StreamHandler()
        s_formatter = logging.Formatter('{asctime}:{levelname}:{message}', style='{')
        s_handler.setFormatter(s_formatter)
        logger.addHandler(s_handler)

        f_handler = logging.FileHandler(file_name, mode='a', encoding='utf8')
        f_formatter = logging.Formatter('{asctime}:{levelname}:{message}', style='{')
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)


def split_data():
    with open(config.origin_train_data, mode='r+',
              encoding='utf8') as origin:
        origin_data = json.load(origin)

    train_data, test_data = train_test_split(
        origin_data, test_size=config.test_size
    )

    with open(config.train_data, 'w+',
              encoding='utf8') as file1:
        json.dump(train_data, file1, ensure_ascii=False)

    with open(config.test_data, 'w+',
              encoding='utf8') as file2:
        json.dump(test_data, file2, ensure_ascii=False)


if __name__ == '__main__':
    split_data()
