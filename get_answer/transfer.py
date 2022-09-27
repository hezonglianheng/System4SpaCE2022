import jsonlines
import json
import config


def transfer(origin_file, target_file):
    with open(origin_file, 'r+',
              encoding='utf8') as f:
        data = json.load(f)

    with jsonlines.open(target_file, 'w') as writer:
        for item in data:
            writer.write(item)


if __name__ == '__main__':
    transfer(config.task3_result_json,
             config.task3_result_jsonl)