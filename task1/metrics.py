import json
import config


def get_accuracy(true_tags, predict_tags,
                 mode='dev'):
    correct_items = 0
    for i in range(len(predict_tags)):
        if true_tags[i] == predict_tags[i]:
            correct_items += 1
    accuracy = correct_items / len(true_tags)
    if mode == 'test':
        with open(config.acc_file, mode='w+') as result:
            json.dump('accuracy: {}'.format(accuracy),
                      fp=result,
                      ensure_ascii=False)
    return accuracy


if __name__ == '__main__':
    a_true = [0, 0, 1, 1, 1]
    a_predict = [1, 0, 1, 1, 0]
    print(get_accuracy(a_true, a_predict, mode='test'))