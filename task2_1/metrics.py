import config
import json


def get_common_part(seq1, seq2):
    seq1.sort()
    seq2.sort()
    seq1_id = 0
    seq2_id = 0
    common_part = 0
    while seq1_id < len(seq1) and seq2_id < len(seq2):
        if seq1[seq1_id] == seq2[seq2_id]:
            common_part += 1
            seq1_id += 1
            seq2_id += 1
        elif seq1[seq1_id] > seq2[seq2_id]:
            seq2_id += 1
        else:
            seq1_id += 1

    return common_part


def reasons_translate(pred_labels, _type):
    if _type == 'A':
        tags = config.tags4a
    elif _type == 'B':
        tags = config.tags4b
    elif _type == 'C':
        tags = config.tags4c
    else:
        raise AttributeError(
            'the type: {}, is wrong.'.format(_type)
        )

    fragments = {}
    for key in tags:
        fragments[key] = []
    for i in range(len(pred_labels)):
        if pred_labels[i] > 0:
            for key in tags:
                if pred_labels[i] == tags[key][0]:
                    fragments[key].append([i])
                elif pred_labels[i] == tags[key][1]:
                    #fragments[key][-1].append(i)
                    if len(fragments[key]) > 0:
                        fragments[key][-1].append(i)

    answers = []
    num_answer = max(
        [len(fragments[key]) for key in fragments]
    )
    for i in range(num_answer):
        curr_ans = {
            'type': _type,
            'fragments': []
        }
        for key in fragments:
            if i < len(fragments[key]):
                curr_ans['fragments'].append(
                    {
                        'role': key,
                        'idxes': fragments[key][i]
                    }
                )
        answers.append(curr_ans)
    return answers


def get_precision_recall_f1(true_labels, pred_labels):
    common_part = get_common_part(true_labels, pred_labels)
    if common_part == 0:
        return [0.0] * 3
    else:
        precision = common_part / len(true_labels)
        recall = common_part / len(pred_labels)
        f1 = 2 * precision * recall / (precision + recall)
        return [precision, recall, f1]


def quotas_calculate(true_reasons, pred_reasons):
    quota = [0.0] * 3
    for pr in pred_reasons:
        pr_quota = [0.0] * 3
        for tr in true_reasons:
            # 更新：指标1，2的更新方式
            # 计算指标1
            if pr['type'] == tr['type']:
                quota1 = 1.0
            else:
                quota1 = 0.0
            # 计算指标2
            pr_tokens = []
            tr_tokens = []
            for f in pr['fragments']:
                pr_tokens.extend(f['idxes'])
            for f in tr['fragments']:
                tr_tokens.extend(f['idxes'])
            precision, recall, f1 = get_precision_recall_f1(
                tr_tokens, pr_tokens
            )
            quota2 = f1
            # 更新指标1，2
            if quota2 > pr_quota[1]:
                pr_quota[0] = quota1
                pr_quota[1] = quota2
            # 计算指标3
            if quota1 == 0:
                quota3 = 0
            else:
                quota3 = 0

                for tf in tr['fragments']:
                    for pf in pr['fragments']:
                        if pf['role'] == tf['role']:
                            precision, recall, f1 = get_precision_recall_f1(
                                tf['idxes'], pf['idxes']
                            )
                            quota3 += f1
                quota3 /= len(tr['fragments'])
            # 更新指标3
            if quota3 > pr_quota[2]:
                pr_quota[2] = quota3
        # quota更新
        if pr_quota[1] > quota[1]:
            quota[0] = pr_quota[0]
            quota[1] = pr_quota[1]
        if pr_quota[2] > quota[2]:
            quota[2] = pr_quota[2]
    return quota


def main(pred_tags, qids, _type):
    if _type == 'A':
        with open(config.test4a, 'r+',
                  encoding='utf8') as file:
            test_data = json.load(file)
    elif _type == 'B':
        with open(config.test4b, 'r+',
                  encoding='utf8') as file:
            test_data = json.load(file)
    elif _type == 'C':
        with open(config.test4c, 'r+',
                  encoding='utf8') as file:
            test_data = json.load(file)
    else:
        raise AttributeError(
            'the type: {}, is wrong.'.format(_type)
        )

    final_quota = [0.0] * 3
    for i in range(len(pred_tags)):
        pred_reasons = reasons_translate(
            pred_tags[i], _type
        )
        true_reasons = None
        for j in range(len(test_data)):
            if qids[i] == test_data[j]['qid']:
                true_reasons = test_data[j]['reasons']

        if true_reasons is not None:
            if len(pred_reasons) == 0:
                curr_quota = [0.0] * 3
            else:
                curr_quota = quotas_calculate(
                    true_reasons, pred_reasons
                )
            final_quota = [
                final_quota[n] + curr_quota[n]
                for n in range(len(final_quota))
            ]

    final_quota = [
        final_quota[n] / len(pred_tags)
        for n in range(len(final_quota))
    ]
    return final_quota


if __name__ == '__main__':
    from dataset import SpaCEDataset
    from transformers import BertTokenizer
    import random
    space = SpaCEDataset(
        config.train4b, 'B', config.device,
        tokenizer=BertTokenizer.from_pretrained(
            config.model_dir
        ),
        mode='train'
    )

    with open(config.train4b, 'r+',
              encoding='utf8') as file:
        data = json.load(file)

    num = random.randint(0, 30)

    print(data[num]['context'])
    print(data[num]['reasons'])
    print(reasons_translate(space.dataset[num][1], 'B'))
    print(quotas_calculate(
        data[num]['reasons'],
        reasons_translate(space.dataset[num][1], 'B')
    ))
