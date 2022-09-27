import config
import json


def get_common_part(seq1: list[int], seq2: list[int]):
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


def get_precision_recall_f1(true_seq: list[int],
                            pred_seq: list[int]):
    common_len = get_common_part(true_seq, pred_seq)
    if common_len == 0:
        precision = recall = f1 = 0
    else:
        precision = common_len / len(true_seq)
        recall = common_len / len(pred_seq)
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def reasons_translate(pred_tags):
    reasons = [
        {
            'type': 'A',
            'fragments': []
        },
        {
            'type': 'B',
            'fragments': []
        },
        {
            'type': 'C',
            'fragments': []
        }
    ]
    for tag_id in range(len(pred_tags)):
        if pred_tags[tag_id] > 0:
            pred_type = None
            pred_role = None
            is_start = None
            for rule in config.tag_dic_list:
                if rule['tag'] == pred_tags[tag_id]:
                    pred_type = rule['type']
                    pred_role = rule['role']
                    is_start = rule['tag'] % 2
            for item in reasons:
                if item['type'] == pred_type:
                    have = 0
                    for f in item['fragments']:
                        if f['role'] == pred_role:
                            have = 1
                            if is_start:
                                f['able'] = 0
                            elif f['able'] == 1:
                                f['idxes'].append(tag_id)
                            break
                    if have == 0 and is_start == 1:
                        item['fragments'].append(
                            {
                                'role': pred_role,
                                'idxes': [tag_id],
                                'able': 1
                            }
                        )
    reasons = [
        r for r in reasons if r['fragments']
    ]
    return reasons


def get_quota(pred_reasons, true_reasons):
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
                true_seq=tr_tokens, pred_seq=pr_tokens
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


def main(pred_tags, qids):
    with open(config.test_data, 'r+',
              encoding='utf8') as testfile:
        testdata = json.load(testfile)

    final_quota = [0.0, 0.0, 0.0]

    for i in range(len(pred_tags)):
        pred_reasons = reasons_translate(pred_tags[i])

        true_reasons = None
        for j in range(len(testdata)):
            if qids[i] == testdata[j]['qid']:
                true_reasons = testdata[j]['reasons']
                break
        # print(true_reasons, end='\n')
        # print(pred_reasons, end='\n')
        if true_reasons is None:
            raise ValueError(
                'the qid: {}, can not be found in test data'.format(qids[i])
            )
        else:
            if len(pred_reasons) == 0:
                curr_quota = [0.0] * 3
            else:
                curr_quota = get_quota(
                    pred_reasons, true_reasons
                )
            final_quota = [
                final_quota[i] + curr_quota[i]
                for i in range(len(final_quota))
            ]

    final_quota = [
        final_quota[i] / len(pred_tags)
        for i in range(len(final_quota))
    ]
    return final_quota


if __name__ == '__main__':
    import random
    from dataset_build import SpaCEDataset

    space = SpaCEDataset(config.train_data,
                         config.device, 'train')
    num = random.randint(0, 30)
    print(num)
    with open(config.train_data, 'r+',
              encoding='utf8') as file:
        train = json.load(file)

    print(train[num]['context'])
    print(train[num]['reasons'])
    print(space.dataset[num][0])
    print(space.dataset[num][1])
    print(reasons_translate(space.dataset[num][1]))
