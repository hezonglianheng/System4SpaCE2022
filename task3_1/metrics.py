def get_common_part(seq1: list, seq2: list):
    common_len = 0
    for i in range(min(len(seq1), len(seq2))):
        if seq1[i] > 0 and seq1[i] == seq2[i]:
            common_len += 1
    return common_len


def get_tags(seq: list):
    tags_len = 0
    for i in range(len(seq)):
        if seq[i] > 0:
            tags_len += 1
    return tags_len


def main(true_tags, pred_tags):
    total_f1 = 0
    for i in range(len(true_tags)):
        curr_common = get_common_part(true_tags[i], pred_tags[i])
        if get_tags(true_tags[i]) > 0:
            precision = curr_common / get_tags(true_tags[i])
        else:
            precision = 0
        if get_tags(pred_tags[i]) > 0:
            recall = curr_common / get_tags(pred_tags[i])
        else:
            recall = 0
        if precision + recall != 0:
            total_f1 += 2 * precision * recall / (precision + recall)
        else:
            total_f1 += 0
    total_f1 = total_f1 / len(true_tags)
    return total_f1
