import torch

# 文件位置
origin_train_data = r'../data/task2/task2_train.json_dir'
dev_data = r'../data/task2/task2_dev_input.json_dir'
train_data = r'../data/task2/task2_train_input.json_dir'
test_data = r'../data/task2/task2_test_input.json_dir'

model_file = r'../chinese_wwm_ext_pytorch'

log_file = r'./experiment/log.txt'  # 日志文件
result_dir = r'./experiment'  # 储存结果文件夹
acc_file = r'./experiment/acc.json_dir'
dev_result_file = r'./experiment/dev_result.json'

#  超参数
batch_size = 16
test_size = 0.1
learning_rate = 1e-5
weight_decay = 0.01
patience = 0.0002
patience_num = 5
epoch_num = 30
min_epoch_num = 5
clip_grad = 5

# 本实验没有解决多标签的情况，多标签情况仍然有待解决
tag_dic_list = [
    {'type': 'O', 'role': 'other', 'tag': 0},
    {'type': 'A', 'role': 'text1', 'tag': 1},
{'type': 'A', 'role': 'text1', 'tag': 2},
    {'type': 'A', 'role': 'text2', 'tag': 3},
{'type': 'A', 'role': 'text2', 'tag': 4},
    {'type': 'B', 'role': 'S1', 'tag': 5},
{'type': 'B', 'role': 'S1', 'tag': 6},
    {'type': 'B', 'role': 'S2', 'tag': 7},
{'type': 'B', 'role': 'S2', 'tag': 8},
    {'type': 'B', 'role': 'P1', 'tag': 9},
{'type': 'B', 'role': 'P1', 'tag': 10},
    {'type': 'B', 'role': 'P2', 'tag': 11},
{'type': 'B', 'role': 'P2', 'tag': 12},
    {'type': 'B', 'role': 'E1', 'tag': 13},
{'type': 'B', 'role': 'E1', 'tag': 14},
    {'type': 'B', 'role': 'E2', 'tag': 15},
{'type': 'B', 'role': 'E2', 'tag': 16},
    {'type': 'C', 'role': 'S', 'tag': 17},
{'type': 'C', 'role': 'S', 'tag': 18},
    {'type': 'C', 'role': 'P', 'tag': 19},
{'type': 'C', 'role': 'P', 'tag': 20},
    {'type': 'C', 'role': 'E', 'tag': 21},
{'type': 'C', 'role': 'E', 'tag': 22}
]

# 使用设备设置
device = torch.device('cuda:1' if torch.cuda.is_available()
                      else 'cpu')


if __name__ == '__main__':
    file = open(origin_train_data, 'r', encoding='utf8')
    file.close()

