import torch

# 选择设备
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)
# 模型文件
model_dir = '../chinese_wwm_ext_pytorch'
# 训练、测试语料文件
origin_train_file = '../data/task2_1/task2_train.json'
# dev_file = '../data/task2_1/task2_dev_input.json_dir'
dev_file = '../data/test_set/task2_test_input.json'
train4a = '../data/task2_1/task2_train4a.json_dir'
train4b = '../data/task2_1/task2_train4b.json_dir'
train4c = '../data/task2_1/task2_train4c.json_dir'
test4a = '../data/task2_1/task2_test4a.json_dir'
test4b = '../data/task2_1/task2_test4b.json_dir'
test4c = '../data/task2_1/task2_test4c.json_dir'
# 其他文件及文件夹
log_file = './experiment/log.txt'
model4a_dir = './experiment/model4a'
model4b_dir = './experiment/model4b'
model4c_dir = './experiment/model4c'
dev_result_file = './experiment/dev_result.json'
# 超参数
batch_size = 16
test_size = 0.1
learning_rate = 1e-5
weight_decay = 0.01
patience = 0.0002
patience_num = 5
epoch_num = 30
min_epoch_num = 5
clip_grad = 5
# 标注用标签集
tags4a = {'text1': [1, 2], 'text2': [3, 4]}
tags4b = {
    'S1': [1, 2], 'P1': [3, 4], 'E1': [5, 6],
    'S2': [7, 8], 'P2': [9, 10], 'E2': [11, 12]
}
tags4c = {
    'S': [1, 2], 'P': [3, 4], 'E': [5, 6]
}


if __name__ == '__main__':
    print(tags4c['S'])
