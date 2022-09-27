import torch

# 选择设备
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

model_dir = '../chinese_wwm_ext_pytorch'
origin_train_file = '../data/task3_1/task3_train.json'
# dev_file = '../data/task3_1/task3_dev_input.json'
dev_file = '../data/test_set/task3_test_input.json'
"""
本次实验不做end2end，而是做成3个步骤：
step1：token classification任务：捕捉S信息（18元组的第0号）
step2：token classification任务：捕捉STEP信息
step3：sequence classification任务：判断真假
"""
train_file = '../data/task3_1/split_train.json'
test_file = '../data/task3_1/split_test.json'

step1_model_dir = './experiment/step1'
step2_model_dir = './experiment/step2'
step3_model_dir = './experiment/step3'

log_file = './experiment/log.txt'
step1_result_file = './experiment/step1_result.json'
step2_result_file = './experiment/step2_result.json'
step3_result_file = './experiment/step3_result.json'

# 超参数
batch_size = 16
test_size = 0.1
learning_rate = 1e-5
weight_decay = 0.01
epoch_num = 30
min_epoch_num = 5
patience = 0.002
patience_num = 5
clip_grad = 5

# 设计标签
tags4step1 = [1, 2]
tags4step2 = [
    [1],  # 空间实体
    [2, 3, 4, 5, 6],  # 空间实体2（null/远/近/变远/变近）
    [7, 8, 9, 10],  # 事件（null/说话时/过去/将来）
    [],  # 事实性
    [11],  # 时间文本
    [12, 13, 14, 15],  # 时间标签参照事件（null/之时/之前/之后）
    [],  # 时间标签
    [16],  # 处所
    [17],  # 起点
    [18],  # 终点
    [19],  # 方向
    [20],  # 朝向
    [21],  # 部件处所
    [22],  # 部位
    [23],  # 形状
    [24],  # 路径
    [25],  # 距离文本
    []  # 距离标签
]

if __name__ == '__main__':
    import json

    with open(origin_train_file, 'r+',
              encoding='utf8') as file:
        data = json.load(file)
    num0 = 0
    num1 = 0
    num2 = 0
    for item in data:
        for output in item['outputs']:
            num0 += 1
            if output[1] is not None:
                num1 += 1
            if output[3] is not None:
                num2 += 1
    print(num0, num1, num2)
