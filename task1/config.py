import os.path
import torch


# 文件位置
directory = os.path.pardir  # 本来应该为3个任务单独创立项目和环境的……

origin_train_data = os.path.join(directory, r'data/task1/task1_train.json_dir')  # 原始训练集
# dev_data = os.path.join(directory, r'data/task1/task1_dev_input.json_dir')  # 测试集
dev_data = '../data/test_set/task1_test_input.json_dir'
train_data = os.path.join(directory, r'data/task1/task1_train_input.json_dir')  # 分割的训练集
test_data = os.path.join(directory, r'data/task1/task1_test_input.json_dir')  # 分割的验证集

model_file = os.path.join(directory, r'chinese_wwm_ext_pytorch') # 下载好的预训练模型

log_file = r'./experiment/log.txt'  # 日志文件
result_dir = r'./experiment'  # 储存结果文件夹
acc_file = r'./experiment/acc.json'
result_file = r'./experiment/result.json'

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

# 运行设备
device = torch.device("cuda:1" if torch.cuda.is_available()
                      else "cpu")


if __name__ == "__main__":
    print(device)
