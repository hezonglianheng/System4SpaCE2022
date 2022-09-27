# System4SpaCE2022
作者参加了由北京大学詹卫东教授课题组组织的SpaCE2022空间语义角色识别的标注任务。此后，作者完成了该任务3个task的transformers模型搭建和训练工作。<br>

任务官网：https://2030nlp.github.io/SpaCE2022/<br>

运行环境：<br>
CUDA v11.2<br>
Linux Ubuntu 20.04.4<br>
Pytorch v1.10<br>

项目结构：（省略部分结构相仿）
- chinese_wwm_ext_pytorch（来自哈工大科大讯飞实验室的用于BERT的预训练模型）<br>
- data（源数据及分割数据）<br>
    - task1<br>
        - task1_train.json（原始训练集）<br>
        - task1_dev_input.json（验证集）<br>
    - task2<br>
    - task2_1<br>
    - task3_1<br>
    - test_set<br>
        - task1_test_input.json（task1测试集）<br>
        - task2_test_input.json<br>
        - task3_test_input.json<br>
- get_answer（将结果.json文件转换为.jsonl文件以提交）<br>
- task1<br>
    - experiment（保存日志文件、训练好的模型及结果）<br>
    - config.py（保存文件文件夹路径）<br>
    - dataloader_build.py（实现Dataset类的子类）<br>
    - metrics.py（评估相关函数）<br>
    - run.py（程序运行入口）<br>
    - test.py（测试部分代码）<br>
    - train.py（训练部分代码）<br>
    - utils.py（创建Logger，分割训练集）<br>
- task2<br>
- task2_1<br>
- task3_1<br>

本项目基于Hugging Face 提供的BERT实现。<br>
其中，task1是简单的SequenceClassification任务模型；<br>
task2为一个TokenClassification任务的联合训练，效果差，遂放弃；<br>
task2_1根据错误种类A、B、C分为3个TokenClassification任务进行训练；<br>
task3_1分为3个step训练：<br>
step1：空间实体标注的TokenClassification任务；<br>
step2：对每个空间实体，标注其空间语义信息的TokenClassification任务；<br>
step3：对每组空间语义信息，判断真值的SequenceClassification任务<br>

训练成绩：
task1:
score: 0.6678299492385786
task2:
score1:  0.008559201141226819
score2:  0.7412048324911064
score3: 0.6396795643493179
task3: 
score； 0.4541270427631183