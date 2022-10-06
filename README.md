# System4SpaCE2022
作者参加了由[北京大学詹卫东教授课题组](http://ccl.pku.edu.cn/seminar/)组织的SpaCE2022空间语义角色识别的标注任务。此后，作者完成了该任务3个task的transformers模型搭建和训练工作。<br>

任务官网：https://2030nlp.github.io/SpaCE2022/<br>

运行环境：<br>
CUDA v11.2<br>
Linux Ubuntu 20.04.4<br>
Pytorch v1.10<br>

项目结构：（省略部分结构相仿）
- chinese_wwm_ext_pytorch（来自[哈工大科大讯飞实验室](https://github.com/ymcui/Chinese-BERT-wwm)的用于BERT的预训练模型）<br>
- data（源数据及分割数据）<br>
    - task1<br>
        - task1_train.json（原始训练集）<br>
        - task1_dev_input.json（验证集）<br>
    - task2<br>
    - task2_1<br>
    - task3_1<br>
    - test_set<br>
        - task1_test_input.json（task1测试集）<br>
        - task2_test_input.json（task2测试集）<br>
        - task3_test_input.json（task3测试集）<br>
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

本项目基于[Hugging Face](https://huggingface.co/)提供的BERT实现。<br>
其中，task1是简单的SequenceClassification任务模型；<br>
task2将三个错误类型合并为一个TokenClassification任务，进行联合训练，
效果差，遂放弃；<br>
task2_1将数据集根据错误种类A、B、C分为3个部分，将识别每个错误类型视为一个TokenClassification任务，之后将测试集分别通过3个TokenClassification模型得到答案；<br>
task3_1分为3个step训练：<br>
step1：空间实体（S信息）标注的TokenClassification任务；<br>
step2：对每个空间实体（S信息），标注其空间语义信息（TEP信息）的TokenClassification任务；<br>
step3：对每组空间语义信息，判断真值的SequenceClassification任务<br>

训练成绩：<br>
task1:<br>
score: 0.6678299492385786<br>
task2:<br>
score1:  0.008559201141226819<br>
score2:  0.7412048324911064<br>
score3: 0.6396795643493179<br>
task3: <br>
score: 0.4541270427631183<br>

一些个人总结：<br>
- 本系统的问题：<br>
    - 本系统提出的方法没有经过充分的验证和试错（尤其是task3）<br>
    - task1的得分过低，应当优化。是否要考虑从Pytorch搭建模型而非使用现有模型？<br>
    - task2会出现某实体同时作为S1和S2的情况。可能在标注设计时应增加标注。<br>
    - task3的错误传递问题无法解决。<br>
- 对本次任务的看法：
    - 本次任务task3的标注太多，而且很多时候互相重叠，不利于ML系统搭建、训练、测试、评估（还是比较for linguistic）；<br>
    - 同样的，task2也存在标注互相重叠的情形；<br>
    - 数据量过少，训练效果很可能不理想。是否要考虑小样本学习？<br>
