比赛数据下载 https://pan.baidu.com/s/1utAjdzBBuASdGtCauBVnrQ 提取码：6f36

项目文件框架：
---data
------->比赛数据集（训练集以及全部测试集）
---model
------->存放所有训练得到的模型文件
---result
------->输出结果文件，其中sub_merge_all为最终结果文件

utils.py 自定义方法类
model.py 模型结构代码
pretrainW2V.py 预训练词向量
train.py 训练模型
predict.py 预测B榜结果

若仅仅预测结果，则只需要执行predict.py文件.
由于在比赛过程中未保留最终模型权重，存放的模型为最近生成，keras复现会存在一定范围内的微小浮动

若需要从头训练，则先执行pretrainW2V生成词向量权重与词表，再执行train.py生产两个模型的5折权重，再通过predict进行预测

