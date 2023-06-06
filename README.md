## 命名实体识别
对古籍官职、人名、书名进行繁体中文命名实体识别

## 使用流程
- 将原始数据放在/data/guwen/ner_data/文件夹中，需要根据实体标签创建labels.txt；
- 修改preprocess.py中的path路径，并运行；
- 修改config.py中的CommonConfig中的模型路径；
- 在main.py里面修改data_name，与数据集名称保持一致，并运行；
- 在predict.py修改data_name和folder_path，加入预测数据，并运行；
- 在fin_process.py中修改路径，生成目标数据格式。

## 问题勘验
- TypeError: init() got an unexpected keyword argument 'batch_first'： `pip install pytorch-crf==0.7.2`
# 依赖

```python
scikit-learn==1.1.3 
scipy==1.10.1 
seqeval==1.2.2
transformers==4.27.4
pytorch-crf==0.7.2
```

# 目录结构

```python
--checkpoint：模型和配置保存位置
--model_hub：预训练模型
----chinese-bert-wwm-ext:
--------vocab.txt
--------pytorch_model.bin
--------config.json
--data：存放数据
----guwen
--------ner_data：所有数据（训练数据、处理之后的数据）
------------labels.txt：标签（如：OFI、BOOK、PER）
------------train.txt：训练数据
------------dev.txt：测试数据
--config.py：配置
--model.py：模型
--process.py：处理ori数据得到ner数据
--predict.py：加载训练好的模型进行预测
--main.py：训练和测试
```

# 说明

这里以guwen数据为例，其余数据类似。

```python
1、在preprocess.py里面ner_data下的数据进行预处理，ner_data下不放呢数据样本：
--labels.txt
OFI
BOOK
PER
--train.txt/dev.txt
{"id": "00000", "text": ["李", "罕", "之", "以", "潞", "州", "降", "梁", "，", "晉", "人", "攻", "潞", "，", "友", "倫", "以", "兵", "入", "潞", "州", "，", "取", "罕", "之", "以", "歸", "。", "累", "遷", "檢", "校", "司", "空", "，", "領", "藤", "州", "刺", "史", "。"], "labels": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "B-OFI", "I-OFI", "I-OFI", "I-OFI", "O", "O", "O", "O", "B-OFI", "I-OFI", "O"]}
一行一条样本，格式为BIO。

2、在config.py里面定义一些参数，比如：
--max_seq_len：句子最大长度，GPU显存不够则调小。
--epochs：训练的epoch数
--train_batch_size：训练的batchsize大小，GPU显存不够则调小。
--dev_batch_size：验证的batchsize大小，GPU显存不够则调小。
--save_step：多少step保存模型
其余的可保持不变。

4、在main.py里面修改data_name为数据集名称。需要注意的是名称和data下的数据集名称保持一致。最后运行：python main.py

5、在predict.py修改data_name，将预测数据的txt文件路径放入folder_path中，最后运行：python predict.py

6、在fin_process.py中修改路径，将生成的数据转为目标要求数据格式。
```

## guwen数据集

```python
max_seq_len=512
train_batch_size=12
dev_batch_size=12
save_step=500
epochs=20
```

```python
              precision    recall  f1-score   support

        BOOK       0.85      0.85      0.85        47
         OFI       0.85      0.87      0.86       661
         PER       0.96      0.96      0.96      1286

   micro avg       0.92      0.92      0.92      1994
   macro avg       0.89      0.89      0.89      1994
weighted avg       0.92      0.92      0.92      1994
```

