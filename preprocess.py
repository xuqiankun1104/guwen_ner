# -*- encoding: utf-8 -*-

import re
import random
import pandas as pd

import json
import random
# labels文件需要人工创建
path= '/home/pgrad/xuqiankun/guwen_ner/data/guwen/ner_data/'# 原始语料的路径
txt_file= path+'GuNER2023_train.txt'
tsv_file= path+'input.tsv'# 一下文件加
train_json_file = path+'train.json'
test_json_file = path+'test.json'
train_txt_file = path+'train.txt'
test_txt_file = path+'dev.txt'

def tag(single_entity):
    token_tag = re.split('\|', single_entity)
    token_tag = [[i,'I-'+ token_tag[1]] for i in list(token_tag[0])] # [i,'I-'+ token_tag[1]]
    token_tag[0][1]=re.sub('I-','B-',token_tag[0][1])
    token_tag = [i[0]+'\t'+i[1] for i in token_tag]
    return token_tag


def format_sen(raw_sen):
    final= []
    for idx, i in enumerate(re.split('(\{.+?\})', raw_sen)): #
        if not re.findall('(\{.+?\})',i):
            final.extend([i+'\t'+'O' for i in list(i)])
        else:
            final.extend(tag(re.sub('\{|\}','',i)))

    return final


data = open(txt_file,'r',encoding='utf-8').readlines()
data = [str(i).strip('\n') for i in data]
with open(tsv_file,'w',encoding='utf-8')as te:
    for i in data:
        for j in format_sen(i):
            te.write(j+'\n')
        te.write('\n')
# random.seed(2023)
# random.shuffle(data)
#
#
#
# cut_idx = len(data) // 10
# # 划分训练、测试集
# train_corpus = data[cut_idx:]
# test_corpus = data[:cut_idx]
#
#
# with open('test.tsv','w',encoding='utf-8')as te:
#     for i in test_corpus:
#         for j in format_sen(i):
#             te.write(j+'\n')
#         te.write('\n')
#
# with open('train.tsv','w',encoding='utf-8')as tr:
#     for i in train_corpus:
#         for j in format_sen(i):
#             tr.write(j+'\n')
#         tr.write('\n')


## 看一下具体有哪些标记
# data = ''.join(open('/home/pgrad/LinLitao/20230422-guner/data/GuNER2023_train.txt','r',encoding='utf-8').readlines())
# entity = re.findall('\{(.+?)\}',data)

# tags = set([re.split('\|', e)[1] for e in set(entity)])
# print(tags)



data = []
with open(tsv_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    temp = []
    for line in lines:
        line = line.strip()
        if line:
            text, label = line.split('\t')
            temp.append({'text': text, 'label': label})
        else:
            if temp:
                data.append(temp)
                temp = []
    if temp:
        data.append(temp)

# 随机划分数据集为训练集和测试集
random.shuffle(data)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

def convert_data_to_json(data):
    json_data_list = []
    for i, item in enumerate(data):
        json_data = {
            'id': str(i).zfill(5),
            'text': [d['text'] for d in item],
            'labels': [d['label'] for d in item]
        }
        json_data_list.append(json_data)
    return json_data_list

# 保存训练集为 JSON 文件
with open(train_json_file, 'w', encoding='utf-8') as file:
    json.dump(convert_data_to_json(train_data), file, ensure_ascii=False)

# 保存测试集为 JSON 文件
with open(test_json_file, 'w', encoding='utf-8') as file:
    json.dump(convert_data_to_json(test_data), file, ensure_ascii=False)

# 将训练集 JSON 写入为 TXT 文件
with open(train_txt_file, 'w', encoding='utf-8') as file:
    train_data = json.load(open(train_json_file, 'r', encoding='utf-8'))
    file.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

# 将测试集 JSON 写入为 TXT 文件
with open(test_txt_file, 'w', encoding='utf-8') as file:
    test_data = json.load(open(test_json_file, 'r', encoding='utf-8'))
    file.write("\n".join([json.dumps(d, ensure_ascii=False) for d in test_data]))
