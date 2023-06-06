import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin")))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    data_name = "guwen"
    predictor = Predictor(data_name)

    folder_path = '/home/pgrad/xuqiankun/guwen_ner/data/guwen/output/'
    output_file = folder_path + 'ner_output6.txt'# 文件夹路径
    file_path = folder_path + 'GuNER2023_test_public.txt'

    with open(output_file, "w", encoding="utf-8") as output:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()  # 去除行尾的换行符和空白字符
                print(text)
                # 对每一行进行处理或打印等操作
                # output.write(text + "\n")
                ner_result = predictor.ner_predict(text)
                print("文本>>>>>：", text)
                print("实体>>>>>：", ner_result)
                print("=" * 100)
                # output.write("文本>>>>>：" + text + "\n")
                output.write("文本>>>>>：" + text)
                output.write("实体>>>>>：" + str(ner_result) + "\n")
                    # output.write("=" * 100 + "\n")
    # for txt_file in txt_files:
    #     file_path = os.path.join(folder_path, txt_file)  # 构建文件的完整路径
    #     with open(file_path, "r", encoding="utf-8") as file:
    #         for line in file:
    #             text = line.strip()  # 去除行尾的换行符和空白字符
    #             # 对每一行进行处理或打印等操作
    #             print(text)
    #             ner_result = predictor.ner_predict(text)
    #             print("文本>>>>>：", text)
    #             print("实体>>>>>：", ner_result)
    #             print("=" * 100)



