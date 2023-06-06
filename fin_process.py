import re
filename = '/home/pgrad/xuqiankun/guwen_ner/data/guwen/output/'
# 从txt文件中逐行读取数据
input_filename = filename + "ner_output6.txt"
output_filename = filename + "result6.txt"

with open(input_filename, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 存储替换后的结果
replaced_lines = []

# 遍历每一行数据并进行处理
for line in lines:
    # 使用正则表达式提取文本和实体信息
    text_match = re.search(r"文本>>>>>：(.+?)实体>>>>>", line)
    entity_match = re.search(r"实体>>>>>：(.+)", line)

    if text_match and entity_match:
        text = text_match.group(1).strip()
        entity_data_str = entity_match.group(1).strip()

        # 提取实体数据
        entity_data = eval(entity_data_str)

        # 对实体按照位置进行排序，确保从后往前替换
        sorted_entities = []
        for entity_type, entities in entity_data.items():
            sorted_entities.extend(entities)
        sorted_entities.sort(key=lambda x: x[1], reverse=True)

        # 按照位置进行替换
        replaced_text = text
        for entity, start, end in sorted_entities:
            entity_type = next((k for k, v in entity_data.items() if (entity, start, end) in v), None)
            if entity_type:
                entity_placeholder = f"{{{entity}|{entity_type}}}"
                replaced_text = replaced_text[:start] + entity_placeholder + replaced_text[end+1:]

        replaced_line = f"{replaced_text}\n"
        replaced_lines.append(replaced_line)
    else:
        replaced_lines.append(line)

# 将替换后的结果写入新的txt文件
with open(output_filename, "w", encoding="utf-8") as file:
    file.writelines(replaced_lines)
