"""
-*- coding: utf-8 -*-
@Time : 2025-10-26 21:57
@File : get_lable.py
@Author : 大漂亮lzh
@Project : bert_model
"""
# get_real_labels.py
import pandas as pd
import json

# 1. 读取 csv（无表头）
df = pd.read_csv(r'E:\PythonProject\bert_model\data\sample.csv',
                 header=None, dtype=str, encoding='gbk')

# 2. 拆分多标签（逗号分隔）
all_labels = set()
for raw in df[0].dropna():
    all_labels.add(raw.strip())

# 3. 排序
label_list = sorted(all_labels)

# 4. 输出
print("===== 真实标签列表 =====")
for lbl in label_list:
    print(f"'{lbl}',")

# 5. 可选：保存 json 备用
with open('real_labels.json', 'w', encoding='utf-8') as f:
    json.dump(label_list, f, ensure_ascii=False, indent=2)

print("\n已写入 real_labels.json，共", len(label_list), "个标签")