"""
-*- coding: utf-8 -*-
@Time : 2025-10-27 2:44
@File : save_label_map.py
@Author : 大漂亮lzh
@Project : bert_model
"""
# coding: utf-8

import os
import json
import pickle

# 1. 改成你的 output 目录
OUTPUT_DIR = r'E:\PythonProject\bert_model\output'

# 2. 导入训练时用的 Processor（确保目录能 import）
#    若提示找不到，可先 sys.path.append 父目录
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_classifier import cyyProcessor   # 类名换成你的

processor = cyyProcessor()
label_list = processor.get_labels()        # 顺序必须和训练时完全一致
label2id = {label: idx for idx, label in enumerate(label_list)}

# 3. 落盘
with open(os.path.join(OUTPUT_DIR, 'label2id.pkl'), 'wb') as f:
    pickle.dump(label2id, f)

with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)

print('✅ 已生成 label 映射：', label2id)