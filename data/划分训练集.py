import pandas as pd
import numpy as np  # 用于处理NaN值
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def train_valid_test_split(x_data, y_data, validation_size=0.1, test_size=0.1):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)  # 增加随机种子确保可复现
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, random_state=42)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


# 读取CSV时保留原始null值（pandas会自动将null转为NaN）
pd_all = pd.read_csv("sample.csv", keep_default_na=True)  # keep_default_na=True确保null被解析为NaN
pd_all = shuffle(pd_all, random_state=42)  # 增加随机种子

x_col = '医院_事件原因分析描述'
y_col = '医院_事件原因分析'

# 提取特征和标签（此时x_data中的null已被转为NaN，保持缺失值状态）
x_data = pd_all[x_col]
y_data = pd_all[y_col]

# 划分数据集
x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(x_data, y_data, 0.1, 0.1)

# 构建DataFrame时保留NaN值
train = pd.DataFrame({'label': y_train, x_col: x_train})  # 用原始列名更清晰
valid = pd.DataFrame({'label': y_valid, x_col: x_valid})
test = pd.DataFrame({'label': y_test, x_col: x_test})

# 保存时确保NaN被写为null（而非空字符串）
# 使用na_rep参数指定缺失值的表示方式为'null'
train.to_csv(f"./{y_col}_train.csv", index=False, encoding='utf-8', sep='\t', na_rep='null')
valid.to_csv(f"./{y_col}_dev.csv", index=False, encoding='utf-8', sep='\t', na_rep='null')
test.to_csv(f"./{y_col}_test.csv", index=False, encoding='utf-8', sep='\t', na_rep='null')