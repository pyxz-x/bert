import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def train_valid_test_split(x_data, y_data, validation_size=0.1, test_size=0.1):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size)
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


pd_all = pd.read_csv("sample.csv")
pd_all = shuffle(pd_all)
x = '医院_事件原因分析描述'
y = '医院_事件原因分析'
x_data, y_data = pd_all[x], pd_all[y]

x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(x_data, y_data, 0.1, 0.1)

train = pd.DataFrame({'label': y_train, 'x_train': x_train})
train.to_csv(f"./{y}_train.csv", index=False, encoding='utf-8', sep='\t')
valid = pd.DataFrame({'label': y_valid, 'x_valid': x_valid})
valid.to_csv(f"./{y}_dev.csv", index=False, encoding='utf-8', sep='\t')
test = pd.DataFrame({'label': y_test, 'x_test': x_test})
test.to_csv(f"./{y}_test.csv", index=False, encoding='utf-8', sep='\t')
