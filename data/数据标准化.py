import pandas as pd
import re

# 读取文件
excel_file = pd.ExcelFile('2019年以来涉及我省注射泵生产企业的不良事件报告.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 定义要标准化的列
columns_to_standardize = [
    '医院_伤害表现', '医院_器械故障表现', '医院_使用过程', '医院_事件原因分析描述',
    '企业_伤害表现', '企业_器械故障表现', '企业_使用过程', '企业_事件原因分析'
]

# 定义函数去除空格、中文符号、英文符号，并将空值填充为 'null'
def standardize_text(text):
    if pd.isnull(text):
        return 'null'
    text = re.sub(r'[^\w\s]', '', text)  # 去除中文和英文符号
    text = re.sub(r'\s+', '', text)  # 去除空格
    return text

# 对指定列进行标准化
for col in columns_to_standardize:
    df[col] = df[col].apply(standardize_text)

# 定义函数构建新字段
def build_new_field(row, columns):
    texts = [row[col] for col in columns]
    return f"[CLS]{'[SEP]'.join(texts)}[SEP]"

# 在医院_伤害列后添加新列医院_伤害_input
hospital_columns = ['医院_伤害表现', '医院_器械故障表现', '医院_使用过程']
df.insert(df.columns.get_loc('医院_伤害') + 1, '医院_伤害_input', df.apply(lambda row: build_new_field(row, hospital_columns), axis=1))

# 在企业_伤害表现列后添加新列企业_伤害_input
enterprise_columns = ['企业_伤害表现', '企业_器械故障表现', '企业_使用过程']
df.insert(df.columns.get_loc('企业_伤害表现') + 1, '企业_伤害_input', df.apply(lambda row: build_new_field(row, enterprise_columns), axis=1))

# 将结果保存为 CSV 文件
csv_path = 'sample.csv'
df.to_csv(csv_path, index=False)