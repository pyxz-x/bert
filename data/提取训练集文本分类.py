import pandas as pd
import json


def process_multi_labels(csv_file, target_column):
    # 读取CSV文件
    df = pd.read_csv(csv_file, dtype=str)

    # 检查目标列是否存在
    if target_column not in df.columns:
        raise ValueError(
            f"列名 '{target_column}' 不存在于CSV文件中。\n"
            f"可用列名：{', '.join(df.columns)}"
        )

    # 存储处理后的唯一标签组合
    unique_combinations = set()

    # 处理每一行的多标签
    for raw_label in df[target_column].dropna():
        # 1. 清洗原始标签（去除两端空格）
        cleaned = raw_label.strip()
        if not cleaned:
            continue  # 跳过空字符串

        # 2. 拆分标签并清洗每个子标签
        labels = [label.strip() for label in cleaned.split(',')]
        # 过滤可能的空标签（如连续逗号导致）
        valid_labels = [lbl for lbl in labels if lbl]

        # 3. 对标签进行排序（核心：使顺序不同但元素相同的组合一致）
        sorted_labels = sorted(valid_labels)

        # 4. 重新组合为字符串，作为唯一标识
        standard_combination = ','.join(sorted_labels)

        # 5. 添加到集合（自动去重）
        unique_combinations.add(standard_combination)

    # 转换为排序后的列表（方便输出）
    result = sorted(unique_combinations)
    return result


if __name__ == "__main__":
    # 配置参数
    CSV_FILE = "sample.csv"
    TARGET_COLUMN = "医院_事件原因分析"  # 替换为你的表头名

    try:
        # 处理多标签组合
        processed_labels = process_multi_labels(CSV_FILE, TARGET_COLUMN)

        # 输出结果
        print(f"===== '{TARGET_COLUMN}' 列的标准化多标签组合 =====")
        for combo in processed_labels:
            print(f"'{combo}',")

        # 保存为JSON
        with open(f'{TARGET_COLUMN}_分类.json', 'w', encoding='utf-8') as f:
            json.dump(processed_labels, f, ensure_ascii=False, indent=2)

        print(f"\n处理完成：共得到 {len(processed_labels)} 种标准化多标签组合，已保存到 standardized_multi_labels.json")

    except Exception as e:
        print(f"处理失败：{str(e)}")