import pandas as pd

question_file_path = '/root/work/cMQA/questions.csv'
questions_df = pd.read_csv(question_file_path)
#print(questions_df.head(10))

ans_file_path = '/root/work/cMQA/answers.csv'
ans_df = pd.read_csv(ans_file_path)
#print(ans_df.head(10))

merged_df = pd.merge(questions_df, ans_df, on='question_id', suffixes=('_question', '_answer'))
#print(merged_df.head(10))

import json
def export_conversations_to_json(df, num_records, file_name):
    """
    将对话数据导出到 JSON 文件。

    :param df: 包含对话数据的 DataFrame。
    :param num_records: 要导出的记录数。
    :param file_name: 输出 JSON 文件的名称。
    """
    output = []

    # 遍历 DataFrame 并构建所需的数据结构
    for i, row in df.head(num_records).iterrows():
        identity = f"identity_{i}"
        conversation = [
            {"from": "user", "value": row['content_question']},
            {"from": "assistant", "value": row['content_answer']}
        ]
        output.append({"id": identity, "conversations": conversation})

    # 将列表转换为 JSON 格式并保存为文件
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

export_conversations_to_json(merged_df, 20000, 'medical_treatment.json')

###
###
###
import pandas as pd

# 假设df是你的DataFrame
df = pd.read_json('medical_treatment.json')

# 只查看前2条数据
print(df.head(2))

