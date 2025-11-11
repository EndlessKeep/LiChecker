import json
from collections import defaultdict

# 初始化一个字典，用于统计每种 relation 的个数
relation_count = defaultdict(int)


with open("D:\\PythonProject\\LiResolver_copy\\RE\\dataset\\ossl2\\val.txt", "r", encoding="utf-8") as file:
    for line in file:

        data = json.loads(line.strip())

        relation = data.get("relation")
        if relation:

            relation_count[relation] += 1

# 输出结果
for relation, count in relation_count.items():
    print(f"{relation}: {count}")
