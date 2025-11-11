import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, classification_report
from math import sqrt
from scipy.stats import binom_test  # 自带于 scipy，无需 statsmodels
import chardet

# 1. 读取带人工 gold label 的标注csv
label_csv = r'D:\PythonProject\LiResolver_copy\license_terms\exception_eval_gold_candidates.csv'
with open(label_csv, 'rb') as f:
    raw = f.read(4096)
    enc = chardet.detect(raw)['encoding']
print("Detected encoding:", enc)

sent2row = []
sents, gold = [], []
with open(label_csv, "r", encoding=enc, errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['label_gold'] = int(row['label_gold'].strip() == "1")
        sent2row.append(row)
        sents.append(row["sentence"].strip())
        gold.append(row['label_gold'])

# 2. 读取检测结果：规则 & NER
def load_rule_ner_sets():
    rules = set()
    with open(r'D:\PythonProject\LiResolver_copy\license_terms\exceptions_scan.json', 'r', encoding='utf-8') as f:
        for row in json.load(f):
            for s in row.get("H2_snippets", []):
                rules.add(s.strip())
            for s in row.get("H3_snippets", []):
                rules.add(s.strip())

    ner = set()
    with open(r'D:\PythonProject\LiResolver_copy\license_terms\exception_sentences_by_ner.json', "r", encoding="utf-8") as f:
        for row in json.load(f):
            ner.add(row["sentence"].strip())
    return rules, ner

rules, ner = load_rule_ner_sets()
union = rules | ner

# 3. 打标签函数
def get_pred_vec(sents, sent_set):
    return [1 if s in sent_set else 0 for s in sents]

y_pred_rule = get_pred_vec(sents, rules)
y_pred_ner = get_pred_vec(sents, ner)
y_pred_union = get_pred_vec(sents, union)
y_gold = gold

# 4. 输出Precision / Recall / F1
print("-----三方法PRF1对比-----")
names = ["Rules", "NER", "Union"]
for name, y_pred in zip(names, [y_pred_rule, y_pred_ner, y_pred_union]):
    P, R, F1, _ = precision_recall_fscore_support(y_gold, y_pred, average="binary")
    print(f"{name:6s}: Precision={P:.3f}, Recall={R:.3f}, F1={F1:.3f}")

print("\n=== Classification Reports ===")
print("Rules:\n", classification_report(y_gold, y_pred_rule, target_names=["Not-exc", "Exception"]))
print("NER:\n", classification_report(y_gold, y_pred_ner, target_names=["Not-exc", "Exception"]))
print("Union:\n", classification_report(y_gold, y_pred_union, target_names=["Not-exc", "Exception"]))

# 5. McNemar 检验（手写版，不依赖 statsmodels）
# 统计b,c：b = Rules正确 NER错误； c = Rules错误 NER正确
b, c = 0, 0
for y, r, n in zip(y_gold, y_pred_rule, y_pred_ner):
    if r == y and n != y:
        b += 1
    elif r != y and n == y:
        c += 1

n = b + c
if n == 0:
    p_value = 1.0
else:
    p_value = 2 * binom_test(x=min(b, c), n=n, p=0.5)  # 双尾检验
print("\n=== McNemar Test (Rules vs NER) ===")
print(f"b = {b}, c = {c}, n = {n}, p-value = {p_value:.5f}")
if p_value < 0.05:
    print("差异显著（p < 0.05）")
else:
    print("差异不显著（p ≥ 0.05）")

# 6. 可视化结果
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16.5,
    "figure.titlesize": 20,
})
plt.figure(figsize=(6, 4))
barlabels = ["Precision", "Recall", "F1"]
prf_mat = []
for y_pred in [y_pred_rule, y_pred_ner, y_pred_union]:
    P, R, F1, _ = precision_recall_fscore_support(y_gold, y_pred, average="binary")
    prf_mat.append([P, R, F1])
prf_mat = np.array(prf_mat)

print(prf_mat)
bar_width = 0.25
x = np.arange(3)
plt.bar(x - bar_width, prf_mat[0], width=bar_width, label="Rules", color="#a0b2c8", edgecolor="black")
plt.bar(x, prf_mat[1], width=bar_width, label="NER", color="#5e7987", edgecolor="black")
plt.bar(x + bar_width, prf_mat[2], width=bar_width, label="Union", color="#ecbc7c", edgecolor="black")

plt.xticks(range(3), barlabels)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.legend()
plt.title("Comparison of Exception Sentence Detection Methods", fontsize=20, pad=10)
plt.tight_layout()
plt.savefig("exception_detection_prf1.png", dpi=300)
plt.show()
