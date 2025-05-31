import json
import matplotlib.pyplot as plt

path1 = "/data/share/data/llama-factory/TokenSkip/outputs/Qwen2.5-7B-Instruct/gsm8k/7b/TokenSkip"
path2 = "/data/share/data/llama-factory/TokenSkip/outputs/Qwen2.5-7B-Instruct-my_SC/gsm8k/7b/TokenSkip"
path3 = "/data/share/data/llama-factory/TokenSkip/outputs/Qwen2.5-7B-Instruct-llmlingua2/gsm8k/7b/TokenSkip"

results1 = []
results2 = []
results3 = []

for i in range(1, 10):
    file_path = path1 + f"/{str(i / 10)}/samples/metrics.json"
    with open(file_path, "r") as f:
        data = json.load(f)
        results1.append({
            "accuracy": data["accuracy"],
            "avg_cot_length": data["avg_cot_length"]
        })
    file_path = path2 + f"/{str(i / 10)}/samples/metrics.json"
    with open(file_path, "r") as f:
        data = json.load(f)
        results2.append({
            "accuracy": data["accuracy"],
            "avg_cot_length": data["avg_cot_length"]
        })
    file_path = path3 + f"/{str(i / 10)}/samples/metrics.json"
    with open(file_path, "r") as f:
        data = json.load(f)
        results3.append({
            "accuracy": data["accuracy"],
            "avg_cot_length": data["avg_cot_length"]
        })


print(results1)
print()
print(results2)
print()
print(results3)

# 按照平均长度排序
results1.sort(key=lambda x: x["avg_cot_length"])
results2.sort(key=lambda x: x["avg_cot_length"])
results3.sort(key=lambda x: x["avg_cot_length"])

# 去掉 results3 中大于 220的 avg_cot_length
results3 = [result for result in results3 if result["avg_cot_length"] <= 220]

# 提取长度和准确率数据
lengths1 = [result["avg_cot_length"] for result in results1]
accuracies1 = [result["accuracy"] * 100 for result in results1]  # 转为百分比
lengths2 = [result["avg_cot_length"] for result in results2]
accuracies2 = [result["accuracy"] * 100 for result in results2]  # 转为百分比
lengths3 = [result["avg_cot_length"] for result in results3]
accuracies3 = [result["accuracy"] * 100 for result in results3]  # 转为百分比


# 创建折线图
plt.figure(figsize=(8, 8))
plt.plot(lengths1, accuracies1, marker='o', linestyle='-', linewidth=2, label='shuffled')
plt.plot(lengths2, accuracies2, marker='s', linestyle='-', linewidth=2, label='none')
plt.plot(lengths3, accuracies3, marker='^', linestyle='-', linewidth=2, label='llmlingua2')

# 添加标题和标签
plt.xlabel('Reasoning Tokens', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

#设置纵坐标间隔为2
plt.yticks(range(64, 90, 2), fontsize=12)
plt.xticks(range(100, 220, 20), fontsize=12)

# 显示图形
plt.tight_layout()
plt.savefig('accuracy_vs_length.png', dpi=300)
plt.show()