from datasets import load_dataset

# 加载数据集
dataset = load_dataset("allenai/qasper")

dataset["train"].to_json("train.json")
dataset["validation"].to_json("validation.json")
dataset["test"].to_json("test.json")