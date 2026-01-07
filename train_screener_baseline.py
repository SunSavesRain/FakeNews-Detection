import os
import platform

# ================= 0. 魔法配置区 (最重要！) =================
# 【关键】强制使用国内镜像站下载模型，解决连接 HuggingFace 超时的问题
# 这行必须放在 import transformers 之前
os.environ["HF_TOKEN"] = "hf_...fYhd"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 忽略一些不重要的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ================= 1. 智能路径与硬件配置 =================
system_type = platform.system()

# 基础配置
MODEL_NAME = "answerdotai/ModernBERT-large"  # 2026 SOTA 模型
OUTPUT_DIR = "./results_baseline"  # 训练结果保存位置

if system_type == 'Darwin':
    # === Mac 系统配置 ===
    # 假设你的移动硬盘挂载路径 (请根据实际情况修改 /Volumes/ 后面的名字)
    DATA_ROOT = "/Volumes/Samsung_T7/Datasets/fake_news"
    BATCH_SIZE = 2  # Mac 跑不动太大，设小点仅用于调试代码
    USE_FP16 = False  # Mac 的 MPS 有时不兼容 FP16，关掉最稳
    print(">>> 环境检测: macOS (调试模式). 正在寻找移动硬盘...")

elif system_type == 'Windows':
    # === 台式机 (5070 Ti) 配置 ===
    # 假设你在台式机上移动硬盘是 G 盘
    DATA_ROOT = r"G:\Datasets\fake_news"
    BATCH_SIZE = 8  # 5070 Ti 显存大，可以开大点 (如果 OOM 显存溢出，就改回 4)
    USE_FP16 = True  # 【关键】开启混合精度，速度翻倍，显存减半
    print(">>> 环境检测: Windows (5070 Ti 火力全开模式).")

else:
    # Linux 服务器等其他情况
    DATA_ROOT = "./data"
    BATCH_SIZE = 8
    USE_FP16 = True

# ================= 2. 数据加载逻辑 =================
train_file = os.path.join(DATA_ROOT, "train.csv")
test_file = os.path.join(DATA_ROOT, "test.csv")


# 定义一个处理 LIAR 数据集的函数 (作为备用方案)
def map_liar_to_binary(example):
    # 0:false, 1:half-true, 2:mostly-true, 3:pants-fire, 4:barely-true, 5:true
    # 将 false(0), pants-fire(3), barely-true(4) 归为假新闻 (Label 0)
    # 其他归为真新闻 (Label 1)
    if example['label'] in [0, 3, 4]:
        example['labels'] = 0
    else:
        example['labels'] = 1
    return example


# 尝试加载数据
if os.path.exists(train_file) and os.path.exists(test_file):
    print(f">>> 成功检测到本地硬盘数据: {DATA_ROOT}")
    # 加载 CSV
    dataset = load_dataset("csv", data_files={"train": train_file, "test": test_file})
    print(">>> 本地数据加载完成。")
else:
    print(f"⚠️ 未找到硬盘文件 ({train_file})，自动下载 LIAR 数据集进行代码测试...")
    # 如果没插硬盘，自动下载网上的数据集，保证代码能跑通
    dataset = load_dataset("liar", trust_remote_code=True)
    # 处理标签
    dataset = dataset.map(map_liar_to_binary)
    # 统一列名：LIAR 里的新闻内容叫 'statement'，我们统一改名叫 'text'
    dataset = dataset.rename_column("statement", "text")
    print(">>> LIAR 数据集下载并处理完成。")

# ================= 3. 模型与预处理 =================
print(f">>> 正在加载模型和分词器: {MODEL_NAME} ...")
print("(如果是第一次运行，会从镜像站下载，请耐心等待...)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def preprocess_function(examples):
    # ModernBERT 支持 8192 长度
    # 但为了训练效率，我们截断到 1024 (这已经比 BERT 的 512 长很多了)
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )


print(">>> 正在进行数据分词预处理...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# ================= 4. 训练器设置 =================
# 定义评估指标 (准确率, F1, 精确率, 召回率)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,  # 训练 3 轮
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,  # 预热步数
    weight_decay=0.01,  # 权重衰减
    logging_steps=50,  # 每50步打印一次日志
    eval_strategy="epoch",  # 每个 epoch 结束后评估一次
    save_strategy="epoch",  # 每个 epoch 结束后保存一次
    load_best_model_at_end=True,  # 训练完自动加载效果最好的那个模型
    fp16=USE_FP16,  # Windows 开 FP16，Mac 不开
    dataloader_num_workers=0,  # 设置为0以避免多进程报错 (最稳妥)
    report_to="none"  # 不上传到 WandB 等平台
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    # 如果数据集里叫 validation 就用 validation，否则用 test
    eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets else tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# ================= 5. 开始训练 =================
if __name__ == "__main__":
    print(f">>> 开始训练 (当前设备: {system_type}) ...")
    trainer.train()

    # 保存最终模型
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    print(f">>> 正在保存最终模型到: {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(">>> ✅ 训练全部完成！你可以直接在级联系统里加载这个路径使用了。")