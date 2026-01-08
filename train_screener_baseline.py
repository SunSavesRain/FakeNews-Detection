import os
import sys

# ================= 1. 5070 Ti 强制兼容补丁 (保持不变) =================
# os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ================= 2. 配置参数 (保持不变) =================
os.environ["HF_TOKEN"] = "hf_...fYhd"  # 记得检查 Token 是否需要填

MODEL_NAME = "answerdotai/ModernBERT-large"

# ⚠️⚠️⚠️ 这里修改为你的混合 CSV 文件路径 ⚠️⚠️⚠️
MIXED_FILE_PATH = r"I:\数据集\datasets.csv"

OUTPUT_DIR = "./results_optimized"

# 显存平衡参数
BATCH_SIZE = 8
GRADIENT_ACC_STEPS = 4
MAX_LENGTH = 1024

# ================= 3. 数据加载与预处理 (只改了这里) =================
print(f">>> 正在加载混合数据: {MIXED_FILE_PATH}")

# 1. 加载单个大文件 (这里只有一个 'train' split)
raw_dataset = load_dataset("csv", data_files=MIXED_FILE_PATH)

# 2. 自动切分：90% 训练，10% 验证 (seed确保每次切分一致)
print(">>> 正在切分训练集与验证集 (90/10)...")
split_datasets = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
# 此时 split_datasets 包含 'train' 和 'test' 两个键

print(f">>> 加载模型与分词器: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16
)


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print(">>> 正在进行并行分词预处理...")
# 对切分好的两部分分别进行 map
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"]  # 移除原始文本列，保留 label
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ================= 4. 评估指标 (保持不变) =================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# ================= 5. 5070 Ti 优化训练设置 (保持不变) =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACC_STEPS,

    bf16=True,
    fp16=False,
    tf32=True,

    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,

    logging_steps=5,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    # 这里的 key 对应 split_datasets 里的 key
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ================= 6. 开火 =================
if __name__ == "__main__":
    print(f">>> 5070 Ti (sm_120) 已就绪 | 等效 Batch Size: {BATCH_SIZE * GRADIENT_ACC_STEPS}")

    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final_model_optimized")
    trainer.save_model(final_path)
    print(f">>> ✅ 训练完成！优化版模型已保存至: {final_path}")