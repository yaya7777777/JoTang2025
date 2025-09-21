from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# 设置随机种子保证可复现性
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# 加载数据集（英文SST-2）
dataset = load_dataset("glue", "sst2")

# 加载BERT token
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128, padding="max_length")

# 预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 定义评估指标函数
def compute_metrics(val_pred):
    predictions, labels = val_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary")
    }

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    val_strategy="epoch",           # 每epoch评估一次
    save_strategy="epoch",           # 每epoch保存一次
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=32,  # 训练batch size
    per_device_eval_batch_size=32,   # 验证batch size
    num_train_epochs=3,              # 训练epoch数
    weight_decay=0.01,               # 权重衰减
    load_best_model_at_end=True,     # 训练结束后加载最佳模型
    metric_for_best_model="f1",      # 用f1选择最佳模型
    seed=seed,                       # 随机种子
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    val_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

val_results = trainer.evaluate(encoded_dataset["validation"])
print(f"验证集准确率: {val_results['val_accuracy']:.4f}")
print(f"验证集F1分数: {val_results['val_f1']:.4f}")

trainer.save_model("best_bert_sst2_model")  # 保存最优权重

with open("training_config.txt", "w") as f:  # 记录超参数
    f.write(f"预训练模型: bert-base-uncased\n")
    f.write(f"学习率: 2e-5\n")
    f.write(f"训练轮数: 3\n")
    f.write(f"批大小: 32\n")
    f.write(f"最大长度: 128\n")
    f.write(f"随机种子: {seed}\n")
    f.write(f"验证集准确率: {val_results['val_accuracy']:.4f}\n")
    f.write(f"验证集F1分数: {val_results['val_f1']:.4f}\n")


# 保存模型
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")


