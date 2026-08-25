#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/3/21 15:29
@File    : chatgpt_train_bert.py
@Function :
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 标签映射
# =========================
LABEL_MAP = {"通用知识": 0, "专业咨询": 1}
ID2LABEL = {0: "通用知识", 1: "专业咨询"}


# =========================
# 数据集
# =========================
class BertClsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append({
                    "text": data["query"],
                    "label": LABEL_MAP[data["label"]]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


# =========================
# 主类（训练 + 推理）
# =========================
class BertClassifier:
    def __init__(self, model_path):
        self.model_path = model_path

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2
        )

    # =========================
    # 构建数据集
    # =========================
    def load_data(self, train_path, dev_path, test_path):
        self.train_dataset = BertClsDataset(train_path, self.tokenizer)
        self.dev_dataset = BertClsDataset(dev_path, self.tokenizer)
        self.test_dataset = BertClsDataset(test_path, self.tokenizer)

    # =========================
    # 评估指标
    # =========================
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        report = classification_report(
            labels,
            preds,
            target_names=["通用知识", "专业咨询"],
            output_dict=True
        )

        return {
            "accuracy": report["accuracy"],
            "f1": report["weighted avg"]["f1-score"]
        }

    # =========================
    # 训练
    # =========================
    def train(self, output_dir="./output"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

        # 保存模型
        self.trainer.save_model(output_dir + "/best_model")
        self.tokenizer.save_pretrained(output_dir + "/best_model")

    # =========================
    # 测试评估
    # =========================
    def evaluate(self):
        preds_output = self.trainer.predict(self.test_dataset)

        logits = preds_output.predictions
        labels = preds_output.label_ids
        preds = np.argmax(logits, axis=1)

        print("\n===== Classification Report =====")
        print(classification_report(labels, preds, target_names=["通用知识", "专业咨询"]))

        print("\n===== Confusion Matrix =====")
        print(confusion_matrix(labels, preds))

    # =========================
    # 加载模型（推理用）
    # =========================
    def load_model(self, model_dir):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    # =========================
    # 推理
    # =========================
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        return ID2LABEL[pred]


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    model_path = "/Users/chan/projects/models/bert-base-chinese"

    classifier = BertClassifier(model_path)

    # 1. 加载数据
    classifier.load_data(
        train_path="../classify_data/chatgpt_generate_200.jsonl",
        dev_path="../classify_data/chatgpt_generate_200_query.jsonl",
        test_path="../classify_data/chatgpt_generate_200_query.jsonl"
    )

    # 2. 训练
    classifier.train(output_dir="./output")

    # 3. 测试评估
    classifier.evaluate()

    # 4. 推理测试
    classifier.load_model("./output/best_model")

    text = "Java培训的课程顾问联系方式？"
    print("\n预测结果：", classifier.predict(text))
