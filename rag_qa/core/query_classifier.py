# 导入标准库
import json
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，从任意目录运行均可 import base
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 导入 PyTorch
import torch
# 导入日志
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class QueryClassifier:
    def __init__(self, model_path="bert_query_classifier"):
        # 初始化模型路径
        self.model_path = model_path
        # 加载 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'mps'
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self.load_model()

    def load_model(self):
        # 检查模型路径是否存在
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 初始化新模型
            self.model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")

    def save_model(self):
        """保存模型"""
        self.model.save_pretrained("./bert_outputs")  # 保存模型的权重
        self.tokenizer.save_pretrained("./bert_outputs")  # 保存分词器
        logger.info(f"模型保存至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        """预处理数据为 BERT 输入格式"""
        encodings = self.tokenizer(
            texts,
            truncation=True,  # 截断
            padding=True,  # 填充
            max_length=128,  # 最大长度
            return_tensors="pt"  # 返回pt
        )
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        """创建 PyTorch 数据集"""

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                """idx 返回对应的张量和标签"""
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """训练 BERT 分类模型"""
        # 加载数据集
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # 数据划分
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # 创建数据集 返回Dataset类型
        train_dataset = self.create_dataset(train_encodings, train_labels)
        # print(f'train_dataset--》{train_dataset[0]}')
        val_dataset = self.create_dataset(val_encodings, val_labels)
        #
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_results",  # 输出路径
            num_train_epochs=1,  # 训练轮次
            per_device_train_batch_size=8,  # 训练时候的batch_size大小
            per_device_eval_batch_size=8,  # 评估时的batch_size大小
            warmup_steps=500,  # 预热的步数
            weight_decay=0.01,  # 学习率的衰减
            logging_dir="./bert_logs",  # 日志保存
            logging_steps=10,  # 日志的输出步骤
            eval_strategy="epoch",  # 每轮评估一次
            save_strategy="epoch",  # 保存策略
            load_best_model_at_end=True,  # 最后加载是否加载最高
            save_total_limit=1,  # 只保存一个检查点，即最优的模型
            metric_for_best_model="eval_loss",  # 评估数据的loss,loss越小越好 eval_acc eval_f1, loss
            fp16=False,  # 禁用混合精度
        )

        # 初始化 Trainer
        trainer = Trainer(
            model=self.model,  # 模型
            args=training_args,  # 训练参数
            train_dataset=train_dataset,  # 训练数据
            eval_dataset=val_dataset,  # 验证数据
            compute_metrics=self.compute_metrics  # 评估方法
        )

        # 训练模型
        logger.info("开始训练 BERT 模型...")
        trainer.train()  # 开始训练
        self.save_model()  # 保存模型

        # 评估模型
        self.evaluate_model(val_texts, val_labels)  # 对最好的模型，评估

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """评估模型性能"""
        # 仅对 texts 进行分词，labels 已为数字
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        # output = model(**inputs, label)
        # loss = output.loss eval_loss
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels  # 直接使用数字标签

        logger.info("分类报告:")
        print(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        self.model.eval()
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)  # 关键词参数
            # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"


if __name__ == "__main__":
    # 初始化分类器
    classifier = QueryClassifier(
        model_path=r"D:\huggingface_cache\modelscope\models\google-bert\bert-base-chinese"
    )

    # 训练模型（路径相对本文件，从任意目录运行均可）
    data_file = Path(__file__).resolve().parents[1] / "classify_data" / "chatgpt_generate_200.jsonl"
    classifier.train_model(data_file=str(data_file))
    # 示例预测
    test_queries = [
        "AI学科的课程大纲是什么",
        "JAVA课程费用多少？",
        "5*9等于多少？",
        "AI培训有哪些老师？"
    ]
    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询: {query} -> 分类: {category}")
