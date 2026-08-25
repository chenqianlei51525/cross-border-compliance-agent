# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入langchain_community的Ollama聊天模型和嵌入模型，用于本地模型调用
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
# 导入json库，用于加载JSON格式的评估数据集
import json

# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open("rag_evaluate_data.json", "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)

# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)

# 3. 配置RAGAS评估环境
# 初始化ChatOllama模型，指定使用qwen2.5:7b模型，设置本地Ollama服务地址
llm = ChatOllama(model="qwen2.5vl:7b", base_url="http://localhost:11434")
# 初始化Ollama嵌入模型，用于计算语义相似度，指定qwen2.5:7b模型
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[
        faithfulness,  # 忠实度：答案是否基于上下文
        answer_relevancy,  # 答案相关性：答案与问题的匹配度
        context_precision,  # 上下文相关性：上下文是否仅包含相关信息
        context_recall  # 上下文召回率：上下文是否包含所有必要信息
    ],
    # 传入配置好的Ollama LLM模型
    llm=llm,
    # 传入配置好的Ollama嵌入模型
    embeddings=embeddings
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)
# 将评估结果转换为pandas DataFrame，便于保存
result_df = pd.DataFrame([result])
# 将DataFrame保存为CSV文件，文件名为ragas_evaluation_results.csv，不保存索引
result_df.to_csv("ragas_evaluation_results.csv", index=False)