# -*- coding:utf-8 -*-
import hashlib
import os

import numpy as np

# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
from rag_qa.core.document_processor import process_documents
from base import logger, Config

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)

conf = Config()


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # print(f'reranker_path--》{reranker_path}')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=m3_path, use_fp16=(self.device == 'cuda'),
                                                         device=self.device)
        # 获取稠密向量的维度# 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 类私有化方法
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents):
        if not documents:
            return 0
        # print(f'documents--》{documents[0]}')
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]

        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        embeddings = self.embedding_function(texts)
        # print(f'embeddings--》{embeddings}')
        # print(f'embeddings--》{embeddings.keys()}')
        # 初始化空列表，存储插入的数据
        data = []
        # 遍历每个文档，带上索引i
        for i, doc in enumerate(documents):
            # 增量更新文档优先使用切分阶段生成的稳定 ID；旧导入流程继续使用内容哈希。
            text_hash = doc.metadata.get("id") or hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()
            if len(text_hash) > 100:
                text_hash = hashlib.sha256(text_hash.encode("utf-8")).hexdigest()
            # print(f'text_hash--》{text_hash}')
            # print(f'text_hash--》{type(text_hash)}')
            # 初始化一个稀疏向量的字典（Milvus要求存储稀疏向量的格式）
            sparse_vector = {}

            try:
                # 新版本 milvus-model 使用 coo_array 格式
                row = embeddings["sparse"][i]
                if hasattr(row, 'col'):  # coo_array 格式
                    indices = row.col
                    values = row.data
                else:  # csr_matrix 格式
                    indices = row.indices
                    values = row.data
            except Exception:
                # 兼容旧版本 milvus-model
                row = embeddings["sparse"].getrow(i)
                indices = row.indices
                values = row.data

            # 获取第i行对应的稀疏向量数据[0.4, 0.2, 0, 0, 0.1]
            # row = embeddings["sparse"].getrow(i)
            # row = embeddings["sparse"][i]  # 新版本milvus-model，支持这种获取稀疏向量的形式
            # indices = row.row
            # print(f'row--》{row}')
            # print(f'row--》{row.shape}')
            # 获取稀疏向量的非零值的索引
            # indices = row.col
            # print(f'indics--》{indics}')
            # 获取稀疏向量的非零值
            # values = row.data

            # 将索引和值进行配对，存储到字典中
            for idx, value in zip(indices, values):
                sparse_vector[int(idx)] = float(value)
            # print(f'sparse_vector--》{sparse_vector}')
            # print(f'sparse_vector--》{len(sparse_vector)}')
            # print(embeddings["dense"][i])
            # print(embeddings["dense"][i].shape)
            # 创建数据字典，包含所有字段
            item = {
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": np.asarray(
                    embeddings["dense"][i], dtype=np.float32
                ),
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            }
            # 集合开启了动态字段，这些字段无需重建既有 Milvus collection。
            for metadata_key in (
                "document_id",
                "document_version",
                "chunk_index",
                "content_hash",
                "file_name",
                "file_path",
            ):
                metadata_value = doc.metadata.get(metadata_key)
                if metadata_value is not None:
                    item[metadata_key] = metadata_value
            data.append(item)
        # 检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 等待数据持久化，使 Attu 和 collection stats 能立即看到最新实体。
            self.client.flush(collection_name=self.collection_name)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新 {len(data)} 个文档")
        return len(data)

    @staticmethod
    def _escape_filter_value(value):
        """转义 Milvus 字符串过滤表达式中的特殊字符。"""
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def delete_document_versions_before(self, document_id, version):
        """删除某文档指定版本之前的残余分块。"""
        escaped_id = self._escape_filter_value(document_id)
        filter_expr = (
            f'document_id == "{escaped_id}" and document_version < {int(version)}'
        )
        result = self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr,
        )
        logger.info(f"已清理文档 {document_id} 的 v{version} 之前分块")
        return result

    def delete_by_document_id(self, document_id):
        """按稳定 document_id 删除文档在 Milvus 中的全部分块。"""
        escaped_id = self._escape_filter_value(document_id)
        result = self.client.delete(
            collection_name=self.collection_name,
            filter=f'document_id == "{escaped_id}"',
        )
        logger.info(f"已删除文档 {document_id} 的全部向量分块")
        return result

    def get_collection_info(self):
        """返回前端知识库控制台需要的 Milvus 连接与集合信息。"""
        stats = self.client.get_collection_stats(
            collection_name=self.collection_name
        )
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "collection": self.collection_name,
            "row_count": int(stats.get("row_count", 0)),
        }

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        # print(f'dense_query_vector--》{dense_query_vector.shape}')
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # ====================
        # 获取查询稀疏向量的第 0 行数据
        # row = query_embeddings["sparse"][0]
        # # 获取稀疏向量的非零值索引
        # indices = row.indices
        # # 获取稀疏向量的非零值
        # values = row.data
        # ====================
        try:
            # 新版本 milvus-model 使用 coo_array 格式
            row = query_embeddings["sparse"][0]
            if hasattr(row, 'col'):  # coo_array 格式
                indices = row.col
                values = row.data
            else:  # csr_matrix 格式
                indices = row.indices
                values = row.data
        except Exception:
            # 兼容旧版本 milvus-model
            row = query_embeddings["sparse"].getrow(0)
            indices = row.indices
            values = row.data

        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        # print(f'sparse_query_vector-->{sparse_query_vector}')
        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # print(f'filter_expr--》{filter_expr}')
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(1.0, 0.7)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]
        # print(f'results--》{results}')
        # print(f'results--》{type(results)}')
        # print(f'results--》{len(results)}')
        # 将上述搜索到的结果进行Document对象封装，便于查询使用
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # print(f'sub_chunks--》{len(sub_chunks)}')
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')
        # # 如果只有1个文档或者没有，直接返回跳过重排序
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # print(f'scores--》{scores}')
            # 根据得分从高到低排序文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # 如果没有父文档，返回空列表
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 m 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
            # 返回去重后的父文档列表
        return unique_docs

    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )


if __name__ == "__main__":
    vector_store = VectorStore()  # 建表
    directory_path = '/Users/chan/projects/Itcast_qa_system/rag_qa/data/ai_data'
    print(f"embedding_function.dim--》{vector_store.embedding_function.dim}")
    documents = process_documents(directory_path)  # 文档读取与切块
    vector_store.add_documents(documents)  # 子块编码并且写入Milvus

    # query = "AI学科学费是多少？"
    # results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    # print(f'results-->{results}')
    # print(f'results-->{len(results)}')
