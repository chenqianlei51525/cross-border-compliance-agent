# 导入配置解析库
import configparser
# 导入路径操作库
import os


class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.ini",
            )
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取项目自身配置，显式使用 UTF-8 兼容 Windows 中文环境。
        self.config.read(config_file, encoding="utf-8")

        # MySQL 配置
        # MySQL 主机地址
        # 获取section 是mysql host变量失败返回localhost
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_PORT = self.config.getint('mysql', 'port', fallback=3306)

        # MySQL 用户名
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # MySQL 密码
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 端口
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        # self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        self.DASHSCOPE_API_KEY = os.getenv("API_KEY") or self.config.get(
            'llm', 'dashscope_api_key', fallback=None
        )
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 业务场景：跨境电商合规智能问答（CE/FCC/RoHS/UN38.3/PSE/KCC/EUP）
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources',
                            fallback='["ce-red", "fcc", "rohs", "un38.3", "pse", "kcc", "eup"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')
        # 日志文件路径
        configured_log_file = self.config.get('logger', 'log_file', fallback='logs/app.log')
        if os.name == "nt" and configured_log_file.startswith("/Users/"):
            configured_log_file = "logs/app.log"
        self.LOG_FILE = (
            configured_log_file
            if os.path.isabs(configured_log_file)
            else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              configured_log_file)
        )

        # model path
        # self.bge_m3 = "/Users/chan/projects/Itcast_qa_system/rag_qa/models/bge-m3"
        # self.bge_reranker = "/Users/chan/projects/Itcast_qa_system/rag_qa/models/bge-reranker-large"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nlp_bert_doc_seg = os.path.join(
            project_root,
            "rag_qa",
            "models",
            "nlp_bert_document-segmentation_chinese-base",
        )
        self.bert_intent_cls = os.path.join(
            project_root,
            "rag_qa",
            "core",
            "bert_query_classifier",
        )


if __name__ == '__main__':
    conf = Config()
    print(conf.MYSQL_HOST)
