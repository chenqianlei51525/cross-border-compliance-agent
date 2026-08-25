# 导入 MySQL 客户端
from mysql_qa.db.mysql_client import MySQLClient
# 导入 Redis 客户端
from mysql_qa.cache.redis_client import RedisClient
# 导入 BM25 搜索
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入日志
from base import logger
# 导入时间库
import time


class MySQLQASystem:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        # 查询 MySQL 系统
        start_time = time.time()
        # 记录查询信息
        self.logger.info(f"处理查询: '{query}'")
        # 执行 BM25 搜索
        answer, _ = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 记录 MySQL 答案
            self.logger.info(f"MySQL 答案: {answer}")
        else:
            # 记录无答案
            self.logger.info("SQL中未找到答案, 需要调用RAG系统")
            # 设置默认答案
            answer = "SQL未找到答案"
        # 计算处理时间
        processing_time = time.time() - start_time
        # 记录处理时间
        self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
        # 返回答案
        return answer


def main():
    # 初始化 MySQL 系统
    mysql_system = MySQLQASystem()
    try:
        # 打印欢迎信息
        print("\n欢迎使用 MySQL 问答系统！")
        print("输入查询进行问答，输入 'exit' 退出。")
        while True:
            # 获取用户输入
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 记录退出日志
                logger.info("退出 MySQL 系统")
                # 打印退出信息
                print("再见！")
                break
            # 执行查询
            answer = mysql_system.query(query)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        mysql_system.mysql_client.close()


if __name__ == "__main__":
    # 运行主程序
    main()
