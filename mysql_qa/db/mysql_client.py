# 导入 MySQL 连接库
import pymysql
# 导入pandas
import pandas as pd
# 导入配置和日志
from base import Config, logger


class MySQLClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        try:
            # 连接 MySQL 数据库
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                port=Config().MYSQL_PORT,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE
            )
            # 创建游标
            self.cursor = self.connection.cursor()
            # 记录连接成功
            self.logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            # 记录连接失败
            self.logger.error(f"MySQL 连接失败: {e}")
            raise

    def create_table(self):
        create_table_query = '''
                             CREATE TABLE IF NOT EXISTS jpkb
                             (
                                 id
                                 INT
                                 AUTO_INCREMENT
                                 PRIMARY
                                 KEY,
                                 subject_name
                                 VARCHAR
                             (
                                 20
                             ),
                                 question VARCHAR
                             (
                                 1000
                             ),
                                 answer VARCHAR
                             (
                                 1000
                             )
                                 ) \
                             '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"表创建失败: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            # 数据单条写入
            for _, row in data.iterrows():
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))
            self.connection.commit()
            self.logger.info("数据插入成功")
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            self.connection.rollback()  # 数据的回滚
            raise

    def fetch_questions(self):
        # 获取所有问题
        try:
            # 执行查询
            self.cursor.execute("SELECT question FROM jpkb")
            # 获取结果
            results = self.cursor.fetchall()
            # 记录获取成功
            self.logger.info("成功获取问题")
            # 返回结果
            return results
        except pymysql.MySQLError as e:
            # 记录查询失败
            self.logger.error(f"查询失败: {e}")
            # 返回空列表
            return []

    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 执行查询
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            # 获取结果
            result = self.cursor.fetchone()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            self.logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    def close(self):
        # 关闭数据库连接
        try:
            # 关闭连接
            self.connection.close()
            self.cursor.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    mysql_client = MySQLClient()
    mysql_client.create_table()
    mysql_client.insert_data('../data/JP学科知识问答.csv')
