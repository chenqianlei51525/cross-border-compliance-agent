# 导入 Locust 相关模块，用于性能测试
from locust import HttpUser, task, between, events
# 导入 JSON 处理模块，用于解析 WebSocket 消息
import json
# 导入 WebSocket 客户端，用于模拟 WebSocket 连接
from websocket import create_connection
# 导入时间模块，用于记录 token 响应时间
import time
# 导入 CSV 写入模块，用于保存测试结果
import csv
# 导入 UUID 模块，用于生成会话 ID
import uuid
# 导入线程锁模块，确保 CSV 写入线程安全
from threading import Lock

# 定义全局变量，用于存储 token 响应时间
token_response_times = []
# 定义线程锁，确保并发写入 CSV 安全
csv_lock = Lock()

# 定义 Locust 用户类，模拟客户端行为
class QASystemUser(HttpUser):
    # 设置用户等待时间，模拟真实用户请求间隔（1-5秒）
    wait_time = between(1, 5)
    # 设置服务器主机地址
    host = "http://localhost:8000"

    def on_start(self):
        # 在用户启动时执行，初始化会话 ID
        self.session_id = str(uuid.uuid4())
        # 创建 HTTP 请求头，指定 JSON 格式
        self.headers = {"Content-Type": "application/json"}

    @task(1)
    def test_http_query(self):
        # 定义 HTTP 查询测试任务
        # 构造查询请求数据
        payload = {
            "query": "什么是 Python？",
            "source_filter": None,
            "session_id": self.session_id
        }
        # 发送 POST 请求到 /api/query 接口
        response = self.client.post("/api/query", json=payload, headers=self.headers)
        # 检查响应状态，确保请求成功
        if response.status_code == 200:
            # 记录请求成功
            self.environment.events.request.fire(
                request_type="POST",
                name="/api/query",
                response_time=response.elapsed.total_seconds() * 1000,  # 响应时间（毫秒）
                response_length=len(response.text),
                exception=None
            )
        else:
            # 记录请求失败
            self.environment.events.request.fire(
                request_type="POST",
                name="/api/query",
                response_time=response.elapsed.total_seconds() * 1000,
                response_length=0,
                exception=Exception(f"HTTP {response.status_code}")
            )

    @task(2)
    def test_websocket_stream(self):
        # 定义 WebSocket 流式查询测试任务
        # 构造 WebSocket URL
        ws_url = "ws://localhost:8000/api/stream"
        # 创建 WebSocket 连接
        ws = None
        try:
            ws = create_connection(ws_url)
            # 构造查询请求数据
            payload = {
                "query": "AI的课程大纲是什么？",
                "source_filter": None,
                "session_id": self.session_id
            }
            # 记录请求发送时间
            start_time = time.time()
            # 发送查询请求
            ws.send(json.dumps(payload))
            # 初始化 token 响应时间记录
            first_token_time = None
            last_token_time = start_time
            token_times = []

            while True:
                # 接收 WebSocket 消息
                message = ws.recv()
                # 解析消息为 JSON
                data = json.loads(message)
                # 获取当前接收时间
                current_time = time.time()

                if data["type"] == "start":
                    # 收到开始标志，继续接收
                    continue
                elif data["type"] == "token":
                    # 收到 token 消息
                    if first_token_time is None:
                        # 记录首个 token 响应时间
                        first_token_time = current_time - start_time
                    else:
                        # 记录后续 token 的响应时间（与上一个 token 的间隔）
                        token_times.append(current_time - last_token_time)
                    # 更新上一个 token 时间
                    last_token_time = current_time
                elif data["type"] == "end":
                    # 收到结束标志，处理结果
                    processing_time = current_time - start_time
                    # 记录 WebSocket 请求成功
                    self.environment.events.request.fire(
                        request_type="WebSocket",
                        name="/api/stream",
                        response_time=processing_time * 1000,  # 总响应时间（毫秒）
                        response_length=len(message),
                        exception=None
                    )
                    # 保存 token 响应时间到全局变量
                    with csv_lock:
                        token_response_times.append({
                            "session_id": self.session_id,
                            "first_token_time": first_token_time * 1000,  # 首个 token 时间（毫秒）
                            "other_token_times": [t * 1000 for t in token_times],  # 其他 token 时间（毫秒）
                            "total_time": processing_time * 1000  # 总时间（毫秒）
                        })
                    break
                elif data["type"] == "error":
                    # 收到错误消息，记录失败
                    self.environment.events.request.fire(
                        request_type="WebSocket",
                        name="/api/stream",
                        response_time=(current_time - start_time) * 1000,
                        response_length=0,
                        exception=Exception(data["error"])
                    )
                    break

        except Exception as e:
            # 捕获异常，记录 WebSocket 请求失败
            self.environment.events.request.fire(
                request_type="WebSocket",
                name="/api/stream",
                response_time=(time.time() - start_time) * 1000,
                response_length=0,
                exception=e
            )
        finally:
            # 确保 WebSocket 连接关闭
            if ws:
                ws.close()

# 定义 Locust 事件监听器，保存测试结果到 CSV
@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    # 在 Locust 测试结束时执行
    # 定义 CSV 文件路径
    csv_file = "token_response_times.csv"
    # 定义 CSV 头部
    headers = ["Session ID", "First Token Time (ms)", "Other Token Times (ms)", "Total Time (ms)"]
    # 写入测试结果到 CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入头部
        writer.writerow(headers)
        # 遍历 token 响应时间记录
        for record in token_response_times:
            # 写入每行数据
            writer.writerow([
                record["session_id"],
                f"{record['first_token_time']:.2f}",
                ";".join([f"{t:.2f}" for t in record["other_token_times"]]),
                f"{record['total_time']:.2f}"
            ])
    # 打印提示信息
    print(f"Token response times saved to {csv_file}")

# 定义 Locust 事件监听器，打印实时统计
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    # 在每次请求完成时执行
    if exception:
        # 打印请求失败信息
        print(f"{request_type} {name} failed: {exception}")
    else:
        # 打印请求成功信息
        print(f"{request_type} {name} succeeded: {response_time:.2f} ms")