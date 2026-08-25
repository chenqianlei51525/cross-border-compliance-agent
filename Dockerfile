# 使用国内镜像拉取 Python 基础镜像（Docker Hub 在国内常无法访问）
FROM docker.m.daocloud.io/library/python:3.10.18-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt requirements.txt

# 清理可能的缓存和临时文件
RUN rm -rf /tmp/* ~/.cache/* /root/.cache/* /opt/miniconda3/ 2>/dev/null || true

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade "pip < 24.1" && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app

# 复制项目文件
COPY . .

# # 更改文件所有权
# RUN chown -R app:app /app

# # 切换到非root用户
# USER app

# 暴露端口
EXPOSE 8080

# 启动命令，使用环境变量控制主机和端口
CMD ["sh", "-c", "python app.py"]
