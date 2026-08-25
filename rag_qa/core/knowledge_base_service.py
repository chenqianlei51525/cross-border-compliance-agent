"""文档型知识库的新增、修改、删除与任务状态管理。"""

import hashlib
import os
import shutil
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor

from base import Config, logger
from rag_qa.core.document_processor import document_loaders, process_document


class KnowledgeDocumentNotFoundError(Exception):
    pass


class KnowledgeDocumentConflictError(Exception):
    pass


class MySQLKnowledgeRegistry:
    """使用短连接记录文档、版本和异步任务，避免共享游标的并发问题。"""

    def __init__(self, config=None):
        self.config = config or Config()
        self._init_schema()

    def _connect(self):
        return pymysql.connect(
            host=self.config.MYSQL_HOST,
            port=self.config.MYSQL_PORT,
            user=self.config.MYSQL_USER,
            password=self.config.MYSQL_PASSWORD,
            database=self.config.MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=False,
        )

    def _init_schema(self):
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_documents (
                        document_id VARCHAR(36) PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        file_path VARCHAR(1024) NOT NULL,
                        source VARCHAR(50) NOT NULL,
                        content_hash CHAR(64) NOT NULL,
                        version INT NOT NULL DEFAULT 1,
                        status VARCHAR(20) NOT NULL,
                        chunk_count INT NOT NULL DEFAULT 0,
                        error_message TEXT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        INDEX idx_kd_status (status),
                        INDEX idx_kd_source (source)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_tasks (
                        task_id VARCHAR(36) PRIMARY KEY,
                        document_id VARCHAR(36) NOT NULL,
                        operation VARCHAR(20) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        file_name VARCHAR(255) NULL,
                        file_path VARCHAR(1024) NULL,
                        source VARCHAR(50) NULL,
                        content_hash CHAR(64) NULL,
                        target_version INT NULL,
                        chunk_count INT NOT NULL DEFAULT 0,
                        error_message TEXT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        INDEX idx_kt_document (document_id),
                        INDEX idx_kt_status (status)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _json_ready(row):
        if row is None:
            return None
        return {
            key: value.isoformat() if isinstance(value, (datetime, date)) else value
            for key, value in row.items()
        }

    def get_document(self, document_id):
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM knowledge_documents WHERE document_id = %s",
                    (document_id,),
                )
                return self._json_ready(cursor.fetchone())
        finally:
            connection.close()

    def list_documents(self, include_deleted=False):
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM knowledge_documents"
                params = ()
                if not include_deleted:
                    sql += " WHERE status <> %s"
                    params = ("deleted",)
                sql += " ORDER BY updated_at DESC"
                cursor.execute(sql, params)
                return [self._json_ready(row) for row in cursor.fetchall()]
        finally:
            connection.close()

    def get_task(self, task_id):
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM knowledge_tasks WHERE task_id = %s",
                    (task_id,),
                )
                return self._json_ready(cursor.fetchone())
        finally:
            connection.close()

    def create_document_and_task(self, document, task):
        now = datetime.now()
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO knowledge_documents (
                        document_id, file_name, file_path, source, content_hash,
                        version, status, chunk_count, error_message, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 0, NULL, %s, %s)
                    """,
                    (
                        document["document_id"], document["file_name"],
                        document["file_path"], document["source"],
                        document["content_hash"], document["version"],
                        document["status"], now, now,
                    ),
                )
                self._insert_task(cursor, task, now)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _insert_task(cursor, task, now):
        cursor.execute(
            """
            INSERT INTO knowledge_tasks (
                task_id, document_id, operation, status, file_name, file_path,
                source, content_hash, target_version, chunk_count,
                error_message, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, NULL, %s, %s)
            """,
            (
                task["task_id"], task["document_id"], task["operation"],
                task["status"], task.get("file_name"), task.get("file_path"),
                task.get("source"), task.get("content_hash"),
                task.get("target_version"), now, now,
            ),
        )

    def create_task_and_set_document_status(self, task, document_status):
        now = datetime.now()
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_documents
                    SET status = %s, error_message = NULL, updated_at = %s
                    WHERE document_id = %s
                    """,
                    (document_status, now, task["document_id"]),
                )
                if cursor.rowcount != 1:
                    raise KnowledgeDocumentNotFoundError(task["document_id"])
                self._insert_task(cursor, task, now)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_task_running(self, task_id):
        self._update_task(task_id, status="running", error_message=None)

    def complete_upsert(self, task_id, chunk_count):
        task = self.get_task(task_id)
        if task is None:
            raise KnowledgeDocumentNotFoundError(f"任务不存在: {task_id}")
        now = datetime.now()
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_documents
                    SET file_name = %s, file_path = %s, source = %s,
                        content_hash = %s, version = %s, status = 'active',
                        chunk_count = %s, error_message = NULL, updated_at = %s
                    WHERE document_id = %s
                    """,
                    (
                        task["file_name"], task["file_path"], task["source"],
                        task["content_hash"], task["target_version"], chunk_count,
                        now, task["document_id"],
                    ),
                )
                cursor.execute(
                    """
                    UPDATE knowledge_tasks
                    SET status = 'succeeded', chunk_count = %s,
                        error_message = NULL, updated_at = %s
                    WHERE task_id = %s
                    """,
                    (chunk_count, now, task_id),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete_delete(self, task_id):
        task = self.get_task(task_id)
        if task is None:
            raise KnowledgeDocumentNotFoundError(f"任务不存在: {task_id}")
        now = datetime.now()
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_documents
                    SET status = 'deleted', chunk_count = 0,
                        error_message = NULL, updated_at = %s
                    WHERE document_id = %s
                    """,
                    (now, task["document_id"]),
                )
                cursor.execute(
                    """
                    UPDATE knowledge_tasks
                    SET status = 'succeeded', error_message = NULL, updated_at = %s
                    WHERE task_id = %s
                    """,
                    (now, task_id),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def fail_task(self, task_id, error_message):
        task = self.get_task(task_id)
        if task is None:
            return
        document = self.get_document(task["document_id"])
        # 只有确实存在上一版有效分块时，更新/删除失败后才能恢复为 active。
        # 首次构建失败或从未成功入库的文档必须保持 failed，避免前端显示“已生效 0 分块”。
        document_status = (
            "active"
            if document and int(document.get("chunk_count") or 0) > 0
            else "failed"
        )
        now = datetime.now()
        safe_error = str(error_message)[:4000]
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_tasks
                    SET status = 'failed', error_message = %s, updated_at = %s
                    WHERE task_id = %s
                    """,
                    (safe_error, now, task_id),
                )
                cursor.execute(
                    """
                    UPDATE knowledge_documents
                    SET status = %s, error_message = %s, updated_at = %s
                    WHERE document_id = %s
                    """,
                    (document_status, safe_error, now, task["document_id"]),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _update_task(self, task_id, status, error_message):
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_tasks
                    SET status = %s, error_message = %s, updated_at = %s
                    WHERE task_id = %s
                    """,
                    (status, error_message, datetime.now(), task_id),
                )
                if cursor.rowcount != 1:
                    raise KnowledgeDocumentNotFoundError(f"任务不存在: {task_id}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class KnowledgeBaseService:
    """协调文件存储、文档解析、Milvus 增量更新和 MySQL 状态。"""

    BUSY_STATUSES = {"pending", "indexing", "updating", "deleting"}

    def __init__(self, vector_store, registry=None, config=None, storage_root=None):
        self.vector_store = vector_store
        self.config = config or Config()
        self.registry = registry or MySQLKnowledgeRegistry(self.config)
        project_root = Path(__file__).resolve().parents[2]
        configured_root = storage_root or os.getenv(
            "KNOWLEDGE_STORAGE_DIR", str(project_root / "knowledge_storage")
        )
        self.storage_root = Path(configured_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.trash_root = self.storage_root / "_trash"
        self.trash_root.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = int(
            os.getenv("KNOWLEDGE_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
        )
        self._locks = {}
        self._locks_guard = threading.Lock()

    @property
    def supported_extensions(self):
        return sorted(document_loaders.keys())

    def _document_lock(self, document_id):
        with self._locks_guard:
            return self._locks.setdefault(document_id, threading.Lock())

    @staticmethod
    def _validate_document_id(document_id):
        try:
            return str(uuid.UUID(str(document_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("document_id 必须是合法 UUID") from exc

    def _validate_source(self, source):
        source = (source or "").strip()
        if source not in self.config.VALID_SOURCES:
            raise ValueError(
                f"无效 source: {source}，可选值: {', '.join(self.config.VALID_SOURCES)}"
            )
        return source

    def _validate_file_name(self, file_name):
        safe_name = Path(file_name or "").name
        extension = Path(safe_name).suffix.lower()
        if not safe_name or extension not in document_loaders:
            raise ValueError(
                f"不支持的文件类型，支持类型: {', '.join(self.supported_extensions)}"
            )
        return safe_name

    def _stage_file(self, document_id, version, file_name, content):
        if not content:
            raise ValueError("上传文件不能为空")
        if len(content) > self.max_upload_bytes:
            raise ValueError(
                f"文件超过大小限制 {self.max_upload_bytes // 1024 // 1024} MB"
            )
        safe_name = self._validate_file_name(file_name)
        version_dir = self.storage_root / document_id / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        file_path = (version_dir / safe_name).resolve()
        if self.storage_root not in file_path.parents:
            raise ValueError("非法文件路径")
        file_path.write_bytes(content)
        return safe_name, str(file_path), hashlib.sha256(content).hexdigest()

    def submit_create(self, file_name, content, source):
        source = self._validate_source(source)
        document_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        safe_name, file_path, content_hash = self._stage_file(
            document_id, 1, file_name, content
        )
        document = {
            "document_id": document_id,
            "file_name": safe_name,
            "file_path": file_path,
            "source": source,
            "content_hash": content_hash,
            "version": 1,
            "status": "pending",
        }
        task = {
            "task_id": task_id,
            "document_id": document_id,
            "operation": "create",
            "status": "pending",
            "file_name": safe_name,
            "file_path": file_path,
            "source": source,
            "content_hash": content_hash,
            "target_version": 1,
        }
        try:
            self.registry.create_document_and_task(document, task)
        except Exception:
            shutil.rmtree(self.storage_root / document_id, ignore_errors=True)
            raise
        return self.registry.get_task(task_id)

    def submit_update(self, document_id, file_name=None, content=None, source=None):
        document_id = self._validate_document_id(document_id)
        lock = self._document_lock(document_id)
        with lock:
            document = self.registry.get_document(document_id)
            if document is None or document["status"] == "deleted":
                raise KnowledgeDocumentNotFoundError(document_id)
            if document["status"] in self.BUSY_STATUSES:
                raise KnowledgeDocumentConflictError(
                    f"文档当前状态为 {document['status']}，请等待当前任务完成"
                )
            if content is None and source is None:
                raise ValueError("修改文档时 file 和 source 至少提供一个")

            target_version = int(document["version"]) + 1
            new_source = self._validate_source(source) if source is not None else document["source"]
            if content is not None:
                safe_name, file_path, content_hash = self._stage_file(
                    document_id, target_version, file_name, content
                )
            else:
                if not Path(document["file_path"]).is_file():
                    raise ValueError("原始文档文件不存在，无法重新构建索引")
                safe_name = document["file_name"]
                file_path = document["file_path"]
                content_hash = document["content_hash"]

            task_id = str(uuid.uuid4())
            task = {
                "task_id": task_id,
                "document_id": document_id,
                "operation": "update",
                "status": "pending",
                "file_name": safe_name,
                "file_path": file_path,
                "source": new_source,
                "content_hash": content_hash,
                "target_version": target_version,
            }
            self.registry.create_task_and_set_document_status(task, "updating")
            return self.registry.get_task(task_id)

    def submit_delete(self, document_id):
        document_id = self._validate_document_id(document_id)
        lock = self._document_lock(document_id)
        with lock:
            document = self.registry.get_document(document_id)
            if document is None or document["status"] == "deleted":
                raise KnowledgeDocumentNotFoundError(document_id)
            if document["status"] in self.BUSY_STATUSES:
                raise KnowledgeDocumentConflictError(
                    f"文档当前状态为 {document['status']}，请等待当前任务完成"
                )
            task_id = str(uuid.uuid4())
            task = {
                "task_id": task_id,
                "document_id": document_id,
                "operation": "delete",
                "status": "pending",
                "target_version": document["version"],
            }
            self.registry.create_task_and_set_document_status(task, "deleting")
            return self.registry.get_task(task_id)

    def process_task(self, task_id):
        """FastAPI BackgroundTasks 调用入口。"""
        task = self.registry.get_task(task_id)
        if task is None:
            logger.error(f"知识库任务不存在: {task_id}")
            return
        if task["status"] != "pending":
            logger.warning(
                f"跳过非 pending 知识库任务: task_id={task_id}, status={task['status']}"
            )
            return

        lock = self._document_lock(task["document_id"])
        with lock:
            try:
                self.registry.mark_task_running(task_id)
                if task["operation"] in {"create", "update"}:
                    chunks = process_document(
                        file_path=task["file_path"],
                        source=task["source"],
                        document_id=task["document_id"],
                        document_version=task["target_version"],
                        content_hash=task["content_hash"],
                    )
                    chunk_count = self.vector_store.add_documents(chunks)
                    if task["operation"] == "update":
                        # 新版本写入成功后再清理旧版本独有分块，旧索引不会先被清空。
                        self.vector_store.delete_document_versions_before(
                            task["document_id"], task["target_version"]
                        )
                    self.registry.complete_upsert(task_id, chunk_count)
                elif task["operation"] == "delete":
                    self.vector_store.delete_by_document_id(task["document_id"])
                    self.registry.complete_delete(task_id)
                    self._archive_document_files(task["document_id"])
                else:
                    raise ValueError(f"未知任务类型: {task['operation']}")
                logger.info(
                    f"知识库任务完成: task_id={task_id}, operation={task['operation']}"
                )
            except Exception as exc:
                logger.exception(f"知识库任务失败: task_id={task_id}")
                try:
                    self.registry.fail_task(task_id, exc)
                except Exception:
                    logger.exception(f"记录知识库任务失败状态时出错: task_id={task_id}")

    def _archive_document_files(self, document_id):
        source_dir = self.storage_root / document_id
        if not source_dir.exists():
            return
        destination = self.trash_root / (
            f"{document_id}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )
        try:
            shutil.move(str(source_dir), str(destination))
        except Exception:
            # 向量和注册表删除已经完成，文件归档失败不应把任务改回 active。
            logger.exception(f"文档文件归档失败: document_id={document_id}")

    def get_document(self, document_id):
        document_id = self._validate_document_id(document_id)
        document = self.registry.get_document(document_id)
        if document is None:
            raise KnowledgeDocumentNotFoundError(document_id)
        return document

    def list_documents(self, include_deleted=False):
        return self.registry.list_documents(include_deleted=include_deleted)

    def get_task(self, task_id):
        task_id = self._validate_document_id(task_id)
        task = self.registry.get_task(task_id)
        if task is None:
            raise KnowledgeDocumentNotFoundError(f"任务不存在: {task_id}")
        return task
