from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
import os
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional
import time
import re

# 导入现有的系统
from new_main import IntegratedQASystem
from rag_qa.core.knowledge_base_service import (
    KnowledgeBaseService,
    KnowledgeDocumentConflictError,
    KnowledgeDocumentNotFoundError,
)
# 导入新的跨境合规 Agent 系统
from compliance.system import get_system as get_compliance_system

# 创建应用实例
app = FastAPI(
    title="跨境电商合规智能问答系统 API",
    description=(
        "Agentic RAG 跨境电商合规问答：CE/FCC/RoHS/UN38.3/PSE/KCC。"
        "RAG 作为 Agent 的工具组件,LLM 自主决策检索策略(混合检索 / 知识图谱 /"
        " Query Rewrite / HyDE / 反思循环 / Web 抓取)。"
    ),
)

# 配置CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
os.makedirs("static", exist_ok=True)

# 创建全局QA系统实例
qa_system = IntegratedQASystem()
knowledge_service = KnowledgeBaseService(
    vector_store=qa_system.vector_store,
    config=qa_system.config,
)

# 定义日常问候用语模式和回复（跨境合规智能问答应小合规）
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是『应小合规』，跨境电商合规智能助手（CE/FCC/RoHS/UN38.3/PSE/KCC），很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是『应小合规』，专注跨境电商合规：CE-RED、FCC Part 15、RoHS、UN38.3、PSE、KCC 等认证领域。需要查认证？直接问我。"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "在的！我是『应小合规』，随时为你解答合规问题。"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我在守候认证资料库。需要查产品出口某国的合规要求？直接发品类和国家。"
    }
]


# 定义请求模型
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


# 定义响应模型
class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float


# 添加静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


# 根路径重定向到index.html
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


# 创建新会话
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# 查询历史消息
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = qa_system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 清除历史消息
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


# 入参 出参
# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 判断是否需要流式处理（基于 need_rag）
    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        # 需要 RAG 和 LLM 处理，返回流式响应提示
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 非流式查询，直接返回 BM25 检索的答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# 流式查询WebSocket接口
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            request_data = json.loads(data)

            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))

            start_time = time.time()

            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:  # 确保客户端已连接
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })

            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break

            # 调用QA系统进行查询，流式返回结果
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token

                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break

                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })

                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break

                await asyncio.sleep(0.01)

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ==================== 跨境合规 Agent 新接口 ====================

@app.get("/api/agent/tools")
async def list_agent_tools():
    """列出 Agent 可调用的所有工具（用于调试和前端展示）。"""
    system = get_compliance_system()
    return {
        "tools": [
            {"name": spec.name, "description": spec.description,
             "schema": spec.schema}
            for spec in system.tool_registry._tools.values()  # noqa: SLF001
        ]
    }


@app.post("/api/agent/chat")
async def agent_chat(request: QueryRequest):
    """同步 Agent 问答。"""
    system = get_compliance_system()
    result = system.ask(
        question=request.query,
        source_filter=request.source_filter,
        use_agent=True,
        session_id=request.session_id,
    )
    return result


@app.post("/api/agent/stream")
async def agent_stream(request: QueryRequest):
    """流式 Agent 输出（SSE）。

    事件类型：trace_start / step / trace / final / error
    """
    from fastapi.responses import StreamingResponse
    import json as _json

    system = get_compliance_system()

    def event_gen():
        for ev in system.stream_agent(
            question=request.query,
            session_id=request.session_id,
        ):
            yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/kg/build")
async def kg_build_from_text(text: str = Form(...), source: str = Form(...)):
    """从一段法规文本里抽三元组并写入 Neo4j（演示用）。"""
    import asyncio
    system = get_compliance_system()
    from kg import extract_triples_with_llm
    triples = extract_triples_with_llm(text, llm=system._llm_callable)  # noqa: SLF001
    n = system.kg_client.upsert_triples(triples)
    return {"status": "ok", "triples_count": n, "samples": [t.to_dict() for t in triples[:5]]}


@app.get("/api/kg/reason")
async def kg_reason(question: str = Query(...), hops: int = Query(2)):
    """直接做多跳图谱推理，便于单独测试 KG 通路。"""
    system = get_compliance_system()
    triples = system.kg_client.multi_hop_reasoning(question, hops=hops)
    return {"hop_results": triples}


@app.post("/api/parser/parse")
async def parse_upload(
    file: UploadFile = File(...),
):
    """上传 PDF/MD/HTML/DOCX/TXT 解析为 MinerU-style JSON。"""
    from rag_qa.mineru import MinerUStyleParser
    content = await file.read()
    parser = MinerUStyleParser()
    parsed = parser.parse_bytes(content, file.filename or "input.bin")
    return parsed


# 获取有效的学科类别
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


async def _read_upload(upload_file: UploadFile):
    """分段读取上传文件，并在进入后台任务前执行大小限制。"""
    content = bytearray()
    try:
        while chunk := await upload_file.read(1024 * 1024):
            content.extend(chunk)
            if len(content) > knowledge_service.max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        "文件超过大小限制 "
                        f"{knowledge_service.max_upload_bytes // 1024 // 1024} MB"
                    ),
                )
        return bytes(content)
    finally:
        await upload_file.close()


def _raise_knowledge_http_error(exc):
    if isinstance(exc, KnowledgeDocumentNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, KnowledgeDocumentConflictError):
        raise HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=f"知识库操作失败: {exc}")


@app.post(
    "/api/knowledge/documents",
    status_code=status.HTTP_202_ACCEPTED,
    summary="新增知识库文档",
)
async def create_knowledge_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source: str = Form(...),
):
    content = await _read_upload(file)
    try:
        task = knowledge_service.submit_create(file.filename, content, source)
        background_tasks.add_task(knowledge_service.process_task, task["task_id"])
        return {
            "message": "文档已接收，正在后台解析并写入知识库",
            "document_id": task["document_id"],
            "task_id": task["task_id"],
            "task_status": task["status"],
            "task_url": f"/api/knowledge/tasks/{task['task_id']}",
        }
    except Exception as exc:
        _raise_knowledge_http_error(exc)


@app.put(
    "/api/knowledge/documents/{document_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="修改并重建知识库文档",
)
async def update_knowledge_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    source: Optional[str] = Form(None),
):
    content = await _read_upload(file) if file is not None else None
    try:
        task = knowledge_service.submit_update(
            document_id=document_id,
            file_name=file.filename if file is not None else None,
            content=content,
            source=source,
        )
        background_tasks.add_task(knowledge_service.process_task, task["task_id"])
        return {
            "message": "文档修改已接收，正在后台构建新版本索引",
            "document_id": task["document_id"],
            "target_version": task["target_version"],
            "task_id": task["task_id"],
            "task_status": task["status"],
            "task_url": f"/api/knowledge/tasks/{task['task_id']}",
        }
    except Exception as exc:
        _raise_knowledge_http_error(exc)


@app.delete(
    "/api/knowledge/documents/{document_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="删除知识库文档",
)
async def delete_knowledge_document(
    document_id: str,
    background_tasks: BackgroundTasks,
):
    try:
        task = knowledge_service.submit_delete(document_id)
        background_tasks.add_task(knowledge_service.process_task, task["task_id"])
        return {
            "message": "文档删除任务已提交",
            "document_id": task["document_id"],
            "task_id": task["task_id"],
            "task_status": task["status"],
            "task_url": f"/api/knowledge/tasks/{task['task_id']}",
        }
    except Exception as exc:
        _raise_knowledge_http_error(exc)


@app.get("/api/knowledge/documents", summary="查询知识库文档列表")
async def list_knowledge_documents(include_deleted: bool = Query(False)):
    try:
        documents = knowledge_service.list_documents(include_deleted=include_deleted)
        return {"total": len(documents), "documents": documents}
    except Exception as exc:
        _raise_knowledge_http_error(exc)


@app.get("/api/knowledge/system", summary="查询知识库与 Milvus 连接信息")
async def get_knowledge_system():
    try:
        return {
            "milvus": qa_system.vector_store.get_collection_info(),
            "supported_extensions": knowledge_service.supported_extensions,
            "max_upload_bytes": knowledge_service.max_upload_bytes,
            "attu_url": "http://127.0.0.1:3000",
        }
    except Exception as exc:
        _raise_knowledge_http_error(exc)


@app.get("/api/knowledge/documents/{document_id}", summary="查询知识库文档详情")
async def get_knowledge_document(document_id: str):
    try:
        return knowledge_service.get_document(document_id)
    except Exception as exc:
        _raise_knowledge_http_error(exc)


@app.get("/api/knowledge/tasks/{task_id}", summary="查询知识库更新任务状态")
async def get_knowledge_task(task_id: str):
    try:
        return knowledge_service.get_task(task_id)
    except Exception as exc:
        _raise_knowledge_http_error(exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=False)
