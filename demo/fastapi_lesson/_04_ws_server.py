#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/2/4 11:34
@File    : _04_ws_server.py
@Function :
"""
from fastapi import WebSocket
from fastapi import FastAPI
import asyncio
import uvicorn
from pydantic import BaseModel

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受连接
    for i in range(5):  # 流式发送
        await asyncio.sleep(1)
        await websocket.send_text(f"Message {i}")
    await websocket.close()  # 关闭


class Message(BaseModel):
    content: str


@app.websocket("/ws-chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        try:
            message = Message(content=data)  # 用 Pydantic 验证
            await websocket.send_text(f"Echo: {message.content}")
        except:
            await websocket.send_text("Invalid message")


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
