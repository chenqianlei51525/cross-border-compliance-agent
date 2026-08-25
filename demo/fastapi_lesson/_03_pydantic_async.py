#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/2/4 11:16
@File    : _03_pydantic_async.py
@Function :
"""

from pydantic import BaseModel
import asyncio
from fastapi import FastAPI
import uvicorn

app = FastAPI()


class User(BaseModel):
    username: str
    email: str


@app.get("/async-delay")
async def async_delay():
    await asyncio.sleep(2)  # 模拟异步等待
    return {"message": "Delayed response"}


@app.post("/users/")
async def create_user(user: User):
    await asyncio.sleep(1)  # 模拟异步数据库保存
    return {"user": user.model_dump(), "status": "created"}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
