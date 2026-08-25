#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/2/4 11:36
@File    : _04_ws_client.py
@Function :
"""
import asyncio
import websockets


async def test_ws():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        messages = []
        for _ in range(5):
            msg = await ws.recv()
            messages.append(msg)
        print(messages)  # 预期: ['Message 0', 'Message 1', ...]


async def test_ws_chat():
    async with websockets.connect("ws://127.0.0.1:8000/ws-chat") as ws:
        messages = []
        for _ in range(5):
            msg = await ws.recv()
            messages.append(msg)
        print(messages)  # 预期: ['Message 0', 'Message 1', ...]


asyncio.run(test_ws())
