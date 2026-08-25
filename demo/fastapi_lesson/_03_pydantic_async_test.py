#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/2/4 11:18
@File    : _03_pydantic_async_test.py
@Function :
"""
import requests
import time
import asyncio
import aiohttp

start = time.time()
response1 = requests.get("http://127.0.0.1:8000/async-delay")
response2 = requests.get("http://127.0.0.1:8000/async-delay")  # 并发测试
print(response1.json(), response2.json())  # 两个响应
print("Time:", time.time() - start)  # 异步下，总时间约2秒（非4秒）

data = {"username": "test", "email": "test@example.com"}
response = requests.post("http://127.0.0.1:8000/users/", json=data)
print(response.json())


async def main():
    async def fetch(session, url):
        async with session.get(url) as resp:
            return await resp.text()

    urls = [
        "http://127.0.0.1:8000/async-delay",
        "http://127.0.0.1:8000/async-delay"
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    for r in results:
        print(r)


start = time.time()
asyncio.run(main())
print("Time:", time.time() - start)  # 异步下，总时间约2秒（非4秒）
