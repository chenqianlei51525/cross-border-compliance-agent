#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2025/10/31 11:19
@File    : demo_markdown.py
@Function :
"""
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

mk_file_path = "/rag_qa/data/人工智能就业课课程大纲.md"

loader = UnstructuredMarkdownLoader(file_path=mk_file_path)
doc = loader.load()
print(doc)
print(doc[0].page_content)
print(doc[0].metadata)
