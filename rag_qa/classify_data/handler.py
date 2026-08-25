#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2025/11/12 21:23
@File    : handler.py
@Function :
"""
import pandas as pd

data = pd.read_json("training_data_1000.json", lines=False)
print(data.head())
data.to_json("training_data_1000.json", orient="records", lines=True, index=None, force_ascii=False)
