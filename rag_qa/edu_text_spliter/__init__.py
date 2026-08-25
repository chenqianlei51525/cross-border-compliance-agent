"""按需加载文本切分器，减少应用启动时的模型依赖。"""

from importlib import import_module


_SPLITTERS = {
    "AliTextSplitter": (
        "rag_qa.edu_text_spliter.edu_model_text_spliter",
        "AliTextSplitter",
    ),
    "ChineseRecursiveTextSplitter": (
        "rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter",
        "ChineseRecursiveTextSplitter",
    ),
}
__all__ = list(_SPLITTERS)


def __getattr__(name):
    if name not in _SPLITTERS:
        raise AttributeError(name)
    module_name, class_name = _SPLITTERS[name]
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value
