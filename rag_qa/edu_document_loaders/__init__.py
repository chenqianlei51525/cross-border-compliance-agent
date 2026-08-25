"""按需加载 OCR 文档解析器，避免未使用的格式阻塞应用启动。"""

from importlib import import_module


_LOADERS = {
    "OCRDOCLoader": ("rag_qa.edu_document_loaders.edu_docloader", "OCRDOCLoader"),
    "OCRPPTLoader": ("rag_qa.edu_document_loaders.edu_pptloader", "OCRPPTLoader"),
    "OCRIMGLoader": ("rag_qa.edu_document_loaders.edu_imgloader", "OCRIMGLoader"),
    "OCRPDFLoader": ("rag_qa.edu_document_loaders.edu_pdfloader", "OCRPDFLoader"),
}
__all__ = list(_LOADERS)


def __getattr__(name):
    if name not in _LOADERS:
        raise AttributeError(name)
    module_name, class_name = _LOADERS[name]
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value
