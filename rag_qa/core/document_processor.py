# 这个脚本讲义的代码架构图没有体现，需要进行补充
import os
from importlib import import_module
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader  # NLTK
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
# ========================================
# import sys
# # 获取当前文件所在目录的绝对路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # print(f'current_dir--》{current_dir}')
# # 获取core文件所在的目录的绝对路径
# rag_qa_path = os.path.dirname(current_dir)
# # print(f'rag_qa_path--》{rag_qa_path}')
# sys.path.insert(0, rag_qa_path)
# # 获取根目录文件所在的绝对位置
# project_root = os.path.dirname(rag_qa_path)
# sys.path.insert(0, project_root)
# ========================================
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
from base import logger, Config

conf = Config()


def _lazy_loader(module_name, class_name):
    """创建延迟导入代理，只在真正解析该文件类型时加载 OCR 依赖。"""
    class LazyLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def load(self):
            loader_class = getattr(import_module(module_name), class_name)
            return loader_class(*self.args, **self.kwargs).load()

    return LazyLoader


OCRPDFLoader = _lazy_loader(
    "rag_qa.edu_document_loaders.edu_pdfloader", "OCRPDFLoader"
)
OCRDOCLoader = _lazy_loader(
    "rag_qa.edu_document_loaders.edu_docloader", "OCRDOCLoader"
)
OCRPPTLoader = _lazy_loader(
    "rag_qa.edu_document_loaders.edu_pptloader", "OCRPPTLoader"
)
OCRIMGLoader = _lazy_loader(
    "rag_qa.edu_document_loaders.edu_imgloader", "OCRIMGLoader"
)
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


def load_document(file_path, source):
    """加载单个文档，并补齐知识库更新所需的统一元数据。"""
    file_path = os.path.abspath(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_class = document_loaders.get(file_extension)
    if loader_class is None:
        raise ValueError(
            f"不支持的文件类型 {file_extension}，支持类型: {', '.join(sorted(document_loaders))}"
        )

    if file_extension == ".txt":
        loader = loader_class(file_path, encoding="utf-8")
    else:
        loader = loader_class(file_path)

    loaded_docs = loader.load()
    timestamp = datetime.now().isoformat()
    for doc in loaded_docs:
        doc.metadata["source"] = source
        doc.metadata["file_path"] = file_path
        doc.metadata["file_name"] = os.path.basename(file_path)
        doc.metadata["timestamp"] = timestamp

    if not loaded_docs:
        raise ValueError(f"文档未解析出任何内容: {file_path}")
    logger.info(f"成功加载文件: {file_path}")
    return loaded_docs


# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()
    # print(f'supported_extensions--》{supported_extensions}')
    # 从目录名提取学科类别（如 "ai_data" -> "ai"）
    # print(f'1---》{os.path.basename(directory_path)}')
    source = os.path.basename(directory_path).replace("_data", "")
    # print(f'source-->{source}')
    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        # print(f'root---》{root}')
        # print(f'files---》{files}')
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # print(f'file_path--》{file_path}')
            # print(os.path.splitext(file_path))
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # print(f'file_extension--》{file_extension}')
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常
                try:
                    loaded_docs = load_document(file_path, source)
                    documents.extend(loaded_docs)
                except Exception as e:
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents


def _split_documents(documents, parent_chunk_size, child_chunk_size, chunk_overlap,
                     document_id=None, document_version=None, content_hash=None):
    """对已加载文档执行 Parent-Child 切分，并生成稳定的分块 ID。"""
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )

    child_chunks = []
    for i, doc in enumerate(documents):
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        is_markdown = file_extension == ".md"
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(
            f"处理文档: {doc.metadata['file_path']}, "
            f"使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}"
        )

        parent_docs = parent_splitter_to_use.split_documents([doc])
        for j, parent_doc in enumerate(parent_docs):
            if document_id:
                # 修改文档时继续使用相同的 ID，使 Milvus upsert 能覆盖原分块。
                parent_id = f"{document_id}:p{j}"
            else:
                parent_id = f"doc_{i}_parent_{j}"

            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = (
                    f"{parent_id}:c{k}" if document_id else f"{parent_id}_child_{k}"
                )
                sub_chunk.metadata["chunk_index"] = len(child_chunks)
                if document_id:
                    sub_chunk.metadata["document_id"] = document_id
                    sub_chunk.metadata["document_version"] = document_version
                    sub_chunk.metadata["content_hash"] = content_hash
                child_chunks.append(sub_chunk)

    logger.info(f"子块数量: {len(child_chunks)}")
    return child_chunks


def process_document(file_path, source, document_id, document_version,
                     content_hash=None,
                     parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    """解析并切分一个可增量更新的知识库文档。"""
    documents = load_document(file_path, source)
    chunks = _split_documents(
        documents=documents,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
        document_id=document_id,
        document_version=document_version,
        content_hash=content_hash,
    )
    if not chunks:
        raise ValueError(f"文档未切分出有效文本块: {file_path}")
    return chunks


# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    # 从指定目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    return _split_documents(
        documents=documents,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == '__main__':
    directory_path = '/Users/chan/projects/Itcast_qa_system/rag_qa/data/ai_data'
    # documents = load_documents_from_directory(directory_path)
    # print(documents)
    child_chunks = process_documents(directory_path)
    print(f'child_chunks--》{child_chunks[0]}')
