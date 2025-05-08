"""
定义数据索引的Haystack Pipeline
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import (
    PreProcessor,
    TextConverter,
    PDFToTextConverter,
    MarkdownConverter,
    DocxToTextConverter,
    FileTypeClassifier
)

from config.settings import (
    INDEX_PIPELINE_NAME, 
    RAW_DOCUMENTS_DIR, 
    DOCUMENT_STORE_PATH,
    EMBEDDING_MODEL_NAME
)

def get_file_converters():
    """获取支持的文件转换器"""
    return {
        "txt": TextConverter(),
        "pdf": PDFToTextConverter(),
        "md": MarkdownConverter(),
        "docx": DocxToTextConverter(),
    }

def get_document_store(embedding_dim: int = 768) -> FAISSDocumentStore:
    """创建或加载文档存储"""
    document_store_path = Path(DOCUMENT_STORE_PATH)
    
    # 检查文档存储是否已存在
    if document_store_path.exists():
        return FAISSDocumentStore.load(str(document_store_path))
    
    # 创建新的文档存储
    return FAISSDocumentStore(
        embedding_dim=embedding_dim,
        faiss_index_factory_str="Flat",
        return_embedding=True,
        similarity="dot_product"
    )

def get_indexing_pipeline() -> Pipeline:
    """创建文档索引pipeline"""
    # 获取文档存储
    document_store = get_document_store()
    
    # 获取文件转换器
    converters = get_file_converters()
    
    # 创建文件类型分类器
    file_classifier = FileTypeClassifier()
    
    # 创建文档预处理器
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_overlap=50,
        split_respect_sentence_boundary=True,
    )
    
    # 构建索引pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(file_classifier, name="FileTypeClassifier", inputs=["File"])
    
    # 添加各种文件转换器
    for file_type, converter in converters.items():
        indexing_pipeline.add_node(
            converter, 
            name=f"{file_type.upper()}Converter",
            inputs=[f"FileTypeClassifier.{file_type}"]
        )
    
    # 添加预处理器
    converter_outputs = [f"{file_type.upper()}Converter" for file_type in converters.keys()]
    indexing_pipeline.add_node(preprocessor, name="PreProcessor", inputs=converter_outputs)
    
    # 添加文档存储
    indexing_pipeline.add_node(document_store, name="DocumentStore", inputs=["PreProcessor"])
    
    return indexing_pipeline

def index_documents(
    documents_dir: Union[str, Path] = RAW_DOCUMENTS_DIR,
    file_extensions: Optional[List[str]] = None
) -> None:
    """
    为指定目录中的文档建立索引
    
    Args:
        documents_dir: 原始文档目录路径
        file_extensions: 要处理的文件扩展名列表
    """
    if file_extensions is None:
        file_extensions = ["pdf", "txt", "md", "docx"]
        
    # 创建索引pipeline    
    indexing_pipeline = get_indexing_pipeline()
    
    # 获取要索引的文件列表
    documents_dir = Path(documents_dir)
    files_to_index = []
    
    for ext in file_extensions:
        files_to_index.extend(list(documents_dir.glob(f"**/*.{ext}")))
    
    if not files_to_index:
        print(f"没有找到任何可索引的文件在 {documents_dir}")
        return
    
    # 执行索引pipeline
    print(f"开始为 {len(files_to_index)} 个文件建立索引...")
    for file_path in files_to_index:
        try:
            indexing_pipeline.run(file_paths=[str(file_path)])
            print(f"已成功索引: {file_path.name}")
        except Exception as e:
            print(f"索引 {file_path.name} 时出错: {str(e)}")
    
    # 保存文档存储
    document_store = indexing_pipeline.get_node("DocumentStore")
    document_store.save(DOCUMENT_STORE_PATH)
    print(f"文档存储已保存到 {DOCUMENT_STORE_PATH}")

if __name__ == "__main__":
    # 当直接运行此脚本时，为默认目录中的文档建立索引
    index_documents() 