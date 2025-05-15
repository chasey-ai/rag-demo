"""
定义数据索引的Haystack Pipeline
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import (
    PreProcessor,
    TextConverter,
    PDFToTextConverter,
    MarkdownConverter,
    DocxToTextConverter,
    FileTypeClassifier,
    EmbeddingRetriever
)
from tqdm import tqdm
import logging
import time

from config.settings import (
    INDEX_PIPELINE_NAME, 
    RAW_DOCUMENTS_DIR, 
    DOCUMENT_STORE_PATH,
    EMBEDDING_MODEL_NAME
)

# 配置日志
logger = logging.getLogger(__name__)

def get_file_converters() -> Dict[str, Any]:
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
        logger.info(f"加载已存在的文档存储: {document_store_path}")
        return FAISSDocumentStore.load(str(document_store_path))
    
    # 创建新的文档存储
    logger.info(f"创建新的文档存储，维度: {embedding_dim}")
    return FAISSDocumentStore(
        embedding_dim=embedding_dim,
        faiss_index_factory_str="Flat",
        return_embedding=True,
        similarity="dot_product"
    )

def get_preprocessor() -> PreProcessor:
    """获取预处理器配置"""
    return PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_overlap=50,
        split_respect_sentence_boundary=True,
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
    preprocessor = get_preprocessor()
    
    # 创建嵌入检索器
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=EMBEDDING_MODEL_NAME
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
    
    # 添加检索器用于更新嵌入
    indexing_pipeline.add_node(retriever, name="Retriever", inputs=["DocumentStore"])
    
    return indexing_pipeline

def index_documents(
    documents_dir: Union[str, Path] = RAW_DOCUMENTS_DIR,
    file_extensions: Optional[List[str]] = None,
    batch_size: int = 10,
    show_progress: bool = True
) -> None:
    """
    为指定目录中的文档建立索引
    
    Args:
        documents_dir: 原始文档目录路径
        file_extensions: 要处理的文件扩展名列表
        batch_size: 批处理文件数量
        show_progress: 是否显示进度条
    """
    start_time = time.time()
    
    if file_extensions is None:
        file_extensions = ["pdf", "txt", "md", "docx"]
        
    # 创建索引pipeline    
    indexing_pipeline = get_indexing_pipeline()
    
    # 获取要索引的文件列表
    documents_dir = Path(documents_dir)
    files_to_index = []
    
    logger.info(f"正在扫描目录 {documents_dir} 查找文件...")
    for ext in file_extensions:
        ext_files = list(documents_dir.glob(f"**/*.{ext}"))
        files_to_index.extend(ext_files)
        logger.info(f"找到 {len(ext_files)} 个 {ext} 文件")
    
    if not files_to_index:
        logger.warning(f"没有找到任何可索引的文件在 {documents_dir}")
        return
    
    # 按文件大小排序（小文件先处理）
    logger.info("按文件大小排序...")
    files_to_index.sort(key=lambda x: x.stat().st_size)
    
    total_size_mb = sum(f.stat().st_size for f in files_to_index) / (1024 * 1024)
    logger.info(f"开始为 {len(files_to_index)} 个文件建立索引 (总大小: {total_size_mb:.2f} MB)...")
    
    # 批处理索引文件
    failed_files = []
    processed_files = 0
    processed_size_mb = 0
    
    # 使用tqdm创建进度条
    iterable = tqdm(range(0, len(files_to_index), batch_size), desc="索引批次") if show_progress else range(0, len(files_to_index), batch_size)
    
    for i in iterable:
        batch = files_to_index[i:i+batch_size]
        batch_size_mb = sum(f.stat().st_size for f in batch) / (1024 * 1024)
        
        batch_start_time = time.time()
        try:
            # 处理一批文件
            batch_paths = [str(file_path) for file_path in batch]
            indexing_pipeline.run(file_paths=batch_paths)
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            processed_files += len(batch)
            processed_size_mb += batch_size_mb
            
            progress_pct = (processed_files / len(files_to_index)) * 100
            logger.info(f"批次 {i//batch_size + 1}/{(len(files_to_index)-1)//batch_size + 1} 成功: "
                       f"{len(batch)} 文件, {batch_size_mb:.2f} MB, "
                       f"{batch_duration:.2f} 秒 "
                       f"({progress_pct:.1f}% 完成)")
            
        except Exception as e:
            logger.error(f"索引批次 {i//batch_size + 1} 时出错: {str(e)}")
            # 如果批处理失败，尝试单个文件处理
            logger.info("尝试单个文件处理...")
            for file_path in batch:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                file_start_time = time.time()
                
                try:
                    indexing_pipeline.run(file_paths=[str(file_path)])
                    
                    file_duration = time.time() - file_start_time
                    processed_files += 1
                    processed_size_mb += file_size_mb
                    
                    logger.info(f"已成功索引: {file_path.name} ({file_size_mb:.2f} MB, {file_duration:.2f} 秒)")
                except Exception as e:
                    logger.error(f"索引 {file_path.name} 时出错: {str(e)}")
                    failed_files.append(file_path.name)
    
    # 保存文档存储
    logger.info("索引完成，正在保存文档存储...")
    document_store = indexing_pipeline.get_node("DocumentStore")
    document_store.save(DOCUMENT_STORE_PATH)
    
    total_duration = time.time() - start_time
    
    # 打印汇总信息
    logger.info(f"索引任务完成:")
    logger.info(f"- 总处理时间: {total_duration:.2f} 秒")
    logger.info(f"- 处理文件数: {processed_files}/{len(files_to_index)}")
    logger.info(f"- 处理数据量: {processed_size_mb:.2f} MB")
    logger.info(f"- 处理速度: {processed_size_mb/total_duration:.2f} MB/秒")
    
    if failed_files:
        logger.warning(f"以下 {len(failed_files)} 个文件索引失败: {', '.join(failed_files)}")
    
    logger.info(f"文档存储已保存到 {DOCUMENT_STORE_PATH}")
    doc_count = document_store.get_document_count()
    logger.info(f"文档存储中共有 {doc_count} 个文档片段")

def update_document_embeddings(show_progress: bool = True):
    """
    更新文档存储中的所有嵌入
    
    Args:
        show_progress: 是否显示进度条
    """
    start_time = time.time()
    
    document_store = get_document_store()
    
    # 确认文档存储中有文档
    document_count = document_store.get_document_count()
    if document_count == 0:
        logger.warning("文档存储为空，没有可更新的嵌入")
        return
    
    logger.info(f"开始更新 {document_count} 个文档的嵌入...")
    
    # 创建检索器
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=EMBEDDING_MODEL_NAME
    )
    
    # 从文档存储获取所有文档并更新嵌入
    documents = document_store.get_all_documents()
    
    # 更新嵌入时显示进度
    document_store.update_embeddings(
        retriever, 
        documents,
        update_existing_embeddings=True,
        batch_size=32,
        show_progress=show_progress
    )
    
    # 保存更新的文档存储
    document_store.save(DOCUMENT_STORE_PATH)
    
    total_duration = time.time() - start_time
    logger.info(f"嵌入更新完成:")
    logger.info(f"- 总处理时间: {total_duration:.2f} 秒")
    logger.info(f"- 处理文档数: {document_count}")
    logger.info(f"- 处理速度: {document_count/total_duration:.2f} 文档/秒")
    logger.info(f"文档存储已保存到 {DOCUMENT_STORE_PATH}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加命令行接口
    import argparse
    
    parser = argparse.ArgumentParser(description="为本地文件夹中的文档建立索引")
    parser.add_argument("--dir", "-d", type=str, default=RAW_DOCUMENTS_DIR,
                      help=f"要索引的文档目录 (默认: {RAW_DOCUMENTS_DIR})")
    parser.add_argument("--types", "-t", nargs="+", default=["pdf", "txt", "md", "docx"],
                      help="要索引的文件类型，例如: -t pdf txt md (默认: pdf txt md docx)")
    parser.add_argument("--batch", "-b", type=int, default=10,
                      help="批处理文件数量 (默认: 10)")
    parser.add_argument("--update-embeddings", "-u", action="store_true",
                      help="是否仅更新已索引文档的嵌入")
    parser.add_argument("--no-progress", action="store_true",
                      help="不显示进度条")
    
    args = parser.parse_args()
    show_progress = not args.no_progress
    
    if args.update_embeddings:
        logger.info("仅更新文档嵌入...")
        update_document_embeddings(show_progress=show_progress)
    else:
        logger.info(f"开始为目录 {args.dir} 中的 {args.types} 文件建立索引...")
        index_documents(
            documents_dir=args.dir,
            file_extensions=args.types,
            batch_size=args.batch,
            show_progress=show_progress
        ) 