"""
Pipeline相关的辅助函数
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    清理文本，移除多余的空格和特殊字符
    
    Args:
        text: 要清理的文本
        
    Returns:
        清理后的文本
    """
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 替换多个换行为单个换行
    text = re.sub(r'\n+', '\n', text)
    # 移除非打印字符
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    return text.strip()

def format_results(results: Dict[str, Any], max_docs: int = 3) -> Dict[str, Any]:
    """
    格式化检索结果，限制返回的文档数量
    
    Args:
        results: Haystack Pipeline的结果
        max_docs: 最大返回文档数量
        
    Returns:
        格式化后的结果
    """
    formatted_results = {}
    
    # 提取答案
    formatted_results["answer"] = results.get("results", [""])[0] if results.get("results") else ""
    
    # 提取并限制文档数量
    documents = results.get("documents", [])
    if documents and len(documents) > max_docs:
        documents = documents[:max_docs]
    
    # 清理文档内容
    for doc in documents:
        if hasattr(doc, "content"):
            doc.content = clean_text(doc.content)
    
    formatted_results["documents"] = documents
    
    return formatted_results

def get_document_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件路径获取文档元数据
    
    Args:
        file_path: 文档文件路径
        
    Returns:
        包含文档元数据的字典
    """
    file_path = Path(file_path)
    
    return {
        "source": file_path.name,
        "file_path": str(file_path),
        "file_type": file_path.suffix.lstrip('.'),
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "creation_date": file_path.stat().st_ctime if file_path.exists() else None
    } 