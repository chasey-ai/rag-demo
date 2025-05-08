"""
应用配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """配置类，包含所有应用设置"""
    
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent.absolute()
    
    # 数据目录
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DOCUMENTS_DIR = os.path.join(DATA_DIR, "raw_documents")
    DOCUMENT_STORE_PATH = os.path.join(DATA_DIR, "document_store.faiss")
    
    # Pipeline设置
    INDEX_PIPELINE_NAME = "indexing_pipeline"
    QUERY_PIPELINE_NAME = "query_pipeline"
    
    # 检索器设置
    TOP_K_RETRIEVER = 5
    TOP_K_READER = 3
    
    # 嵌入模型设置
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM设置
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # 可选值：openai, anthropic
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    
    # 应用设置
    APP_NAME = "RAG Demo"
    APP_DESCRIPTION = "使用Haystack和Chainlit构建的RAG应用"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # 日志设置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE = os.path.join(BASE_DIR, "app.log")
    
    @classmethod
    def init(cls):
        """初始化设置，确保必要的目录存在"""
        os.makedirs(cls.RAW_DOCUMENTS_DIR, exist_ok=True)


# 初始化设置
Settings.init()