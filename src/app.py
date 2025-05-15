"""
RAG 应用主入口
"""

import os
import logging
import chainlit as cl
from typing import Dict, Any

from config.settings import (
    APP_NAME,
    APP_DESCRIPTION,
    RAW_DOCUMENTS_DIR,
    DOCUMENT_STORE_PATH
)
from pipelines.file_sync_pipeline import DocumentSyncPipeline
from pipelines.query import process_query, get_query_pipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 初始化查询pipeline
query_pipeline = None

@cl.on_chat_start
async def on_chat_start():
    """当用户开始聊天时执行"""
    global query_pipeline
    
    # 显示欢迎消息
    await cl.Message(
        content=f"欢迎使用 {APP_NAME}！我将帮助您查询文档。",
        disable_feedback=True
    ).send()
    
    # 检查文档存储是否存在
    if not os.path.exists(DOCUMENT_STORE_PATH):
        # 自动同步文档
        sync_files_msg = cl.Message(content="正在首次同步文档，请稍候...")
        await sync_files_msg.send()
        
        # 运行文档同步
        try:
            sync_pipeline = DocumentSyncPipeline()
            stats = sync_pipeline.synchronize()
            
            # 更新消息
            stats_text = (
                f"文档同步完成：\n"
                f"- 新增: {stats['added']} 个文件\n"
                f"- 更新: {stats['updated']} 个文件\n"
                f"- 删除: {stats['deleted']} 个文件\n"
                f"现在您可以开始提问了！"
            )
            await sync_files_msg.update(content=stats_text)
            
        except Exception as e:
            logger.error(f"文档同步失败: {str(e)}")
            await sync_files_msg.update(content=f"文档同步失败: {str(e)}")
            return
    
    # 初始化查询pipeline
    query_pipeline = get_query_pipeline()
    
    # 添加上传文件的功能
    await cl.Message(
        content="您可以上传新文档，系统会自动将其编入索引。",
        disable_feedback=True
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户消息"""
    global query_pipeline
    
    # 如果用户上传了文件
    if message.elements:
        files_msg = cl.Message(content="正在处理您上传的文件...")
        await files_msg.send()
        
        uploaded_files = []
        for element in message.elements:
            if isinstance(element, cl.File):
                # 保存文件
                file_path = os.path.join(RAW_DOCUMENTS_DIR, element.name)
                os.makedirs(RAW_DOCUMENTS_DIR, exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(await element.content())
                
                uploaded_files.append(element.name)
        
        if uploaded_files:
            # 同步新上传的文件
            try:
                sync_pipeline = DocumentSyncPipeline()
                stats = sync_pipeline.synchronize()
                
                # 更新查询pipeline以包含新文档
                query_pipeline = get_query_pipeline()
                
                await files_msg.update(
                    content=f"文件上传成功: {', '.join(uploaded_files)}，已将其编入索引。"
                )
            except Exception as e:
                logger.error(f"处理上传文件失败: {str(e)}")
                await files_msg.update(content=f"处理上传文件失败: {str(e)}")
        return
    
    # 处理用户查询
    question = message.content
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    # 显示思考中状态
    await response_msg.update(content="正在查询相关文档...", disable_feedback=True)
    
    try:
        # 使用RAG pipeline处理查询
        result = process_query(question, query_pipeline)
        answer = result["answer"]
        documents = result.get("documents", [])
        
        # 添加源文档引用
        if documents:
            answer += "\n\n**参考文档:**"
            for i, doc in enumerate(documents[:3], 1):  # 最多显示3个文档
                source = doc.meta.get("source", "未知来源")
                if isinstance(source, str) and os.path.exists(source):
                    source = os.path.basename(source)
                answer += f"\n{i}. {source}"
        
        # 更新消息
        await response_msg.update(content=answer)
        
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}")
        await response_msg.update(content=f"抱歉，处理您的查询时出错: {str(e)}")


@cl.action_callback("sync_documents")
async def on_sync_action(action):
    """处理同步文档的动作"""
    sync_msg = cl.Message(content="正在同步文档...")
    await sync_msg.send()
    
    try:
        sync_pipeline = DocumentSyncPipeline()
        stats = sync_pipeline.synchronize()
        
        # 更新全局查询pipeline
        global query_pipeline
        query_pipeline = get_query_pipeline()
        
        # 更新消息
        stats_text = (
            f"文档同步完成：\n"
            f"- 新增: {stats['added']} 个文件\n"
            f"- 更新: {stats['updated']} 个文件\n"
            f"- 删除: {stats['deleted']} 个文件"
        )
        await sync_msg.update(content=stats_text)
        
    except Exception as e:
        logger.error(f"文档同步失败: {str(e)}")
        await sync_msg.update(content=f"文档同步失败: {str(e)}")


if __name__ == "__main__":
    # Chainlit会自动运行此文件
    pass 