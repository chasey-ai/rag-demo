import os
from typing import List

import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage

from pipelines.query import get_query_pipeline

# 初始化查询管道
query_pipeline = get_query_pipeline()

@cl.on_chat_start
async def on_chat_start():
    """当新的聊天会话开始时执行的函数"""
    # 发送欢迎消息
    await cl.Message(
        content="👋 欢迎使用基于RAG的问答系统！请输入您的问题，我将尽力回答。"
    ).send()
    
    # 设置用户会话状态
    cl.user_session.set("query_pipeline", query_pipeline)

@cl.on_message
async def on_message(message: cl.Message):
    """处理用户发送的每条消息"""
    # 获取用户问题
    query = message.content
    
    # 获取查询管道
    pipeline = cl.user_session.get("query_pipeline")
    
    # 显示思考状态
    thinking_msg = cl.Message(content="正在思考...")
    await thinking_msg.send()
    
    try:
        # 执行RAG查询
        result = pipeline.run(query=query)
        
        # 提取结果
        answer = result.get("answer", "抱歉，我无法回答这个问题。")
        documents = result.get("documents", [])
        
        # 更新回答内容
        await thinking_msg.update(content=answer)
        
        # 如果有检索到的文档，添加为参考文献
        if documents:
            elements = []
            for i, doc in enumerate(documents[:3]):  # 仅显示前3个结果
                source_name = doc.meta.get("source", f"文档_{i+1}")
                elements.append(
                    cl.Text(name=source_name, content=doc.content[:500] + "...")
                )
            
            if elements:
                await cl.Message(
                    content="以下是我用来回答的参考资料:",
                    elements=elements
                ).send()
                
    except Exception as e:
        # 发生错误时通知用户
        await thinking_msg.update(content=f"处理您的问题时出错: {str(e)}") 