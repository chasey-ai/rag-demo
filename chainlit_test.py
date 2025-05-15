import chainlit as cl
from rag_demo import rag_pipeline, init_document_store
from haystack import Pipeline
import json
import os

# 全局变量用于保存文档存储的引用和初始化状态
document_store = None
is_initialized = False

# 初始化文档存储
@cl.on_chat_start
async def on_chat_start():
    global document_store, is_initialized
    
    # 避免重复初始化
    if is_initialized:
        await cl.Message(content="已连接到文档库").send()
        await cl.Message(content="欢迎使用基于Haystack的RAG问答系统，请输入您的问题").send()
        return
        
    # 在聊天开始时初始化
    await cl.Message(content="正在初始化文档库，请稍候...").send()
    
    # 检查files目录是否存在
    if not os.path.exists("files"):
        os.makedirs("files")
        await cl.Message(content="警告: 'files'目录不存在，已创建空目录。请将文本文件放入该目录后重启应用。").send()
        return
    
    # 初始化文档存储
    try:
        document_store = init_document_store()
        files_count = len([f for f in os.listdir("files") if f.endswith('.txt')])
        docs_count = document_store.count_documents()
        
        await cl.Message(content=f"文档库初始化完成! 已索引{files_count}个文本文件，包含{docs_count}个文档片段。").send()
        is_initialized = True
    except Exception as e:
        await cl.Message(content=f"初始化文档库时出错: {str(e)}").send()
        return
    
    await cl.Message(content="欢迎使用基于Haystack的RAG问答系统，请输入您的问题").send()

@cl.on_message
async def main(message: cl.Message):
    # 获取用户查询
    user_query = message.content
    
    # 创建消息元素以显示进度
    msg = cl.Message(content="")
    await msg.send()

    try:
        # 开始处理用户查询
        await msg.stream_token("正在分析您的问题...")
        
        # 使用单次调用执行完整的RAG流程，参考test.py中的ask_question实现
        results = rag_pipeline.run(
            {
                "query_decomposer": {"query": user_query},
                "text_embedder": {"text": user_query},
                "prompt_builder": {"question": user_query},
            },
            include_outputs_from={"query_decomposer", "text_embedder", "multi_query_retriever", "prompt_builder", "llm"}
        )
        
        # 从结果中提取分解的子查询
        sub_queries = results["query_decomposer"]["sub_queries"]
        sub_queries_md = "### 问题分解:\n" + "\n".join([f"- {q}" for q in sub_queries])
        await msg.stream_token(f"\n\n{sub_queries_md}\n\n")
        
        # 展示检索到的文档
        await msg.stream_token("正在检索相关文档...\n\n")
        retrieved_docs = results["multi_query_retriever"]["documents"]
        
        if not retrieved_docs:
            await msg.stream_token("没有找到相关文档。请尝试调整您的问题，或者确认文档库中有相关内容。")
            return
            
        docs_md = "### 检索到的文档:\n"
        for i, doc in enumerate(retrieved_docs, 1):
            docs_md += f"**文档 {i}**:\n{doc.content}\n\n"
        await msg.stream_token(f"{docs_md}\n\n")
        
        # 生成并展示回答
        await msg.stream_token("正在生成回答...\n\n")
        answer = results["llm"]["replies"][0].text
        await msg.stream_token(f"### 回答:\n{answer}")
        
    except Exception as e:
        await msg.stream_token(f"\n\n处理您的问题时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 启动Chainlit应用
    cl.run()