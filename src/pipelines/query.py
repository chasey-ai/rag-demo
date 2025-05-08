"""
定义查询和生成的Haystack Pipeline
"""

from typing import Dict, Any, List, Optional

from haystack import Pipeline
from haystack.nodes import (
    BM25Retriever,
    EmbeddingRetriever,
    DensePassageRetriever,
    FARMReader,
    PromptNode,
    PromptTemplate,
    JoinDocuments
)

from .index import get_document_store
from .llm import get_llm_client
from config.settings import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, TOP_K_RETRIEVER, TOP_K_READER

def get_query_pipeline() -> Pipeline:
    """
    创建并返回用于查询的Haystack Pipeline
    """
    # 获取文档存储
    document_store = get_document_store()
    
    # 创建检索器
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=EMBEDDING_MODEL_NAME,
        top_k=TOP_K_RETRIEVER
    )
    
    # 获取LLM客户端
    llm_client = get_llm_client()
    
    # 创建Prompt模板
    rag_prompt_template = PromptTemplate(
        prompt="""
        基于以下文档，回答用户问题。如果文档中不包含问题的答案，请直接说明你不知道，不要捏造信息。
        
        文档：
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        
        问题：{{ query }}
        
        请用中文回答：
        """,
        output_parser=None
    )
    
    # 创建Prompt节点
    prompt_node = PromptNode(
        model_name_or_path=LLM_MODEL_NAME,
        default_prompt_template=rag_prompt_template,
        api_key=llm_client.api_key,
        max_length=500
    )
    
    # 创建文档合并节点
    join_documents = JoinDocuments(join_mode="concatenate", separator="\n\n")
    
    # 构建查询pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["Retriever"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["JoinDocuments"])
    
    return query_pipeline

def process_query(query: str, pipeline: Optional[Pipeline] = None) -> Dict[str, Any]:
    """
    处理用户查询并返回回答和相关文档
    
    Args:
        query: 用户查询
        pipeline: 可选的查询pipeline实例
        
    Returns:
        包含回答和相关文档的字典
    """
    if pipeline is None:
        pipeline = get_query_pipeline()
    
    # 运行查询pipeline
    result = pipeline.run(query=query)
    
    # 提取回答和文档
    answer = result.get("results", ["无法回答您的问题"])[0]
    documents = result.get("documents", [])
    
    return {
        "answer": answer,
        "documents": documents
    } 