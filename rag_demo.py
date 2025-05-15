# 导入必要的库和模块
import os
import json
from typing import Dict, List, Optional, Any
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack import component

# 创建查询拆分组件
@component
class QueryDecomposer:
    """
    将复杂的用户查询拆分为多个子查询，以提高检索的准确性。
    使用LLM来理解查询并生成子查询。
    """
    
    def __init__(self, llm: OpenAIChatGenerator):
        self.llm: OpenAIChatGenerator = llm
    
    @component.output_types(sub_queries=List[str])
    def run(self, query: str) -> Dict[str, List[str]]:
        # 创建提示以拆分查询
        prompt = [
            ChatMessage.from_system(
                "你是一个查询分析专家。你的任务是将用户的复杂查询拆分为2-5个简单的子查询，"
                "每个子查询都应该能够独立检索文档库中的相关信息。"
                "返回这些子查询的列表，不要包含任何其他解释或文本。"
                "每个子查询应该用换行符分隔。"
                "如果用户的查询已经很简单，可以只返回1-2个子查询。"
            ),
            ChatMessage.from_user(f"请将这个查询拆分为多个子查询: {query}")
        ]
        
        # 使用LLM生成子查询
        response = self.llm.run(messages=prompt)
        
        # 处理LLM的回复，提取子查询
        sub_queries_text = response['replies'][0].text
        sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]
        
        print(f"原始查询: {query}")
        print(f"生成的子查询: {sub_queries}")
        
        return {"sub_queries": sub_queries}

# 创建多查询检索组件
@component
class MultiQueryRetriever:
    """
    处理多个子查询并合并结果。
    将多个查询传递给检索器，并合并结果。
    """
    
    def __init__(self, retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
        self.text_embedder = OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
    
    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]) -> Dict[str, List[Document]]:
        # 存储所有唯一的文档
        unique_docs = {}
        
        # 针对每个子查询进行检索
        for query in queries:
            print(f"处理子查询: {query}")
            
            # 为向量检索器生成查询嵌入
            query_embedding_result = self.text_embedder.run(text=query)
            query_embedding = query_embedding_result["embedding"]
            
            # 执行检索 - 使用embedding参数而不是query
            retrieval_result = self.retriever.run(query_embedding=query_embedding)
            retrieved_docs = retrieval_result["documents"]
            
            # 将找到的文档添加到唯一文档集合中
            for doc in retrieved_docs:
                if doc.id not in unique_docs:
                    unique_docs[doc.id] = doc
        
        # 将唯一文档转换为列表并限制数量
        result_docs = list(unique_docs.values())
        result_docs = result_docs[:self.top_k]
        
        print(f"合并后的文档数量: {len(result_docs)}")
        
        return {"documents": result_docs}

# 用于将复杂对象转换为可JSON序列化的辅助函数
def serialize_for_json(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return serialize_for_json(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return serialize_for_json(obj.__dict__)
    else:
        return str(obj)

# 初始化内存文档存储，配置为支持向量检索，使用余弦相似度
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# 创建文档索引处理管道
indexing_pipeline = Pipeline()

# 添加文本文件转换器组件，用于将文本文件转换为文档对象(txt文件 --> Document对象)
file_converter = TextFileToDocument()
indexing_pipeline.add_component("file_converter", file_converter)

# 添加文档分割器组件，提高分割的粒度和质量
document_splitter = DocumentSplitter(
    split_by="passage",
    split_length=1,  # 每段包含2个句子
    split_overlap=0  # 1个句子的重叠，提高上下文连贯性
)
indexing_pipeline.add_component("document_splitter", document_splitter)

# 添加OpenAI文档嵌入器，生成高质量的文档嵌入
document_embedder = OpenAIDocumentEmbedder(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="text-embedding-3-small"  # OpenAI最新的嵌入模型，支持多语言
)
indexing_pipeline.add_component("document_embedder", document_embedder)

# 添加文档写入器组件，将处理后的文档写入文档存储
document_writer = DocumentWriter(document_store=document_store)
indexing_pipeline.add_component("document_writer", document_writer)

# 连接索引管道的各个组件，形成处理流程
indexing_pipeline.connect("file_converter", "document_splitter")
indexing_pipeline.connect("document_splitter", "document_embedder")
indexing_pipeline.connect("document_embedder", "document_writer")

# 构建RAG（检索增强生成）管道的提示模板
prompt_template = [
    ChatMessage.from_system("你是一个专业助手，根据提供的文档准确回答问题。如果文档中没有相关信息，请说明你不知道，不要编造答案。"),
    ChatMessage.from_user(
        "以下是相关文档内容:\n"
        "{% for doc in documents %}\n"
        "文档{{loop.index}}：{{ doc.content }}\n"
        "{% endfor %}\n\n"
        "请根据上述文档内容，回答以下问题：{{question}}\n"
    )
]

# 创建提示构建器，明确指定所需的变量
prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables={"question", "documents"})

# 创建OpenAI文本嵌入器，用于查询向量化
text_embedder = OpenAITextEmbedder(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="text-embedding-3-small"  # 与文档嵌入器使用相同的模型
)

# 初始化向量检索器和OpenAI聊天生成器
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=2)
llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")

# 构建RAG管道
rag_pipeline = Pipeline()

# 添加查询拆分组件
query_decomposer = QueryDecomposer(llm=OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))
rag_pipeline.add_component("query_decomposer", query_decomposer)

# 添加文本嵌入器组件
rag_pipeline.add_component("text_embedder", text_embedder)

# 添加检索器组件
rag_pipeline.add_component("retriever", retriever)

# 添加多查询检索组件（使用已有的检索器实例）
multi_query_retriever = MultiQueryRetriever(retriever=retriever, top_k=5)
rag_pipeline.add_component("multi_query_retriever", multi_query_retriever)

# 添加提示构建器和LLM
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# 定义连接
# 查询拆分与多查询检索器连接
rag_pipeline.connect("query_decomposer.sub_queries", "multi_query_retriever.queries")
# 文本嵌入器与检索器连接
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
# 多查询检索器与提示构建器连接
rag_pipeline.connect("multi_query_retriever.documents", "prompt_builder.documents")
# 提示构建器与LLM连接
rag_pipeline.connect("prompt_builder", "llm.messages")

# 将核心功能封装到函数中，避免在被导入时执行
def init_document_store():
    """初始化和索引文档存储"""
    global document_store, indexing_pipeline
    
    # 检查文档存储中是否已有文档，避免重复索引
    if document_store.count_documents() > 0:
        print(f"文档存储中已有 {document_store.count_documents()} 个文档，跳过索引步骤")
        return document_store
    
    # 处理指定目录下的所有txt文件
    files_dir = "files"  # 请确保这个目录存在并包含要处理的文本文件
    file_paths = [os.path.join(files_dir, f) for f in os.listdir(files_dir) if f.endswith('.txt')]
    
    if not file_paths:
        print("警告: 'files'目录中没有找到txt文件")
        return document_store
    
    print(f"开始索引 {len(file_paths)} 个文本文件...")
    index_results = indexing_pipeline.run(
        {"file_converter": {"sources": file_paths}}
    )
    
    # 将索引结果转换为可序列化的形式并写入json文件
    serialized_index_results = serialize_for_json(index_results)
    with open("index_results.json", "w", encoding="utf-8") as f:
        json.dump(serialized_index_results, f, ensure_ascii=False, indent=2)
    
    print(f"索引完成，共处理 {len(file_paths)} 个文件，存储了 {document_store.count_documents()} 个文档")
    return document_store

def ask_question(question):
    """使用初始化好的RAG管道回答问题"""
    results = rag_pipeline.run(
        {
            "query_decomposer": {"query": question},
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
        },
        include_outputs_from={"query_decomposer", "text_embedder", "multi_query_retriever", "prompt_builder", "llm"}
    )
    
    return results

if __name__ == "__main__":
    # 初始化文档存储和索引
    init_document_store()
    
    # 执行问题查询
    question = "查理是什么职业？他有什么爱好？"
    results = ask_question(question)
    
    # 将 results 转换为可序列化的形式并写入json文件
    serialized_results = serialize_for_json(results)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(serialized_results, f, ensure_ascii=False, indent=2)
    
    # 打印LLM的回答结果
    print(results["llm"]["replies"])
