# rag-demo
一个使用 Chainlit+Haystack 的 RAG（检索增强生成）学习项目

## 项目概述
本项目演示了如何使用现代工具构建检索增强生成(RAG)系统，通过结合文档检索和大语言模型能力，提供基于知识库的精准回答。

## 技术栈
- **Chainlit**：提供直观的对话式前端界面
- **Haystack**：提供完整的RAG系统组件支持
- **向量数据库**：支持FAISS或Chroma存储文档向量
- **大语言模型**：可配置使用不同的LLM提供生成能力

## 文件结构
```
rag_demo/
├── data/                      # 存放原始数据和索引数据
│   ├── raw_documents/         # 存放原始文档 (PDF, TXT, MD 等)
│   │   ├── doc1.pdf
│   │   └── another_topic.txt
│   └── faiss_document_store.db # (示例) Haystack FAISS 索引文件
│   └── document_store.faiss   # (示例) FAISS 索引的另一个常见命名
├── pipelines/                 # 存放 Haystack Pipeline 的定义
│   ├── __init__.py
│   ├── index.py               # 定义数据索引的 Haystack Pipeline
│   ├── query.py               # 定义查询和生成的 Haystack Pipeline
│   ├── llm.py                 # 定义与 LLM 交互的 Client
│   └── utils.py               # Pipeline 相关的辅助函数 (可选)
├── app.py                     # Chainlit 应用的主文件（用户与 RAG 系统的前端交互）
├── config/                    # 配置文件
│   ├── __init__.py
│   ├── settings.py            # 应用配置 (API Keys, 路径, 模型名称等)
│   └── logging_config.yaml    # (可选) 日志配置
├── scripts/                   # (可选) 辅助脚本
│   ├── run_indexing.py        # 运行索引流程的脚本
│   └── bulk_upload_data.sh    # (示例) 批量上传数据的脚本
├── .env                       # 存储环境变量 (API 密钥等，不应提交到 Git)
├── requirements.txt           # Python 依赖包列表
├── README.md                  # 项目说明文件
└── .gitignore                 # 指定 Git 应忽略的文件和目录
```

## 系统流程
1. **数据处理**：将原始文档放在`data/raw_documents/`目录下
2. **索引构建**：使用`pipelines/index.py`中定义的流程将文档转换为向量并存入向量数据库
3. **查询处理**：当用户提问时，系统使用`pipelines/query.py`中的流程检索相关文档并生成回答
4. **前端交互**：通过Chainlit提供的界面，用户可以自然地与RAG系统进行对话

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   创建`.env`文件并添加必要的API密钥

3. 构建索引：
   ```bash
   python scripts/run_indexing.py
   ```

4. 启动应用：
   ```bash
   chainlit run app.py
   ```

## 代码仓库
GitHub: https://github.com/YOUR_USERNAME/rag-demo

## 贡献指南
欢迎提交PR或Issue来改进此项目。具体贡献流程请参考贡献指南文档。

## 许可证
MIT

# Haystack RAG Token使用跟踪系统

这是一个用于跟踪和分析Haystack RAG Pipeline中token使用情况的工具。该工具可以帮助您监控OpenAI API调用的token使用情况，生成可视化报告，以及分析pipeline中各组件的执行性能。

## 功能特点

- 记录每次查询的token使用情况（提示词、补全、总计）
- 跟踪整个pipeline中各组件的执行时间和性能
- 生成可视化图表，包括token分布、查询token使用趋势等
- 生成HTML格式的详细报告
- 保存原始数据以便进一步分析

## 安装要求

```bash
pip install haystack-ai python-dotenv matplotlib pandas
```

## 使用方法

1. 将token追踪组件集成到您的Haystack RAG pipeline中：

```python
# 创建token使用跟踪器
token_tracker = TokenUsageTracker()

# 创建pipeline监控器
pipeline_monitor = PipelineMonitor()

# 向pipeline添加回调
for component_name in ["retriever", "prompt_builder", "llm"]:
    rag_pipeline.add_component_callback(
        component_name, 
        pipeline_monitor.component_callback, 
        pipeline_monitor.component_completion_callback
    )

# 在查询处理中记录token使用情况
usage_data = token_tracker.record_usage(question, results)
```

2. 运行查询并获取统计信息：

```python
# 打印token使用情况汇总
token_tracker.print_summary()

# 保存token使用记录
token_tracker.save_to_file()

# 打印pipeline组件统计
pipeline_monitor.print_summary()

# 保存pipeline统计
pipeline_monitor.save_to_file()

# 生成HTML报告
generate_token_usage_report()
```

## 输出示例

### 控制台输出

```
===== Token使用汇总 =====
查询总数: 3
提示词tokens总数: 207
补全tokens总数: 18
总tokens: 225
平均每次查询使用tokens: 75.00

===== Pipeline组件执行统计 =====

组件: retriever
  调用次数: 3
  总执行时间: 0.01秒
  平均执行时间: 0.00秒

组件: prompt_builder
  调用次数: 3
  总执行时间: 0.02秒
  平均执行时间: 0.01秒

组件: llm
  调用次数: 3
  总执行时间: 3.25秒
  平均执行时间: 1.08秒
  提示词tokens总数: 207
  补全tokens总数: 18
  总tokens: 225
```

### 数据文件

- `token_usage_log.json`: 包含所有查询的详细token使用记录
- `pipeline_stats.json`: 包含pipeline各组件的执行统计信息

### 可视化图表

- `token_distribution_pie.png`: token使用分布饼图
- `token_usage_by_query.png`: 每次查询的token使用柱状图
- `token_usage_trend.png`: token使用趋势图
- `component_timing.png`: 组件执行时间图

### HTML报告

生成的HTML报告(`token_usage_report.html`)包含所有统计数据和图表，可在浏览器中查看。

## 自定义与扩展

您可以根据需要自定义以下内容：

- `TokenUsageTracker`: 修改记录的token使用数据格式
- `PipelineMonitor`: 添加更多组件性能指标
- `generate_token_usage_report`: 自定义HTML报告格式和样式

## 注意事项

- 确保在生产环境中谨慎使用，因为记录详细数据可能会增加内存使用
- 对于大型应用，考虑使用数据库而非JSON文件存储使用记录
- 可视化功能需要安装matplotlib和pandas库