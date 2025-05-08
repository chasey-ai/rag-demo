#!/usr/bin/env python3
"""
Haystack RAG Pipeline命令行工具 - 带有Token使用跟踪功能
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime

from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.pipeline import AsyncPipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from dotenv import load_dotenv
import time
import json
import inspect

# 尝试导入可选依赖
try:
    from token_reporter import generate_token_usage_report
    REPORTER_AVAILABLE = True
except ImportError:
    REPORTER_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class PipelineMonitor:
    """管道监控器，用于记录和分析pipeline中每个组件的执行情况"""
    
    def __init__(self):
        self.component_stats = {}
        self.current_run = {}
    
    async def component_callback(self, component_name, data):
        """组件运行前后的回调函数"""
        # 创建组件统计记录（如果不存在）
        if component_name not in self.component_stats:
            self.component_stats[component_name] = {
                "total_calls": 0,
                "total_time": 0,
                "runs": []
            }
        
        # 记录运行开始时间
        start_time = time.time()
        
        # 记录输入数据
        run_record = {
            "start_time": datetime.now().isoformat(),
            "input_data": self._sanitize_data(data)
        }
        
        # 保存当前运行记录
        self.current_run[component_name] = run_record
        
        # 返回原始数据，允许pipeline继续运行
        return data
    
    async def component_completion_callback(self, component_name, data):
        """组件运行完成后的回调函数"""
        if component_name in self.current_run:
            run_record = self.current_run[component_name]
            end_time = time.time()
            
            # 计算持续时间
            if "start_time" in run_record:
                start = datetime.fromisoformat(run_record["start_time"])
                duration = (datetime.now() - start).total_seconds()
                run_record["duration"] = duration
                
                # 更新组件统计
                self.component_stats[component_name]["total_calls"] += 1
                self.component_stats[component_name]["total_time"] += duration
            
            # 记录输出数据
            run_record["output_data"] = self._sanitize_data(data)
            
            # 保存运行记录到组件统计中
            self.component_stats[component_name]["runs"].append(run_record)
            
            # 清理当前运行记录
            del self.current_run[component_name]
        
        # 返回原始数据，允许pipeline继续运行
        return data
    
    def _sanitize_data(self, data):
        """净化数据，删除不可序列化的内容"""
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif hasattr(data, "__dict__"):
            # 尝试获取对象的字典表示
            try:
                return self._sanitize_data(vars(data))
            except:
                return str(data)
        elif inspect.isfunction(data) or inspect.ismethod(data):
            return f"<function {data.__name__}>"
        else:
            # 尝试JSON序列化，如果失败则转换为字符串
            try:
                json.dumps(data)
                return data
            except:
                return str(data)
    
    def get_summary(self):
        """获取所有组件的统计摘要"""
        summary = {}
        
        for component_name, stats in self.component_stats.items():
            component_summary = {
                "total_calls": stats["total_calls"],
                "total_time": stats["total_time"],
                "avg_time": stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
            }
            
            # 对于LLM组件，添加token使用情况
            if component_name == "llm" and stats["runs"]:
                token_stats = {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0
                }
                
                for run in stats["runs"]:
                    if "output_data" in run and "replies" in run["output_data"]:
                        for reply in run["output_data"]["replies"]:
                            if hasattr(reply, "_meta") and "usage" in reply._meta:
                                usage = reply._meta["usage"]
                                token_stats["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                                token_stats["total_completion_tokens"] += usage.get("completion_tokens", 0)
                                token_stats["total_tokens"] += usage.get("total_tokens", 0)
                
                component_summary.update(token_stats)
            
            summary[component_name] = component_summary
        
        return summary
    
    def print_summary(self):
        """打印所有组件的统计摘要"""
        summary = self.get_summary()
        
        print("\n===== Pipeline组件执行统计 =====")
        for component_name, stats in summary.items():
            print(f"\n组件: {component_name}")
            print(f"  调用次数: {stats['total_calls']}")
            print(f"  总执行时间: {stats['total_time']:.2f}秒")
            print(f"  平均执行时间: {stats['avg_time']:.2f}秒")
            
            # 打印token使用情况（如果有）
            if "total_prompt_tokens" in stats:
                print(f"  提示词tokens总数: {stats['total_prompt_tokens']}")
                print(f"  补全tokens总数: {stats['total_completion_tokens']}")
                print(f"  总tokens: {stats['total_tokens']}")
    
    def save_to_file(self, filename="pipeline_stats.json"):
        """保存统计数据到文件"""
        with open(filename, "w") as f:
            json.dump({
                "component_stats": self.component_stats,
                "summary": self.get_summary()
            }, f, indent=2, default=str)
        
        print(f"\n组件统计数据已保存到 {filename}")


class TokenUsageTracker:
    def __init__(self, log_file="token_usage_log.json"):
        self.log_file = log_file
        self.usage_records = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.query_count = 0
    
    def record_usage(self, query, results):
        """记录单次查询的token使用情况"""
        usage_data = self._extract_usage_from_results(results)
        
        if not usage_data:
            return None
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "usage": usage_data,
        }
        
        # 更新总计
        self.usage_records.append(record)
        self.total_prompt_tokens += usage_data["prompt_tokens"]
        self.total_completion_tokens += usage_data["completion_tokens"]
        self.total_tokens += usage_data["total_tokens"]
        self.query_count += 1
        
        return usage_data
    
    def _extract_usage_from_results(self, results):
        """从结果中提取token使用信息"""
        if "llm" in results and "replies" in results["llm"]:
            message = results["llm"]["replies"][0]
            if hasattr(message, "_meta") and "usage" in message._meta:
                # 提取原始使用数据
                raw_usage = message._meta["usage"]
                
                # 创建干净的字典而不是对象引用
                usage_data = {
                    "prompt_tokens": raw_usage["prompt_tokens"],
                    "completion_tokens": raw_usage["completion_tokens"],
                    "total_tokens": raw_usage["total_tokens"],
                    "model": message._meta.get("model", "unknown")
                }
                
                # 如果有详细信息，也记录下来
                if "prompt_tokens_details" in raw_usage:
                    prompt_details = raw_usage["prompt_tokens_details"]
                    usage_data["prompt_tokens_details"] = {
                        key: getattr(prompt_details, key)
                        for key in prompt_details.__dict__
                    }
                
                if "completion_tokens_details" in raw_usage:
                    completion_details = raw_usage["completion_tokens_details"]
                    usage_data["completion_tokens_details"] = {
                        key: getattr(completion_details, key)
                        for key in completion_details.__dict__
                    }
                
                return usage_data
        
        return None
    
    def print_usage(self, usage_data):
        """打印单次查询的使用情况"""
        if not usage_data:
            print("未找到token使用情况信息")
            return
        
        print("\n----- Token使用情况 -----")
        print(f"模型: {usage_data.get('model', 'unknown')}")
        print(f"提示词tokens: {usage_data['prompt_tokens']}")
        print(f"补全tokens: {usage_data['completion_tokens']}")
        print(f"总tokens: {usage_data['total_tokens']}")
    
    def print_summary(self):
        """打印所有查询的汇总使用情况"""
        print("\n===== Token使用汇总 =====")
        print(f"查询总数: {self.query_count}")
        print(f"提示词tokens总数: {self.total_prompt_tokens}")
        print(f"补全tokens总数: {self.total_completion_tokens}")
        print(f"总tokens: {self.total_tokens}")
        if self.query_count > 0:
            print(f"平均每次查询使用tokens: {self.total_tokens / self.query_count:.2f}")
    
    def save_to_file(self):
        """将使用记录保存到文件"""
        data = {
            "records": self.usage_records,
            "summary": {
                "query_count": self.query_count,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            }
        }
        
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n使用记录已保存到 {self.log_file}")


class RAGTokenTracker:
    """RAG管道Token跟踪器"""
    
    def __init__(self, documents=None):
        # 加载环境变量
        load_dotenv()
        
        # 创建文档存储
        self.document_store = InMemoryDocumentStore()
        
        # 加载默认文档（如果没有提供）
        if documents is None:
            documents = [
                Document(content="My name is Jean and I live in Paris."),
                Document(content="My name is Mark and I live in Berlin."),
                Document(content="My name is Giorgio and I live in Rome.")
            ]
        
        # 写入文档
        self.document_store.write_documents(documents)
        
        # 创建提示模板
        self.prompt_template = [
            ChatMessage.from_user(
                '''
                Given these documents, answer the question.
                Documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}
                Question: {{question}}
                Answer:
                ''')
        ]
        
        # 初始化组件
        self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
        self.prompt_builder = ChatPromptBuilder(template=self.prompt_template)
        self.llm = OpenAIChatGenerator()
        
        # 创建异步管道
        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        
        # 连接组件
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")
        
        # 创建监控器
        self.pipeline_monitor = PipelineMonitor()
        self.token_tracker = TokenUsageTracker()
        
        # 添加回调
        for component_name in ["retriever", "prompt_builder", "llm"]:
            self.pipeline.add_component_callback(
                component_name, 
                self.pipeline_monitor.component_callback, 
                self.pipeline_monitor.component_completion_callback
            )
    
    async def run_query(self, question):
        """运行查询并记录token使用情况"""
        print(f"\n处理查询: '{question}'")
        
        start_time = time.time()
        
        data = {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
        
        results = await self.pipeline.run(data)
        
        # 记录使用情况
        usage_data = self.token_tracker.record_usage(question, results)
        
        # 获取回答文本
        answer = results['llm']['replies'][0]._content[0].text
        
        # 打印结果和使用情况
        print(f"回答: {answer}")
        print(f"查询耗时: {time.time() - start_time:.2f}秒")
        self.token_tracker.print_usage(usage_data)
        
        return answer
    
    async def run_interactive(self):
        """运行交互式模式"""
        print("\n=== Haystack RAG Token追踪器 - 交互模式 ===")
        print("输入问题或命令 (输入'quit'或'exit'退出, 输入'stats'查看统计信息)")
        
        while True:
            try:
                query = input("\n> ")
                
                if query.lower() in ["quit", "exit", "q"]:
                    break
                
                if query.lower() in ["stats", "summary", "s"]:
                    self.token_tracker.print_summary()
                    self.pipeline_monitor.print_summary()
                    continue
                
                if query.strip() == "":
                    continue
                
                await self.run_query(query)
                
            except KeyboardInterrupt:
                print("\n已中断，正在退出...")
                break
            except Exception as e:
                print(f"错误: {str(e)}")
    
    def generate_reports(self, output_dir="reports"):
        """生成所有报告"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义文件路径
        token_log_file = os.path.join(output_dir, "token_usage_log.json")
        pipeline_stats_file = os.path.join(output_dir, "pipeline_stats.json")
        html_report_file = os.path.join(output_dir, "token_usage_report.html")
        
        # 保存token使用记录
        token_data = {
            "records": self.token_tracker.usage_records,
            "summary": {
                "query_count": self.token_tracker.query_count,
                "total_prompt_tokens": self.token_tracker.total_prompt_tokens,
                "total_completion_tokens": self.token_tracker.total_completion_tokens,
                "total_tokens": self.token_tracker.total_tokens,
            }
        }
        
        with open(token_log_file, "w") as f:
            json.dump(token_data, f, indent=2)
        
        print(f"\n使用记录已保存到 {token_log_file}")
        
        # 保存pipeline统计
        with open(pipeline_stats_file, "w") as f:
            json.dump({
                "component_stats": self.pipeline_monitor.component_stats,
                "summary": self.pipeline_monitor.get_summary()
            }, f, indent=2, default=str)
        
        print(f"组件统计数据已保存到 {pipeline_stats_file}")
        
        # 生成HTML报告
        if REPORTER_AVAILABLE:
            generate_token_usage_report(
                token_log_file=token_log_file,
                pipeline_stats_file=pipeline_stats_file,
                output_html=html_report_file
            )
            print(f"HTML报告已生成: {html_report_file}")
        else:
            print("提示: 缺少token_reporter模块，无法生成HTML报告")


async def main():
    parser = argparse.ArgumentParser(description="Haystack RAG Pipeline Token跟踪器")
    parser.add_argument("--query", "-q", help="要查询的问题", nargs="?")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    parser.add_argument("--output", "-o", default="reports", help="报告输出目录")
    
    args = parser.parse_args()
    
    # 创建RAG Token跟踪器
    rag_tracker = RAGTokenTracker()
    
    if args.interactive:
        # 进入交互模式
        await rag_tracker.run_interactive()
    elif args.query:
        # 处理单个查询
        await rag_tracker.run_query(args.query)
    else:
        # 运行默认示例
        await rag_tracker.run_query("Who lives in Paris?")
        await rag_tracker.run_query("Who lives in Berlin?")
        await rag_tracker.run_query("Who lives in Rome?")
    
    # 打印token使用情况汇总
    rag_tracker.token_tracker.print_summary()
    
    # 打印pipeline组件统计
    rag_tracker.pipeline_monitor.print_summary()
    
    # 生成报告
    rag_tracker.generate_reports(args.output)


if __name__ == "__main__":
    asyncio.run(main()) 