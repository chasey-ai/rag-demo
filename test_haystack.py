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
from datetime import datetime
import inspect
import os

# 检查是否可以使用可视化功能
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # 配置matplotlib支持中文字体
    import matplotlib as mpl
    # 尝试使用系统中的中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("提示: 安装 matplotlib 和 pandas 以启用可视化功能 (pip install matplotlib pandas)")

# 加载环境变量
load_dotenv()


# 初始化文档存储
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

prompt_template = [
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


class TokenUsageTracker:
    """Token使用跟踪器，用于记录LLM调用的token使用情况"""
    
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
        
        # 如果有详细信息，也显示出来
        if "prompt_tokens_details" in usage_data:
            print("\n提示词详情:")
            for key, value in usage_data["prompt_tokens_details"].items():
                print(f"  - {key}: {value}")
            
        if "completion_tokens_details" in usage_data:
            print("\n补全详情:")
            for key, value in usage_data["completion_tokens_details"].items():
                print(f"  - {key}: {value}")
    
    def print_summary(self):
        """打印所有查询的汇总使用情况"""
        print("\n===== Token使用汇总 =====")
        print(f"查询总数: {self.query_count}")
        print(f"提示词tokens总数: {self.total_prompt_tokens}")
        print(f"补全tokens总数: {self.total_completion_tokens}")
        print(f"总tokens: {self.total_tokens}")
        if self.query_count > 0:
            print(f"平均每次查询使用tokens: {self.total_tokens / self.query_count:.2f}")
    
    def visualize_token_usage(self):
        """可视化token使用情况"""
        if not VISUALIZATION_AVAILABLE:
            print("无法生成可视化: 缺少 matplotlib 或 pandas 库")
            return
        
        # 创建存储图表的目录
        charts_dir = "token_usage_charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # 准备查询数据
        if not self.usage_records:
            print("无记录数据可供可视化")
            return
            
        queries = [record["query"] for record in self.usage_records]
        prompt_tokens = [record["usage"]["prompt_tokens"] for record in self.usage_records]
        completion_tokens = [record["usage"]["completion_tokens"] for record in self.usage_records]
        total_tokens = [record["usage"]["total_tokens"] for record in self.usage_records]
        
        # 1. 创建每次查询的token使用饼图
        plt.figure(figsize=(12, 8))
        
        # 准备汇总数据
        summary_data = [
            self.total_prompt_tokens,
            self.total_completion_tokens
        ]
        
        labels = ["提示词Tokens", "补全Tokens"]
        colors = ["#3498db", "#e74c3c"]
        explode = (0.05, 0)  # 轻微分离第一个扇形
        
        plt.pie(
            summary_data, 
            explode=explode, 
            labels=labels, 
            colors=colors, 
            autopct="%1.1f%%", 
            shadow=True, 
            startangle=140
        )
        plt.axis("equal")  # 保持圆形
        plt.title(f"Token使用分布 (总计: {self.total_tokens})")
        plt.savefig(os.path.join(charts_dir, "token_distribution_pie.png"))
        plt.close()
        
        # 2. 创建每次查询的token使用柱状图
        df = pd.DataFrame({
            "查询": queries,
            "提示词Tokens": prompt_tokens,
            "补全Tokens": completion_tokens,
            "总计Tokens": total_tokens
        })
        
        # 排序查询以确保一致的显示顺序
        query_order = sorted(queries)
        df["查询"] = pd.Categorical(df["查询"], categories=query_order, ordered=True)
        df = df.sort_values("查询")
        
        # 创建堆叠柱状图
        ax = df.plot(
            x="查询",
            y=["提示词Tokens", "补全Tokens"],
            kind="bar",
            stacked=True,
            figsize=(12, 8),
            color=["#3498db", "#e74c3c"]
        )
        
        # 添加数据标签
        for c in ax.containers:
            ax.bar_label(c, label_type="center", fmt="%d")
        
        # 添加总计标签
        for i, total in enumerate(total_tokens):
            ax.text(i, total + 1, f"{total}", ha="center", va="bottom", fontweight="bold")
            
        plt.title("每次查询的Token使用情况")
        plt.ylabel("Tokens数量")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "token_usage_by_query.png"))
        plt.close()
        
        # 3. 创建token使用趋势图
        plt.figure(figsize=(12, 8))
        plt.plot(queries, prompt_tokens, marker="o", linewidth=2, label="提示词Tokens")
        plt.plot(queries, completion_tokens, marker="s", linewidth=2, label="补全Tokens")
        plt.plot(queries, total_tokens, marker="^", linewidth=3, label="总计Tokens")
        
        plt.title("Token使用趋势")
        plt.xlabel("查询")
        plt.ylabel("Tokens数量")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "token_usage_trend.png"))
        plt.close()
        
        print(f"Token使用图表已保存到 {charts_dir} 目录")
    
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
        
        # 生成可视化
        if VISUALIZATION_AVAILABLE and self.usage_records:
            self.visualize_token_usage()


class PipelineAnalyzer:
    """Pipeline分析器，用于记录Pipeline执行性能"""
    
    def __init__(self, stats_file="pipeline_stats.json"):
        self.stats_file = stats_file
        self.component_times = {}
        self.component_counts = {}
        self.llm_token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0
        }
    
    def record_llm_usage(self, results):
        """记录LLM的token使用情况"""
        if "llm" in results and "replies" in results["llm"]:
            message = results["llm"]["replies"][0]
            if hasattr(message, "_meta") and "usage" in message._meta:
                usage = message._meta["usage"]
                self.llm_token_usage["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.llm_token_usage["total_completion_tokens"] += usage.get("completion_tokens", 0)
                self.llm_token_usage["total_tokens"] += usage.get("total_tokens", 0)
    
    async def analyze_pipeline_execution(self, pipeline, data, include_outputs=None):
        """分析Pipeline执行过程中的性能指标"""
        start_times = {}
        component_results = {}
        
        # 监控执行开始
        overall_start = time.time()
        
        # 使用run_async_generator来监控每个组件的执行
        async for partial_result in pipeline.run_async_generator(
            data=data,
            include_outputs_from=include_outputs
        ):
            # 获取执行完成的组件名
            for component_name in partial_result.keys():
                if component_name not in start_times:
                    # 记录组件完成时间
                    component_end = time.time()
                    
                    # 更新组件统计
                    if component_name not in self.component_times:
                        self.component_times[component_name] = 0
                        self.component_counts[component_name] = 0
                    
                    # 假设我们不知道确切的开始时间，所以每个组件的时间是从整体开始到组件完成的时间
                    # 在实际应用中，这个时间不够准确，但可作为估计
                    component_time = component_end - overall_start
                    self.component_times[component_name] += component_time
                    self.component_counts[component_name] += 1
                    
                    # 保存组件结果
                    component_results[component_name] = partial_result[component_name]
                    
                    # 如果是LLM组件，记录token使用情况
                    if component_name == "llm":
                        self.record_llm_usage(partial_result)
        
        # 计算总执行时间
        overall_time = time.time() - overall_start
        
        # 返回结果
        return {
            "overall_time": overall_time,
            "component_results": component_results
        }
    
    def print_summary(self):
        """打印Pipeline执行汇总情况"""
        print("\n===== Pipeline执行统计 =====")
        
        for component_name, total_time in self.component_times.items():
            calls = self.component_counts[component_name]
            avg_time = total_time / calls if calls > 0 else 0
            
            print(f"\n组件: {component_name}")
            print(f"  调用次数: {calls}")
            print(f"  总执行时间: {total_time:.2f}秒")
            print(f"  平均执行时间: {avg_time:.2f}秒")
            
            # 如果是LLM组件，打印token使用情况
            if component_name == "llm" and self.llm_token_usage["total_tokens"] > 0:
                print(f"  提示词tokens总数: {self.llm_token_usage['total_prompt_tokens']}")
                print(f"  补全tokens总数: {self.llm_token_usage['total_completion_tokens']}")
                print(f"  总tokens: {self.llm_token_usage['total_tokens']}")
    
    def visualize_execution_times(self, output_dir="token_usage_charts"):
        """可视化组件执行时间"""
        if not VISUALIZATION_AVAILABLE:
            print("无法生成可视化: 缺少 matplotlib 或 pandas 库")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        components = list(self.component_times.keys())
        avg_times = [self.component_times[c] / self.component_counts[c] if self.component_counts[c] > 0 else 0 
                   for c in components]
        total_times = [self.component_times[c] for c in components]
        
        # 创建数据框
        df = pd.DataFrame({
            "组件": components,
            "平均执行时间(秒)": avg_times,
            "总执行时间(秒)": total_times
        })
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 平均执行时间图表
        df.plot(x="组件", y="平均执行时间(秒)", kind="bar", ax=ax1, color="skyblue")
        ax1.set_title("各组件平均执行时间")
        ax1.set_ylabel("执行时间(秒)")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 总执行时间图表
        df.plot(x="组件", y="总执行时间(秒)", kind="bar", ax=ax2, color="salmon")
        ax2.set_title("各组件总执行时间")
        ax2.set_ylabel("执行时间(秒)")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "component_timing.png")
        plt.savefig(output_file)
        plt.close()
        
        print(f"组件执行时间图表已保存到 {output_file}")
    
    def save_to_file(self):
        """保存执行统计到文件"""
        # 计算平均时间
        avg_times = {
            component: self.component_times[component] / self.component_counts[component] 
            if self.component_counts[component] > 0 else 0
            for component in self.component_times
        }
        
        data = {
            "component_stats": {
                component: {
                    "total_calls": self.component_counts[component],
                    "total_time": self.component_times[component],
                    "avg_time": avg_times[component]
                }
                for component in self.component_times
            },
            "summary": {
                "components": list(self.component_times.keys()),
                "llm_token_usage": self.llm_token_usage
            }
        }
        
        # 为LLM组件添加token使用情况
        if "llm" in data["component_stats"] and self.llm_token_usage["total_tokens"] > 0:
            data["component_stats"]["llm"].update(self.llm_token_usage)
        
        with open(self.stats_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n组件统计数据已保存到 {self.stats_file}")
        
        # 生成可视化
        self.visualize_execution_times()


# 创建组件
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = OpenAIChatGenerator()

# 创建异步管道
rag_pipeline = AsyncPipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# 创建token跟踪器
token_tracker = TokenUsageTracker()
pipeline_analyzer = PipelineAnalyzer()

async def run_query(question):
    """运行查询并记录token使用情况"""
    print(f"\n处理查询: '{question}'")
    
    start_time = time.time()
    
    data = {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
    
    # 使用pipeline_analyzer来分析pipeline执行
    result_with_analysis = await pipeline_analyzer.analyze_pipeline_execution(
        rag_pipeline, 
        data, 
        include_outputs=["retriever", "prompt_builder", "llm"]
    )
    
    # 获取最终结果
    results = result_with_analysis["component_results"]
    
    # 记录使用情况
    usage_data = token_tracker.record_usage(question, results)
    
    answer_text = "未获得回答"
    if "llm" in results and "replies" in results["llm"]:
        answer_text = results["llm"]["replies"][0]._content[0].text
    
    # 打印结果和使用情况
    print(f"回答: {answer_text}")
    print(f"查询耗时: {time.time() - start_time:.2f}秒")
    token_tracker.print_usage(usage_data)
    
    # 返回组合的结果
    return {
        "answer": answer_text,
        "results": results,
        "usage_data": usage_data,
        "execution_time": time.time() - start_time
    }

# 运行示例查询
import asyncio

async def main():
    # 运行查询
    await run_query("Who lives in Paris?")
    await run_query("Who lives in Berlin?")
    await run_query("Who lives in Rome?")
    
    # 打印token使用情况汇总
    token_tracker.print_summary()
    
    # 保存token使用记录
    token_tracker.save_to_file()
    
    # 打印pipeline执行统计
    pipeline_analyzer.print_summary()
    
    # 保存pipeline统计
    pipeline_analyzer.save_to_file()
    
    # 生成HTML报告
    try:
        from token_reporter import generate_token_usage_report
        generate_token_usage_report(
            token_log_file="token_usage_log.json",
            pipeline_stats_file="pipeline_stats.json",
            output_html="token_usage_report.html"
        )
        print("\nHTML报告已生成: token_usage_report.html")
    except ImportError:
        print("\n提示: 缺少token_reporter模块，无法生成HTML报告")

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())