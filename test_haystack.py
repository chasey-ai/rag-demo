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

# 尝试导入token报告生成器
try:
    from token_reporter import generate_token_usage_report
    REPORTER_AVAILABLE = True
except ImportError:
    REPORTER_AVAILABLE = False

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import pandas as pd
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
    
    def visualize_component_timing(self, output_file="component_timing.png"):
        """可视化组件执行时间"""
        if not VISUALIZATION_AVAILABLE:
            print("无法生成可视化: 缺少 matplotlib 或 pandas 库")
            return
        
        summary = self.get_summary()
        
        # 准备数据
        components = []
        avg_times = []
        total_times = []
        
        for component_name, stats in summary.items():
            components.append(component_name)
            avg_times.append(stats["avg_time"])
            total_times.append(stats["total_time"])
        
        # 创建DataFrame
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
        plt.savefig(output_file)
        print(f"组件时间图表已保存到 {output_file}")
    
    def save_to_file(self, filename="pipeline_stats.json"):
        """保存统计数据到文件"""
        with open(filename, "w") as f:
            json.dump({
                "component_stats": self.component_stats,
                "summary": self.get_summary()
            }, f, indent=2, default=str)
        
        print(f"\n组件统计数据已保存到 {filename}")
        
        # 生成可视化
        if VISUALIZATION_AVAILABLE:
            # 创建存储图表的目录
            charts_dir = "token_usage_charts"
            os.makedirs(charts_dir, exist_ok=True)
            
            # 生成组件时间图表
            self.visualize_component_timing(os.path.join(charts_dir, "component_timing.png"))

# 创建管道监控器
pipeline_monitor = PipelineMonitor()

# 创建组件
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = OpenAIChatGenerator()

# 创建异步管道
rag_pipeline = AsyncPipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# 连接组件
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# 添加回调
for component_name in ["retriever", "prompt_builder", "llm"]:
    rag_pipeline.add_component_callback(
        component_name, 
        pipeline_monitor.component_callback, 
        pipeline_monitor.component_completion_callback
    )

# 初始化token使用记录器
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

# 创建token使用跟踪器
token_tracker = TokenUsageTracker()

async def run_query(question):
    """运行查询并记录token使用情况"""
    print(f"\n处理查询: '{question}'")
    
    start_time = time.time()
    
    data = {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
    
    results = await rag_pipeline.run(data)
    
    # 记录使用情况
    usage_data = token_tracker.record_usage(question, results)
    
    # 打印结果和使用情况
    print(f"回答: {results['llm']['replies'][0]._content[0].text}")
    print(f"查询耗时: {time.time() - start_time:.2f}秒")
    token_tracker.print_usage(usage_data)
    
    return results

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
    
    # 打印pipeline组件统计
    pipeline_monitor.print_summary()
    
    # 保存pipeline统计
    pipeline_monitor.save_to_file()
    
    # 生成HTML报告
    if REPORTER_AVAILABLE:
        generate_token_usage_report(
            token_log_file="token_usage_log.json",
            pipeline_stats_file="pipeline_stats.json",
            output_html="token_usage_report.html"
        )
        print("\nHTML报告已生成: token_usage_report.html")
    else:
        print("\n提示: 缺少token_reporter模块，无法生成HTML报告")

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())