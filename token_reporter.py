import json
import os
from datetime import datetime

def generate_token_usage_report(token_log_file="token_usage_log.json", pipeline_stats_file="pipeline_stats.json", output_html="token_usage_report.html"):
    """
    生成token使用情况的HTML报告
    """
    # 检查文件是否存在
    if not os.path.exists(token_log_file) or not os.path.exists(pipeline_stats_file):
        print("错误: 未找到日志文件")
        return
    
    # 加载token使用数据
    with open(token_log_file, "r") as f:
        token_data = json.load(f)
    
    # 加载管道统计数据
    with open(pipeline_stats_file, "r") as f:
        pipeline_data = json.load(f)
    
    # 提取摘要数据
    token_summary = token_data["summary"]
    records = token_data["records"]
    
    # 构建HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Token使用情况报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .summary-box {{
                background-color: #f9f9f9;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin-right: 20px;
                font-size: 1.1em;
            }}
            .metric span {{
                font-weight: bold;
                color: #3498db;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .chart-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 20px 0;
            }}
            .chart {{
                width: 48%;
                margin-bottom: 20px;
                border: 1px solid #ddd;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .chart img {{
                width: 100%;
                height: auto;
            }}
            footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }}
            @media (max-width: 768px) {{
                .chart {{
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Token使用情况报告</h1>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <h2>使用摘要</h2>
        <div class="summary-box">
            <div class="metric">查询总数: <span>{token_summary["query_count"]}</span></div>
            <div class="metric">提示词tokens总数: <span>{token_summary["total_prompt_tokens"]}</span></div>
            <div class="metric">补全tokens总数: <span>{token_summary["total_completion_tokens"]}</span></div>
            <div class="metric">总tokens: <span>{token_summary["total_tokens"]}</span></div>
    """
    
    # 计算平均值
    if token_summary["query_count"] > 0:
        avg_tokens = token_summary["total_tokens"] / token_summary["query_count"]
        html += f"""
            <div class="metric">平均每次查询使用tokens: <span>{avg_tokens:.2f}</span></div>
        """
    
    html += """
        </div>
        
        <h2>查询详情</h2>
        <table>
            <tr>
                <th>查询</th>
                <th>模型</th>
                <th>提示词Tokens</th>
                <th>补全Tokens</th>
                <th>总Tokens</th>
                <th>时间</th>
            </tr>
    """
    
    # 添加每个查询的详情
    for record in records:
        query = record["query"]
        usage = record["usage"]
        timestamp = datetime.fromisoformat(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        html += f"""
            <tr>
                <td>{query}</td>
                <td>{usage.get("model", "未知")}</td>
                <td>{usage["prompt_tokens"]}</td>
                <td>{usage["completion_tokens"]}</td>
                <td>{usage["total_tokens"]}</td>
                <td>{timestamp}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>可视化</h2>
        <div class="chart-container">
    """
    
    # 添加图表引用
    charts_dir = "token_usage_charts"
    charts = [
        {"file": "token_distribution_pie.png", "title": "Token使用分布"},
        {"file": "token_usage_by_query.png", "title": "每次查询的Token使用情况"},
        {"file": "token_usage_trend.png", "title": "Token使用趋势"},
        {"file": "component_timing.png", "title": "组件执行时间"}
    ]
    
    for chart in charts:
        chart_path = os.path.join(charts_dir, chart["file"])
        if os.path.exists(chart_path):
            html += f"""
                <div class="chart">
                    <h3>{chart["title"]}</h3>
                    <img src="{chart_path}" alt="{chart["title"]}">
                </div>
            """
    
    html += """
        </div>
        
        <h2>Pipeline组件分析</h2>
    """
    
    # 添加Pipeline组件信息
    component_summary = pipeline_data["summary"]
    for component_name, stats in component_summary.items():
        html += f"""
            <h3>组件: {component_name}</h3>
            <div class="summary-box">
                <div class="metric">调用次数: <span>{stats["total_calls"]}</span></div>
                <div class="metric">总执行时间: <span>{stats["total_time"]:.2f}秒</span></div>
                <div class="metric">平均执行时间: <span>{stats["avg_time"]:.2f}秒</span></div>
        """
        
        # 对于LLM组件，添加token信息
        if "total_prompt_tokens" in stats:
            html += f"""
                <div class="metric">提示词tokens总数: <span>{stats["total_prompt_tokens"]}</span></div>
                <div class="metric">补全tokens总数: <span>{stats["total_completion_tokens"]}</span></div>
                <div class="metric">总tokens: <span>{stats["total_tokens"]}</span></div>
            """
        
        html += """
            </div>
        """
    
    html += """
        <footer>
            <p>Haystack RAG Pipeline Token使用报告</p>
        </footer>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_html, "w") as f:
        f.write(html)
    
    print(f"HTML报告已生成: {output_html}")

if __name__ == "__main__":
    generate_token_usage_report() 