#!/usr/bin/env python
"""
执行文档索引的脚本
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目根目录到PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from pipelines.index import index_documents
from config.settings import RAW_DOCUMENTS_DIR

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="为文档建立索引")
    parser.add_argument(
        "--dir", 
        type=str, 
        default=RAW_DOCUMENTS_DIR,
        help="包含要索引的文档的目录"
    )
    parser.add_argument(
        "--extensions", 
        type=str, 
        default="pdf,txt,md,docx",
        help="要索引的文件扩展名，以逗号分隔"
    )
    return parser.parse_args()

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 解析参数
    args = parse_args()
    
    # 验证目录存在
    documents_dir = Path(args.dir)
    if not documents_dir.exists():
        logging.error(f"目录不存在: {documents_dir}")
        return 1
    
    # 解析文件扩展名
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    
    # 执行索引
    logging.info(f"开始为目录 {documents_dir} 中的文件建立索引")
    logging.info(f"要处理的文件类型: {', '.join(extensions)}")
    
    try:
        index_documents(documents_dir=documents_dir, file_extensions=extensions)
        logging.info("索引完成")
        return 0
    except Exception as e:
        logging.error(f"索引过程中出错: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 