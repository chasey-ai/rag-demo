#!/bin/bash
# 批量上传数据的脚本

# 设置文档源目录和目标目录
SOURCE_DIR=${1:-"./source_documents"}
TARGET_DIR="./rag_demo/data/raw_documents"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录 $SOURCE_DIR 不存在"
    exit 1
fi

# 确保目标目录存在
mkdir -p "$TARGET_DIR"

# 复制所有文档文件到目标目录
echo "正在将文档从 $SOURCE_DIR 复制到 $TARGET_DIR"
find "$SOURCE_DIR" -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" -o -name "*.docx" \) -exec cp {} "$TARGET_DIR" \;
echo "文件复制完成"

# 统计复制的文件数量
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
echo "目标目录中现有 $FILE_COUNT 个文档文件"

# 提示用户运行索引脚本
echo "要为这些文档建立索引，请运行: python rag_demo/scripts/run_indexing.py" 