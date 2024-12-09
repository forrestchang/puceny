import json
import os
import re
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional

import PyPDF2
from bs4 import BeautifulSoup
from flask import Flask, render_template_string, request, send_file

# 假设 IndexWriter, IndexReader, Query, Searcher, IndexMerger 在同一目录中的另一个文件中有实现
from puceny import (  # 替换为实际导入路径
    Analyzer,
    Document,
    Field,
    FieldType,
    IndexReader,
    IndexWriter,
    Query,
    Searcher,
)

# 全局定义数据目录和索引目录
DATA_DIR = "/Users/jiayuan/dev/devvai/playground/cs61a_content/20241129_154852"
INDEX_DIR = "puceny_index_cs61a"
DATA_DIR = os.path.abspath(DATA_DIR)
ABS_DATA_DIR = os.path.abspath(DATA_DIR)


def extract_text_from_file(file_path: str) -> str:
    # 根据文件扩展名处理不同类型的文件
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        # 直接读取文本
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            return ""
    elif ext == ".html":
        # 使用BeautifulSoup解析
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ")
        except:
            return ""
    elif ext == ".pdf":
        # 使用PyPDF2解析PDF文本
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text() or "")
                return " ".join(text)
        except:
            return ""
    else:
        # 不支持的文件类型，返回空字符串
        return ""


################################
# 构建索引函数
################################


def build_index_from_directory(index_dir: str, data_dir: str):
    analyzer = Analyzer()
    writer = IndexWriter(index_dir, analyzer)

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            text_content = extract_text_from_file(file_path)
            if text_content.strip():
                doc_id = file_path  # 使用文件路径作为文档ID
                doc = Document(doc_id)
                # 将文件路径存储为keyword字段方便查询（可选）
                doc.add_field(Field("path", file_path, FieldType.KEYWORD))
                # 将文本内容存为TEXT字段建立索引
                doc.add_field(Field("content", text_content, FieldType.TEXT))
                writer.add_document(doc)

    writer.commit()
    print("索引构建完成。")


################################
# Flask简易前端
################################
app = Flask(__name__)

# Add configuration after Flask app initialization
app.config["DATA_DIR"] = DATA_DIR
app.config["ABS_DATA_DIR"] = ABS_DATA_DIR
app.config["INDEX_DIR"] = INDEX_DIR


# 加高亮函数（在Flask分之前添加）
def highlight_text(text: str, query_terms: List[str]) -> str:
    """高亮显示文本中的搜索关键词"""
    if not query_terms or not text:
        return text

    # 将查询词转换为正则模式，忽略大小写
    pattern = "|".join(map(re.escape, query_terms))

    def replace(match):
        return f'<span class="highlight">{match.group(0)}</span>'

    return re.sub(f"({pattern})", replace, text, flags=re.IGNORECASE)


# 更新HTML模板，添加重建索引按钮和元数据显示
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>搜索页面</title>
    <style>
        .result-item { 
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
        }
        .snippet { 
            color: #333; 
            margin: 10px 0;
            line-height: 1.5;
            white-space: pre-line;
        }
        .path { 
            color: #0066cc;
            margin-bottom: 8px;
            cursor: pointer;  /* 添加鼠标指针样式 */
        }
        .path:hover {
            text-decoration: underline;  /* 悬停时添加下划线 */
        }
        .highlight { 
            background-color: #ffeb3b; 
            font-weight: bold;
            padding: 2px;
        }
        .metadata {
            color: #666;
            margin: 10px 0;
            font-size: 0.9em;
        }
        .rebuild-button {
            margin: 10px 0;
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .rebuild-button:hover {
            background-color: #45a049;
        }
        .benchmark {
            margin: 10px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>简易搜索引擎</h1>
    <form method="POST" action="/rebuild_index">
        <button type="submit" class="rebuild-button">重建索引</button>
    </form>
    {% if benchmark_data %}
    <div class="benchmark">
        <h3>索引重建信息：</h3>
        <p>处理文件数：{{ benchmark_data.file_count }}</p>
        <p>索引构建时间：{{ "%.2f"|format(benchmark_data.build_time) }} 秒</p>
        <p>索引大小：{{ benchmark_data.index_size }}</p>
    </div>
    {% endif %}
    <form method="GET" action="/">
        <input type="text" name="q" placeholder="请输入搜索关键词" value="{{query|e}}" style="width: 300px; padding: 5px;">
        <input type="submit" value="搜索">
    </form>
    {% if metadata %}
    <div class="metadata">
        找到 {{ metadata.result_count }} 个结果 (用时 {{ "%.3f"|format(metadata.search_time) }} 秒)
    </div>
    {% endif %}
    {% if results is not none %}
        <h2>搜索结果：</h2>
        {% if results %}
            {% for result in results %}
            <div class="result-item">
                <div class="path" onclick="window.location.href='/raw/{{result.path}}'">
                    📄 {{result.path}}
                </div>
                <div class="snippet">{{result.snippet|safe}}</div>
            </div>
            {% endfor %}
        {% else %}
            <p>没有找到相关结果</p>
        {% endif %}
    {% endif %}
</body>
</html>
"""


# 添加重建索引的路由
@app.route("/rebuild_index", methods=["POST"])
def rebuild_index():
    start_time = time.time()
    file_count = 0

    # 清理现有索引
    import shutil

    if os.path.exists(app.config["INDEX_DIR"]):
        shutil.rmtree(app.config["INDEX_DIR"])
    os.makedirs(app.config["INDEX_DIR"])

    # 重建索引
    build_index_from_directory(app.config["INDEX_DIR"], app.config["DATA_DIR"])

    # 计算索引大小
    index_size = 0
    for root, dirs, files in os.walk(app.config["INDEX_DIR"]):
        for f in files:
            fp = os.path.join(root, f)
            file_count += 1
            index_size += os.path.getsize(fp)

    build_time = time.time() - start_time

    # 格式化索引大小
    if index_size > 1024 * 1024:
        index_size_str = f"{index_size / (1024 * 1024):.2f} MB"
    else:
        index_size_str = f"{index_size / 1024:.2f} KB"

    benchmark_data = {
        "file_count": file_count,
        "build_time": build_time,
        "index_size": index_size_str,
    }

    return render_template_string(HTML_TEMPLATE, benchmark_data=benchmark_data)


# 更新搜索路由以包含元数据
@app.route("/", methods=["GET"])
def search_page():
    query_text = request.args.get("q", "").strip()
    results = None
    metadata = None

    if query_text:
        start_time = time.time()
        analyzer = Analyzer()
        reader = IndexReader(app.config["INDEX_DIR"])
        searcher = Searcher(reader, analyzer)
        query_terms = query_text.split()
        q = Query(query_terms, operator="AND")
        scored_docs = searcher.search_with_scores(q)

        results = []
        for doc_id, score in scored_docs:
            doc = reader.get_document(doc_id)
            if doc:
                content = doc.get("content", "")
                snippet = content[:1000] + "..." if len(content) > 1000 else content
                highlighted_snippet = highlight_text(snippet, query_terms)
                results.append(
                    {
                        "path": doc.get("path", ""),
                        "snippet": highlighted_snippet,
                        "score": score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)

        search_time = time.time() - start_time
        metadata = {"result_count": len(results), "search_time": search_time}

    return render_template_string(
        HTML_TEMPLATE,
        query=query_text,
        results=results,
        metadata=metadata,
        benchmark_data=request.args.get("benchmark_data", None),
    )


@app.route("/raw/<path:filepath>")
def raw_file(filepath):
    # 确保访问的是 DATA_DIR 内的文件
    abs_filepath = os.path.abspath(filepath)
    if not abs_filepath.startswith(app.config["ABS_DATA_DIR"]):
        return "Access Denied", 403
    try:
        return send_file(abs_filepath)
    except Exception as e:
        return str(e), 404


if __name__ == "__main__":
    INDEX_DIR = "puceny_index_cs61a"
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
        # Use config instead of global variable
        build_index_from_directory(INDEX_DIR, app.config["DATA_DIR"])

    # 启动Flask
    app.run(host="0.0.0.0", port=5050, debug=True)
