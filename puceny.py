import json
import os
import re
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional


class FieldType(Enum):
    TEXT = "text"  # 表示需要进行分词的文本字段
    KEYWORD = "keyword"  # 表示不分词的关键词字段（如ID、分类标签）
    STORED = "stored"  # 表示需要原始存储的字段（可用于原文检索）


class Field:
    """
    表示文档的一个字段:
    - name: 字段名
    - value: 字段值（字符串、数值等）
    - field_type: 字段类型（TEXT、KEYWORD、STORED）
    """

    def __init__(self, name: str, value: Any, field_type: FieldType = FieldType.TEXT):
        self.name = name
        self.value = value
        self.field_type = field_type

    def __repr__(self):
        return f"Field(name={self.name}, value={self.value}, type={self.field_type})"


class Document:
    """
    表示一个文档，由多个字段组成。
    - doc_id: 文档唯一标识(可用整数或字符串)
    - fields: 字段列表
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.fields: List[Field] = []

    def add_field(self, field: Field):
        self.fields.append(field)

    def get_field(self, name: str) -> Optional[Field]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def __repr__(self):
        return f"Document(doc_id={self.doc_id}, fields={self.fields})"


class Token:
    """
    表示分词后的基本元Token。
    token_str: 原始Token字符串
    """

    def __init__(self, token_str: str):
        self.token_str = token_str

    def __repr__(self):
        return f"Token({self.token_str})"


class Tokenizer:
    """
    基本分词器：
    使用正则表达式将文本按非字母数字字符分割，
    只保留英文字符和数字。
    """

    def tokenize(self, text: str) -> List[Token]:
        # 使用正则按非字母数字字符分割
        raw_tokens = re.split(r"\W+", text)
        # 过滤空字符串
        raw_tokens = [t for t in raw_tokens if t.strip() != ""]
        return [Token(t) for t in raw_tokens]


class LowercaseFilter:
    """
    小写过滤器，将所有 Token 转为小写。
    """

    def filter(self, tokens: List[Token]) -> List[Token]:
        return [Token(t.token_str.lower()) for t in tokens]


class StopwordFilter:
    """
    停用词过滤器，移除常见停用词（可根据需要扩展）。
    """

    def __init__(self, stopwords: List[str]):
        # 确保停用词是小写，用于与小写化后的token对比
        self.stopwords = set(word.lower() for word in stopwords)

    def filter(self, tokens: List[Token]) -> List[Token]:
        return [t for t in tokens if t.token_str.lower() not in self.stopwords]


class Analyzer:
    """
    分析器，将文本经过一系列步骤处理成标准化 Token 列表。
    默认流程：分词 -> 小写化 -> 停用词过滤
    可根据需要新增/修改处理步骤。
    """

    def __init__(self, stopwords: List[str] = None):
        self.tokenizer = Tokenizer()
        if stopwords is None:
            # 简单停用词列表示例，可根据需要扩展
            stopwords = ["the", "is", "a", "an", "of", "for", "and", "to", "in"]
        self.lower_filter = LowercaseFilter()
        self.stop_filter = StopwordFilter(stopwords)

    def analyze(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        tokens = self.lower_filter.filter(tokens)
        tokens = self.stop_filter.filter(tokens)
        return [t.token_str for t in tokens]


class IndexWriter:
    def __init__(self, index_dir: str, analyzer: Analyzer):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.analyzer = analyzer
        self.inverted_index: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        self.document_store: Dict[str, Dict[str, Any]] = {}
        self.doc_count = 0
        self.total_docs = 0  # Track total documents for progress

        self._load_segments_info()

    def _load_segments_info(self):
        self.segments_file = os.path.join(self.index_dir, "segments.json")
        if os.path.exists(self.segments_file):
            with open(self.segments_file, "r", encoding="utf-8") as f:
                self.segments_info = json.load(f)
        else:
            self.segments_info = {"segments": []}
        # 根据已有段数确定segment计数器
        self.segment_count = len(self.segments_info["segments"])

    def _save_segments_info(self):
        with open(self.segments_file, "w", encoding="utf-8") as f:
            json.dump(self.segments_info, f, ensure_ascii=False, indent=2)

    def add_document(self, doc: Document):
        doc_id = doc.doc_id
        self.doc_count += 1
        doc_fields_data = {}

        for field in doc.fields:
            doc_fields_data[field.name] = field.value
            if field.field_type == FieldType.TEXT:
                tokens = self.analyzer.analyze(field.value)
                for pos, token in enumerate(tokens):
                    if doc_id not in self.inverted_index[token]:
                        self.inverted_index[token][doc_id] = []
                    self.inverted_index[token][doc_id].append(pos)

            elif field.field_type == FieldType.KEYWORD:
                token = field.value.lower()
                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = []
                self.inverted_index[token][doc_id].append(0)

            # STORED 不建立倒排索引
        self.document_store[doc_id] = doc_fields_data

    def add_documents(self, documents: List[Document], show_progress: bool = True):
        """Batch add documents with optional progress bar"""
        self.total_docs = len(documents)
        for i, doc in enumerate(documents, 1):
            self.add_document(doc)
            if show_progress:
                self._update_progress(i, self.total_docs)

    def _update_progress(self, current: int, total: int):
        """Display progress bar in console"""
        bar_width = 50
        progress = current / total
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        percent = progress * 100
        print(
            f"\rProcessing documents: [{bar}] {percent:.1f}% ({current}/{total})",
            end="",
        )
        if current == total:
            print()  # New line when complete

    def commit(self):
        print("\nCommitting changes...")
        # 每次commit创建新的段目录
        segment_name = f"segment_{self.segment_count:03d}"
        segment_dir = os.path.join(self.index_dir, segment_name)
        os.makedirs(segment_dir, exist_ok=True)

        inverted_index_path = os.path.join(segment_dir, "inverted_index.json")
        doc_store_path = os.path.join(segment_dir, "document_store.json")

        normal_index = {
            term: dict(postings) for term, postings in self.inverted_index.items()
        }

        with open(inverted_index_path, "w", encoding="utf-8") as f:
            json.dump(normal_index, f, ensure_ascii=False, indent=2)

        with open(doc_store_path, "w", encoding="utf-8") as f:
            json.dump(self.document_store, f, ensure_ascii=False, indent=2)

        # 更新segments信息
        self.segments_info["segments"].append(
            {"name": segment_name, "doc_count": self.doc_count}
        )
        self._save_segments_info()

        # 清空内存索引
        self.inverted_index.clear()
        self.document_store.clear()
        self.doc_count = 0
        self.segment_count += 1

        print(f"已提交新段: {segment_name}")


class IndexReader:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        segments_file = os.path.join(self.index_dir, "segments.json")
        with open(segments_file, "r", encoding="utf-8") as f:
            self.segments_info = json.load(f)
        self.segments = self.segments_info["segments"]

        # 用于一次性加载所有段数据的结构
        self.inverted_index: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        self.doc_store: Dict[str, Dict[str, Any]] = {}

        # 加载所有段的数据到内存
        for seg in self.segments:
            seg_name = seg["name"]
            segment_dir = os.path.join(self.index_dir, seg_name)
            inv_idx_path = os.path.join(segment_dir, "inverted_index.json")
            doc_store_path = os.path.join(segment_dir, "document_store.json")

            if os.path.exists(inv_idx_path):
                with open(inv_idx_path, "r", encoding="utf-8") as f:
                    inv_idx = json.load(f)
                for term, postings in inv_idx.items():
                    for doc_id, positions in postings.items():
                        if doc_id not in self.inverted_index[term]:
                            self.inverted_index[term][doc_id] = positions
                        else:
                            self.inverted_index[term][doc_id].extend(positions)

            if os.path.exists(doc_store_path):
                with open(doc_store_path, "r", encoding="utf-8") as f:
                    ds = json.load(f)
                self.doc_store.update(ds)

        self.total_doc_count = len(self.doc_store)

        # 预计算每个term的doc_freq
        self.term_doc_freq: Dict[str, int] = {
            term: len(postings) for term, postings in self.inverted_index.items()
        }

    def terms_docs(self, term: str) -> Dict[str, List[int]]:
        # 现在直接从内存返回，无需文件I/O
        return self.inverted_index.get(term, {})

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        return self.doc_store.get(doc_id, {})


class Query:
    # 简单的Term查询
    def __init__(self, terms: List[str], operator="OR"):
        self.terms = terms
        self.operator = operator.upper()  # OR或AND


class Searcher:
    def __init__(self, reader: IndexReader, analyzer: Analyzer):
        self.reader = reader
        self.analyzer = analyzer

    def search_with_scores(self, query: Query) -> List[tuple[str, float]]:
        # 对查询词进行分析
        normalized_terms = []
        for t in query.terms:
            analyzed = self.analyzer.analyze(t)
            normalized_terms.extend(analyzed)

        if not normalized_terms:
            return []

        doc_scores: Dict[str, float] = defaultdict(float)
        total_docs = self.reader.total_doc_count

        for term in normalized_terms:
            term_docs = self.reader.terms_docs(term)  # 现在是内存访问
            if term_docs:
                doc_freq = self.reader.term_doc_freq.get(term, 0)
                # 根据doc_freq和total_docs计算IDF
                # （这里的idf计算公式可以随需要调整）
                idf = 1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5)

                for doc_id, positions in term_docs.items():
                    tf = len(positions)
                    score = tf * idf
                    doc_scores[doc_id] += score

        # AND逻辑过滤
        if query.operator == "AND":
            for doc_id in list(doc_scores.keys()):
                # 确保该doc_id包含所有查询词
                for term in normalized_terms:
                    if doc_id not in self.reader.terms_docs(term):
                        del doc_scores[doc_id]
                        break

        scored_docs = [(doc_id, score) for doc_id, score in doc_scores.items()]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    def search(self, query: Query) -> List[str]:
        # 保持原有的search方法，但使用search_with_scores实现
        scored_docs = self.search_with_scores(query)
        return [doc_id for doc_id, _ in scored_docs]


class IndexMerger:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir

    def merge_all(self, new_segment_name="merged_segment"):
        segments_file = os.path.join(self.index_dir, "segments.json")
        with open(segments_file, "r", encoding="utf-8") as f:
            segments_info = json.load(f)

        segments = segments_info["segments"]
        if len(segments) <= 1:
            print("无需合并，段数量为1或0。")
            return

        # 读入所有段的索引
        merged_inverted_index: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        merged_doc_store: Dict[str, Dict[str, Any]] = {}

        for seg in segments:
            seg_name = seg["name"]
            segment_dir = os.path.join(self.index_dir, seg_name)
            inv_idx_path = os.path.join(segment_dir, "inverted_index.json")
            doc_store_path = os.path.join(segment_dir, "document_store.json")

            if not os.path.exists(inv_idx_path) or not os.path.exists(doc_store_path):
                continue

            with open(inv_idx_path, "r", encoding="utf-8") as f:
                inv_idx = json.load(f)
            with open(doc_store_path, "r", encoding="utf-8") as f:
                doc_store = json.load(f)

            # 合并倒排索引
            for term, postings in inv_idx.items():
                for doc_id, positions in postings.items():
                    if doc_id not in merged_inverted_index[term]:
                        merged_inverted_index[term][doc_id] = positions
                    else:
                        merged_inverted_index[term][doc_id].extend(positions)

            # 合并文档存储（doc_id独立分布于各段）
            for doc_id, fields in doc_store.items():
                if doc_id not in merged_doc_store:
                    merged_doc_store[doc_id] = fields
                else:
                    merged_doc_store[doc_id].update(fields)

        # 写入新段
        new_segment_dir = os.path.join(self.index_dir, new_segment_name)
        os.makedirs(new_segment_dir, exist_ok=True)
        with open(
            os.path.join(new_segment_dir, "inverted_index.json"), "w", encoding="utf-8"
        ) as f:
            normal_index = {t: dict(p) for t, p in merged_inverted_index.items()}
            json.dump(normal_index, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(new_segment_dir, "document_store.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(merged_doc_store, f, ensure_ascii=False, indent=2)

        # 更新segments文件, 只保留新合并段
        new_segments_info = {
            "segments": [{"name": new_segment_name, "doc_count": len(merged_doc_store)}]
        }
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(new_segments_info, f, ensure_ascii=False, indent=2)

        # 删除旧段目录
        for seg in segments:
            seg_name = seg["name"]
            seg_dir = os.path.join(self.index_dir, seg_name)
            if os.path.exists(seg_dir):
                for root, dirs, files in os.walk(seg_dir, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    os.rmdir(root)

        print(f"合并完成，新段：{new_segment_name}")


# 以下为简单的测试示例（可根据需要删除或修改）
if __name__ == "__main__":
    analyzer = Analyzer()
    writer = IndexWriter("puceny_index", analyzer)
    doc1 = Document("1")
    doc1.add_field(Field("title", "Lucene in Action", FieldType.TEXT))
    doc1.add_field(Field("author", "Erik Hatcher", FieldType.KEYWORD))
    doc1.add_field(
        Field(
            "content",
            "Lucene is a powerful Java library used for implementing search.",
            FieldType.TEXT,
        )
    )
    writer.add_document(doc1)

    doc2 = Document("2")
    doc2.add_field(Field("title", "Learning Python", FieldType.TEXT))
    doc2.add_field(Field("author", "Mark Lutz", FieldType.KEYWORD))
    doc2.add_field(
        Field("content", "Python is easy to learn and powerful.", FieldType.TEXT)
    )
    writer.add_document(doc2)
    writer.commit()

    reader = IndexReader("puceny_index")
    searcher = Searcher(reader, analyzer)
    q = Query(["Lucene", "python"], operator="OR")
    results = searcher.search(q)
    print("查询结果文档ID列表：", results)
    for doc_id in results:
        print("DocID:", doc_id, "原文:", reader.get_document(doc_id))
