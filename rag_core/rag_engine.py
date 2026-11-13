"""
RAG 核心引擎

编排层：串联文档加载 → 切分 → 向量化 → 存储 → 检索 → 生成
"""

import hashlib
import json
import logging
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

from .database import DatabaseManager
from .vector_store import VectorStore
from .embedder import Embedder

logger = logging.getLogger(__name__)


def _count_words(text: str) -> int:
    """统计文本词数：中文按字符数，英文按空格分词数。

    采用简单混合策略：提取所有中文字符与英文单词，分别计数后求和。
    """
    if not text:
        return 0
    # 英文/数字单词（含连字符、撇号）
    latin_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
    # 中文字符
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(latin_words) + len(cjk_chars)


class RAGEngine:
    """RAG 核心引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.db = DatabaseManager(config)
        self.vector_store = VectorStore(config)
        emb_cfg = config["embedding"]
        self.embedder = Embedder(
            model_name=emb_cfg["model_name"],
            device=emb_cfg["device"],
            batch_size=emb_cfg.get("batch_size", 32),
        )

        self.db.connect()
        self._init_all_schemas()
        logger.info("RAGEngine 初始化完成")

    def _init_all_schemas(self):
        """启动时一次性初始化所有项目 schema，避免每次摄入重复执行 DDL。"""
        schemas = set()
        for proj_cfg in self.config["projects"].values():
            schemas.add(proj_cfg["schema"])
        for schema_name in schemas:
            try:
                self.db.init_schema(schema_name)
            except Exception as e:
                logger.warning("初始化 schema %s 失败: %s", schema_name, e)

    def _get_collection_name(self, project: str) -> str:
        project_config = self.config["projects"].get(project, self.config["projects"]["default"])
        return project_config["collection"]

    def _get_schema_name(self, project: str) -> str:
        project_config = self.config["projects"].get(project, self.config["projects"]["default"])
        return project_config["schema"]

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def _purge_doc_chunks(self, schema_name: str, collection_name: str, doc_id: str):
        """清理某文档已存在的 chunk 与向量（用于重复摄入前的幂等处理）。

        容错：即使向量库中无对应数据，也继续清理 PG 侧元数据。
        """
        existing = self.db.get_chunks_by_doc(schema_name, doc_id)
        if not existing:
            return
        chroma_ids = [c["chroma_id"] for c in existing]
        try:
            collection = self.vector_store.get_or_create_collection(collection_name)
            collection.delete(ids=chroma_ids)
        except Exception as e:
            logger.warning("幂等清理向量失败（继续清理元数据）: %s", e)
        with self.db.cursor() as cur:
            self.db._set_search_path(cur, schema_name)
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    def _split_text(self, text: str) -> List[str]:
        """简单文本切分"""
        chunk_config = self.config["chunking"]
        chunk_size = chunk_config["chunk_size"]
        chunk_overlap = chunk_config["chunk_overlap"]
        separators = chunk_config["separators"]

        effective_overlap = min(chunk_overlap, max(0, chunk_size - 1))
        max_chunks = max(1, (len(text) // max(1, chunk_size - effective_overlap)) + 2)

        chunks = []
        current = text
        iterations = 0
        while len(current) > chunk_size and iterations < max_chunks:
            iterations += 1
            split_at = chunk_size
            found = False
            for sep in separators:
                pos = current.rfind(sep, max(0, chunk_size - 100), chunk_size + 100)
                if pos > 0:
                    split_at = pos + len(sep)
                    found = True
                    break
            if not found:
                split_at = chunk_size
            chunks.append(current[:split_at].strip())
            current = current[max(0, split_at - effective_overlap):]
            if split_at == 0 or len(current) == 0:
                break
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def ingest_document(
        self,
        content: str,
        title: str = "",
        source: str = "",
        file_type: str = "text",
        project: str = "default",
        metadata: dict = None,
    ) -> dict:
        """
        摄入文档：切分 → 向量化 → 存入 ChromaDB + PostgreSQL

        Returns:
            {"doc_id": ..., "chunk_count": ..., "chroma_ids": [...]}
        """
        schema_name = self._get_schema_name(project)
        collection_name = self._get_collection_name(project)

        # schema 已在引擎启动时初始化，这里仅兜底确保存在（如项目动态新增）
        self.db.ensure_schema(schema_name)

        # 生成文档 ID
        doc_id = self._hash_text(content + title)
        word_count = _count_words(content)

        # 幂等：若该文档已存在，先清理旧 chunk 与向量，避免重复摄入导致数据膨胀
        self._purge_doc_chunks(schema_name, collection_name, doc_id)

        # 存入 PostgreSQL 文档表
        self.db.insert_document(
            schema_name, doc_id, title, source, file_type,
            word_count, metadata,
        )

        # 文本切分
        chunks = self._split_text(content)
        if not chunks:
            chunks = [content]

        # 批量向量化
        embeddings = self.embedder.encode_documents(chunks)
        chunk_ids = []
        chroma_ids = []

        # 组装 chunk 数据并批量插入（避免逐条取连接/提交）
        chunk_rows = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chroma_id = str(chunk_id)
            chunk_rows.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "content": chunk_text,
                "token_count": _count_words(chunk_text),
                "chroma_id": chroma_id,
            })
            chroma_ids.append(chroma_id)
            chunk_ids.append(chunk_id)

        self.db.insert_chunks_bulk(schema_name, chunk_rows)

        # 批量写入 ChromaDB
        metadatas = [{
            "doc_id": doc_id,
            "chunk_index": i,
            "title": title,
            "source": source,
        } for i in range(len(chunks))]

        self.vector_store.add(
            collection_name=collection_name,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=chroma_ids,
        )

        logger.info("Ingest complete: doc=%s, chunks=%d, project=%s",
                     title or doc_id, len(chunks), project)

        return {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "chroma_ids": chroma_ids,
        }

    def query(
        self,
        query_text: str,
        project: str = "default",
        top_k: int = None,
        threshold: float = None,
        metadata_filter: dict = None,
    ) -> List[dict]:
        """检索相关文档块"""
        if top_k is None:
            top_k = self.config["retrieval"]["top_k"]
        if top_k < 1:
            return []
        if threshold is None:
            threshold = self.config["retrieval"]["similarity_threshold"]

        collection_name = self._get_collection_name(project)

        query_vec = self.embedder.encode_query(query_text)

        results = self.vector_store.query(
            collection_name=collection_name,
            query_embedding=query_vec.tolist(),
            n_results=top_k,
            where=metadata_filter,
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, chroma_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] and len(results["distances"][0]) > i else 0
                similarity = max(0.0, 1 - distance)

                if similarity < threshold:
                    continue

                doc_text = ""
                if results["documents"] and results["documents"][0] and i < len(results["documents"][0]):
                    doc_text = results["documents"][0][i]

                meta = {}
                if results["metadatas"] and results["metadatas"][0] and i < len(results["metadatas"][0]):
                    meta = results["metadatas"][0][i] or {}

                items.append({
                    "chunk_id": chroma_id,
                    "content": doc_text,
                    "score": round(similarity, 4),
                    "metadata": meta,
                })

        return items

    def delete_document(self, doc_id: str, project: str = "default"):
        """删除文档及其所有切块"""
        schema_name = self._get_schema_name(project)
        collection_name = self._get_collection_name(project)

        with self.db.cursor() as cur:
            self.db._set_search_path(cur, schema_name)
            cur.execute("SELECT chroma_id FROM chunks WHERE doc_id = %s", (doc_id,))
            chroma_ids = [row["chroma_id"] for row in cur.fetchall()]

        if not chroma_ids:
            with self.db.cursor() as cur:
                self.db._set_search_path(cur, schema_name)
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            self._cleanup_wordcloud(doc_id)
            logger.info("Deleted document (no chunks): %s", doc_id)
            return

        collection = self.vector_store.get_or_create_collection(collection_name)
        collection.delete(ids=chroma_ids)

        with self.db.cursor() as cur:
            self.db._set_search_path(cur, schema_name)
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

        self._cleanup_wordcloud(doc_id)
        logger.info("Deleted document: %s (%d chunks)", doc_id, len(chroma_ids))

    # ════════════════════════════════════════════════════════
    # 文档元数据与内容管理
    # ════════════════════════════════════════════════════════

    def _cleanup_wordcloud(self, doc_id: str):
        """Clean up wordcloud image for a deleted document"""
        import os as _os
        wc_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "static", "wordclouds")
        wc_path = _os.path.join(wc_dir, f"{doc_id}.png")
        if _os.path.isfile(wc_path):
            try:
                _os.remove(wc_path)
                logger.info("Cleaned wordcloud for deleted doc: %s", doc_id)
            except Exception as e:
                logger.warning("Failed to clean wordcloud %s: %s", doc_id, e)
    def get_document_info(self, schema_name: str, doc_id: str) -> Optional[dict]:
        doc = self.db.get_document(schema_name, doc_id)
        if doc:
            doc = dict(doc)
            cats = self.db.get_categories_for_doc(schema_name, doc_id)
            doc["categories"] = [dict(c) for c in cats]
        return doc

    def update_document_metadata(self, schema_name: str, doc_id: str, updates: dict):
        self.db.update_document_metadata(schema_name, doc_id, updates)

    def update_document_content(
        self, schema_name: str, doc_id: str, new_text: str,
        title: str = None, project: str = "default",
    ):
        """更新文档内容：重新切分 + 重新向量化"""
        collection_name = self._get_collection_name(project)

        old_chunks = self.db.get_chunks_by_doc(schema_name, doc_id)
        old_chroma_ids = [c["chroma_id"] for c in old_chunks]
        if old_chroma_ids:
            try:
                collection = self.vector_store.get_or_create_collection(collection_name)
                collection.delete(ids=old_chroma_ids)
            except Exception as e:
                logger.warning("Clean old vectors failed: %s", e)

        with self.db.cursor() as cur:
            self.db._set_search_path(cur, schema_name)
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

        if title:
            with self.db.cursor() as cur:
                self.db._set_search_path(cur, schema_name)
                cur.execute(
                    "UPDATE documents SET title = %s, updated_at = NOW() WHERE doc_id = %s",
                    (title, doc_id),
                )

        chunks = self._split_text(new_text)
        if not chunks:
            chunks = [new_text]

        embeddings = self.embedder.encode_documents(chunks)
        chroma_ids = []

        chunk_rows = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chroma_id = str(chunk_id)
            chunk_rows.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "content": chunk_text,
                "token_count": _count_words(chunk_text),
                "chroma_id": chroma_id,
            })
            chroma_ids.append(chroma_id)

        self.db.insert_chunks_bulk(schema_name, chunk_rows)

        metadatas = [{
            "doc_id": doc_id,
            "chunk_index": i,
            "title": title or "",
            "source": "",
        } for i in range(len(chunks))]

        self.vector_store.add(
            collection_name=collection_name,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=chroma_ids,
        )

        with self.db.cursor() as cur:
            self.db._set_search_path(cur, schema_name)
            cur.execute(
                "UPDATE documents SET word_count = %s, updated_at = NOW() WHERE doc_id = %s",
                (_count_words(new_text), doc_id),
            )

        logger.info("Content updated: doc=%s, chunks=%d", doc_id, len(chunks))

    def search_documents(self, schema_name: str, keyword: str = None,
                         file_type: str = None, limit: int = 20) -> list:
        return self.db.search_documents(schema_name, keyword, file_type, limit)

    # ════════════════════════════════════════════════════════
    # 分类管理
    # ════════════════════════════════════════════════════════

    def get_categories(self, schema_name: str) -> list:
        return self.db.get_categories(schema_name)

    def create_category(self, schema_name: str, name: str,
                        parent_id: int = None, color: str = None,
                        project: str = "default") -> int:
        return self.db.create_category(schema_name, name, parent_id, project, color or "#3b82f6")

    def delete_category(self, schema_name: str, category_id: int):
        self.db.delete_category(schema_name, category_id)

    def update_category(self, schema_name: str, category_id: int,
                        name: str = None, parent_id: int = None, color: str = None):
        self.db.update_category(schema_name, category_id, name, parent_id, color)

    # ════════════════════════════════════════════════════════
    # 文档-分类关联
    # ════════════════════════════════════════════════════════

    def add_doc_to_category(self, schema_name: str, doc_id: str, category_id: int):
        self.db.add_doc_to_category(schema_name, doc_id, category_id)

    def remove_doc_from_category(self, schema_name: str, doc_id: str, category_id: int):
        self.db.remove_doc_from_category(schema_name, doc_id, category_id)

    def get_categories_for_doc(self, schema_name: str, doc_id: str) -> list:
        return self.db.get_categories_for_doc(schema_name, doc_id)

    def get_docs_by_category(self, schema_name: str, category_id: int, limit: int = 200) -> list:
        return self.db.get_docs_by_category(schema_name, category_id, limit)

    def get_all_uncategorized_docs(self, schema_name: str, limit: int = 200) -> list:
        return self.db.get_all_uncategorized_docs(schema_name, limit)

    # ════════════════════════════════════════════════════════
    # AI 摘要
    # ════════════════════════════════════════════════════════

    def ai_summarize(self, content: str, llm_client, title: str = "") -> str:
        if not llm_client or not llm_client.enabled:
            logger.warning("LLM not enabled, skip summarization")
            return ""

        system_prompt = (
            "你是一个文档分析助手。请为以下文档生成信息丰富的中文摘要。\n"
            "要求：\n"
            "1. 摘要长度控制在 400 字以内\n"
            "2. 充分保留关键信息、核心观点、数据、结论\n"
            "3. 语言流畅、客观，不要过度精简\n"
            "4. 使用中文\n"
            "5. 只返回摘要本身，不要额外的说明"
        )

        user_content = content[:12000]
        if title:
            user_content = f"标题：{title}\n\n正文：{user_content}"

        try:
            summary = llm_client.chat(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=system_prompt,
                max_tokens=500,
            )
            return summary.strip()
        except Exception as e:
            logger.error("LLM summarization failed: %s", e)
            return ""

    # ════════════════════════════════════════════════════════
    # 爬取 + 摄入一体化
    # ════════════════════════════════════════════════════════

    @staticmethod
    def crawl_and_ingest(
        engine: "RAGEngine", crawler, source: dict,
        project: str = "default", llm_client=None,
    ) -> List[dict]:
        results = []
        articles = crawler.crawl_source_advanced(source)
        logger.info("Crawled %s: %d articles", source["name"], len(articles))

        for article in articles:
            if not article.get("content") or len(article["content"].strip()) < 20:
                continue

            content = article["content"]
            meta = {
                "source_category": source.get("category", "通用"),
                "source_name": source.get("name", ""),
                "crawled_url": article.get("url", ""),
                "content_type": article.get("content_type", ""),
                "quality": article.get("quality", ""),
            }
            if article.get("tags"):
                meta["auto_tags"] = article["tags"]
            if article.get("entities"):
                meta["entities"] = article["entities"]

            ingest_result = engine.ingest_document(
                content=content,
                title=article.get("title", ""),
                source=article.get("url", ""),
                file_type="crawled",
                project=project,
                metadata=meta,
            )

            # Save tags to document metadata
            if article.get("tags"):
                try:
                    engine.db.update_document_metadata(
                        engine._get_schema_name(project),
                        ingest_result["doc_id"],
                        {"tags": article["tags"]},
                    )
                except Exception as e:
                    logger.warning("Save tags failed: %s", e)

            # AI summary (manual trigger only, not auto)
            ingest_result["title"] = article.get("title", "")
            ingest_result["content_type"] = article.get("content_type", "")
            ingest_result["quality"] = article.get("quality", "")
            ingest_result["tags"] = article.get("tags", [])
            results.append(ingest_result)

        return results

    def close(self):
        self.db.close()
        self.vector_store.close()
