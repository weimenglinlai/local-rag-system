"""
ChromaDB 向量存储层

架构说明：
- ChromaDB 负责向量存储与 ANN 检索（HNSW 索引）
- PostgreSQL 负责元数据与结构化查询
- 通过 chroma_id 关联两边的数据

多项目隔离：
- 每个项目使用独立 Collection
- Collection 命名：{project}_{namespace}
"""

import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储管理器"""

    def __init__(self, config: dict):
        persist_dir = config["chromadb"]["persist_directory"]
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: Dict[str, Any] = {}
        logger.info("ChromaDB 初始化完成: %s", persist_dir)

    def get_or_create_collection(self, name: str, dimension: int = 1024):
        """获取或创建 collection"""
        if name in self._collections:
            return self._collections[name]

        try:
            collection = self._client.get_collection(name)
            logger.info("使用已有 collection: %s", name)
        except Exception:
            # get_collection raises on truly missing collection; other errors bubble up below
            try:
                collection = self._client.create_collection(
                    name=name,
                    metadata={
                        "dimension": dimension,
                        "hnsw:space": "cosine",
                    },
                )
                logger.info("创建 collection: %s (dim=%d)", name, dimension)
            except Exception:
                # Race condition: another thread created it between get and create
                collection = self._client.get_collection(name)
                logger.info("collection 已存在(race): %s", name)

        self._collections[name] = collection
        return collection

    def add(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict] = None,
        ids: List[str] = None,
    ) -> List[str]:
        """添加向量和文档到 collection"""
        collection = self.get_or_create_collection(collection_name)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{} for _ in embeddings],
            ids=ids,
        )
        return ids

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5,
        where: dict = None,
        where_document: dict = None,
    ) -> dict:
        """
        向量检索

        Args:
            collection_name: collection 名称
            query_embedding: 查询向量
            n_results: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤

        Returns:
            {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

    def delete_collection(self, name: str):
        """删除 collection"""
        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
            logger.info("已删除 collection: %s", name)
        except Exception as e:
            logger.warning("删除 collection 失败: %s", e)

    def count(self, collection_name: str) -> int:
        """返回 collection 中的文档数量"""
        try:
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def list_collections(self) -> List[str]:
        """列出所有 collection"""
        return [c.name for c in self._client.list_collections()]

    def get_by_ids(self, collection_name: str, ids: List[str]) -> dict:
        """按 ID 获取文档"""
        collection = self.get_or_create_collection(collection_name)
        return collection.get(ids=ids)

    def close(self):
        """关闭 ChromaDB 客户端"""
        self._collections.clear()
        self._client = None
