"""
PostgreSQL 数据库管理器（线程安全版本）
- 多项目隔离：每个项目独立 schema
- 元数据存储：文档来源、切分信息、时间戳等
- SQL + 向量混合查询能力
- 使用 ThreadedConnectionPool 保证多线程并发安全
"""

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2 import sql
from contextlib import contextmanager
from typing import Optional, Any
import threading
import logging
import re

logger = logging.getLogger(__name__)

# schema 名只允许小写字母、数字、下划线，且不能以数字开头（防止 SQL 注入）
_SCHEMA_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _quote_schema(schema_name: str) -> sql.Composable:
    """校验并安全引用 schema 名，防止 SQL 注入。

    非法 schema 名抛出 ValueError，避免被拼接到 SQL 中。
    """
    if not schema_name or not _SCHEMA_NAME_RE.match(schema_name):
        raise ValueError(f"非法 schema 名: {schema_name!r}")
    return sql.Identifier(schema_name)


class DatabaseManager:
    """PostgreSQL 连接与 schema 管理（线程安全）"""

    def __init__(self, config: dict):
        self.host = config["postgresql"]["host"]
        self.port = config["postgresql"]["port"]
        self.user = config["postgresql"]["user"]
        self.password = config["postgresql"]["password"]
        self.database = config["postgresql"]["database"]
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} "
            f"dbname={self.database} user={self.user} password={self.password}"
        )

    def connect(self):
        """建立连接池（2-8 连接）"""
        self._pool = psycopg2.pool.ThreadedConnectionPool(2, 8, self.dsn)
        logger.info("PostgreSQL pool created: %s:%s/%s", self.host, self.port, self.database)

    def close(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None

    @contextmanager
    def cursor(self):
        """从连接池获取连接，返回游标（线程安全）"""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        conn = self._pool.getconn()
        conn.autocommit = True
        cur = None
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield cur
        except Exception:
            logger.warning("DB cursor error, returning connection to pool", exc_info=True)
            raise
        finally:
            if cur is not None and not cur.closed:
                cur.close()
            try:
                self._pool.putconn(conn)
            except Exception:
                logger.error("Failed to return connection to pool", exc_info=True)

    @staticmethod
    def _set_search_path(cur, schema_name: str):
        """安全设置 search_path（schema 名经校验与标识符引用，防注入）"""
        cur.execute(sql.SQL("SET search_path TO {}").format(_quote_schema(schema_name)))

    def ensure_schema(self, schema_name: str):
        """确保 schema 存在，不存在则创建"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            if not cur.fetchone():
                cur.execute(sql.SQL("CREATE SCHEMA {}").format(_quote_schema(schema_name)))
                logger.info("Created schema: %s", schema_name)

    def init_schema(self, schema_name: str):
        """初始化项目 schema：创建文档表、切块表、分类表和文档-分类关联表"""
        self.ensure_schema(schema_name)
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(64) UNIQUE NOT NULL,
                    title VARCHAR(512),
                    source VARCHAR(1024),
                    file_type VARCHAR(32),
                    word_count INTEGER,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(64) UNIQUE NOT NULL,
                    doc_id VARCHAR(64) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER,
                    chroma_id VARCHAR(64),
                    created_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(256) NOT NULL,
                    parent_id INTEGER DEFAULT NULL REFERENCES categories(id) ON DELETE SET NULL,
                    project VARCHAR(64) DEFAULT 'default',
                    color VARCHAR(16) DEFAULT '#3b82f6',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_categories (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(64) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
                    added_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(doc_id, category_id)
                )
            """)

            _extra_cols = [
                ("description", "TEXT DEFAULT ''"),
                ("tags", "TEXT[] DEFAULT '{}'"),
                ("summary", "TEXT DEFAULT ''"),
                ("source_url", "TEXT DEFAULT ''"),
            ]
            for col_name, col_def in _extra_cols:
                cur.execute(
                    f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS {col_name} {col_def}"
                )

            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chroma_id ON chunks(chroma_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_cats_doc_id ON document_categories(doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_cats_cat_id ON document_categories(category_id)")

            logger.info("Schema %s initialized", schema_name)

    # ── 文档操作 ────────────────────────────────────────

    def insert_document(
        self, schema_name: str, doc_id: str, title: str, source: str,
        file_type: str, word_count: int, metadata: dict = None
    ) -> int:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                """INSERT INTO documents (doc_id, title, source, file_type, word_count, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (doc_id) DO UPDATE SET
                     title=EXCLUDED.title, updated_at=NOW()
                   RETURNING id""",
                (doc_id, title, source, file_type, word_count, psycopg2.extras.Json(metadata or {})),
            )
            return cur.fetchone()["id"]

    def insert_chunk(
        self, schema_name: str, chunk_id: str, doc_id: str,
        chunk_index: int, content: str, token_count: int,
        chroma_id: str, metadata: dict = None
    ):
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                """INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, token_count, chroma_id, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                (chunk_id, doc_id, chunk_index, content, token_count,
                 chroma_id, psycopg2.extras.Json(metadata or {})),
            )

    def insert_chunks_bulk(self, schema_name: str, chunks: list):
        """批量插入多个 chunk，避免逐条取连接/提交。

        Args:
            chunks: list of dict，每个含 chunk_id/doc_id/chunk_index/content/
                    token_count/chroma_id/metadata 键。
        """
        if not chunks:
            return
        rows = []
        for c in chunks:
            rows.append((
                c["chunk_id"], c["doc_id"], c["chunk_index"], c["content"],
                c["token_count"], c["chroma_id"],
                psycopg2.extras.Json(c.get("metadata") or {}),
            ))
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, token_count, chroma_id, metadata)
                   VALUES %s
                   ON CONFLICT (chunk_id) DO NOTHING""",
                rows,
            )

    def get_chunks_by_doc(self, schema_name: str, doc_id: str) -> list:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                "SELECT * FROM chunks WHERE doc_id = %s ORDER BY chunk_index",
                (doc_id,),
            )
            return cur.fetchall()

    def get_document(self, schema_name: str, doc_id: str) -> Optional[dict]:
        """获取单个文档元数据（含分类名）"""
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                "SELECT d.* FROM documents d WHERE d.doc_id = %s",
                (doc_id,),
            )
            return cur.fetchone()

    def search_documents(self, schema_name: str, keyword: str = None,
                         file_type: str = None, limit: int = 20) -> list:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            if keyword:
                query += " AND (title ILIKE %s OR source ILIKE %s)"
                kw = f"%{keyword}%"
                params.extend([kw, kw])
            if file_type:
                query += " AND file_type = %s"
                params.append(file_type)
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            return cur.fetchall()

    # ── 分类操作 ────────────────────────────────────────

    def get_categories(self, schema_name: str) -> list:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute("""
                SELECT c.*,
                    (SELECT COUNT(*) FROM document_categories dc WHERE dc.category_id = c.id) AS doc_count
                FROM categories c ORDER BY c.name
            """)
            return cur.fetchall()

    def create_category(
        self, schema_name: str, name: str,
        parent_id: int = None, project: str = "default",
        color: str = "#3b82f6"
    ) -> int:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                "INSERT INTO categories (name, parent_id, project, color) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (name, parent_id, project, color),
            )
            row = cur.fetchone()
            new_id = row["id"]
            logger.info("Created category: %s (id=%d)", name, new_id)
            return new_id

    def delete_category(self, schema_name: str, category_id: int):
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                "UPDATE categories SET parent_id = NULL WHERE parent_id = %s",
                (category_id,),
            )
            cur.execute(
                "DELETE FROM categories WHERE id = %s",
                (category_id,),
            )
            logger.info("Deleted category: id=%d", category_id)

    def update_category(
        self, schema_name: str, category_id: int,
        name: str = None, parent_id: int = None, color: str = None
    ):
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            updates = []
            params = []
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if parent_id is not None:
                updates.append("parent_id = %s")
                params.append(parent_id)
            if color is not None:
                updates.append("color = %s")
                params.append(color)
            if not updates:
                return
            params.append(category_id)
            cur.execute(
                f"UPDATE categories SET {', '.join(updates)} WHERE id = %s",
                params,
            )
            logger.info("Updated category: id=%d", category_id)

    # ── 文档-分类关联操作 ────────────────────────────────────────

    def add_doc_to_category(self, schema_name: str, doc_id: str, category_id: int):
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                """INSERT INTO document_categories (doc_id, category_id)
                   VALUES (%s, %s) ON CONFLICT (doc_id, category_id) DO NOTHING""",
                (doc_id, category_id),
            )

    def remove_doc_from_category(self, schema_name: str, doc_id: str, category_id: int):
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                "DELETE FROM document_categories WHERE doc_id = %s AND category_id = %s",
                (doc_id, category_id),
            )

    def remove_doc_from_all_categories(self, schema_name: str, doc_id: str):
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                "DELETE FROM document_categories WHERE doc_id = %s",
                (doc_id,),
            )

    def get_categories_for_doc(self, schema_name: str, doc_id: str) -> list:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                """SELECT c.* FROM categories c
                   JOIN document_categories dc ON c.id = dc.category_id
                   WHERE dc.doc_id = %s ORDER BY c.name""",
                (doc_id,),
            )
            return cur.fetchall()

    def get_docs_by_category(self, schema_name: str, category_id: int, limit: int = 200) -> list:
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                """SELECT d.* FROM documents d
                   JOIN document_categories dc ON d.doc_id = dc.doc_id
                   WHERE dc.category_id = %s
                   ORDER BY d.created_at DESC LIMIT %s""",
                (category_id, limit),
            )
            return cur.fetchall()

    def get_all_uncategorized_docs(self, schema_name: str, limit: int = 200) -> list:
        """获取所有未分类的文档"""
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(
                """SELECT d.* FROM documents d
                   WHERE NOT EXISTS (
                     SELECT 1 FROM document_categories dc WHERE dc.doc_id = d.doc_id
                   )
                   ORDER BY d.created_at DESC LIMIT %s""",
                (limit,),
            )
            return cur.fetchall()

    # ── 文档元数据更新 ────────────────────────────────────────

    def update_document_metadata(self, schema_name: str, doc_id: str, updates: dict):
        """
        更新文档的描述/标签/摘要/来源URL等字段
        """
        allowed_fields = {"description", "tags", "summary", "source_url", "title"}
        set_clauses = []
        params = []
        for field, value in updates.items():
            if field not in allowed_fields:
                continue
            if field == "tags":
                set_clauses.append(f"{field} = %s::TEXT[]")
            else:
                set_clauses.append(f"{field} = %s")
            params.append(value)
        if not set_clauses:
            return
        set_clauses.append("updated_at = NOW()")
        sql = f"UPDATE documents SET {', '.join(set_clauses)} WHERE doc_id = %s"
        params.append(doc_id)
        with self.cursor() as cur:
            self._set_search_path(cur, schema_name)
            cur.execute(sql, params)
