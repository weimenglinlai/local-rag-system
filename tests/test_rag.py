"""
RAG 项目 pytest 测试

分层：
- 纯单元测试（无外部依赖，CI 可跑）
- 集成测试（需 PostgreSQL + BGE-M3，用环境变量 RAG_RUN_INTEGRATION=1 启用，否则 skip）

用法：
  pytest tests/ -v
  RAG_RUN_INTEGRATION=1 pytest tests/test_rag.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_core.rag_engine import _count_words  # noqa: E402
from rag_core.crawler import (  # noqa: E402
    CrawlerManager,
    _normalize_url,
    _simhash,
    _hamming_distance,
    validate_public_url,
)


# ──────────────────────────────────────────────
# 纯单元测试
# ──────────────────────────────────────────────

class TestCountWords:
    def test_empty(self):
        assert _count_words("") == 0
        assert _count_words(None) == 0

    def test_pure_chinese(self):
        # 中文字符逐字计数
        assert _count_words("我爱编程") == 4

    def test_pure_english(self):
        assert _count_words("hello world foo bar") == 4

    def test_mixed(self):
        # "人工智能" = 4 个汉字 + "AI" 一个词
        assert _count_words("人工智能 AI") == 5

    def test_punctuation_not_counted(self):
        assert _count_words("你好，世界！") == 4


class TestURLNormalize:
    def test_strip_tracking_params(self):
        url = "https://example.com/article/1?utm_source=x&id=5&from=rss"
        assert _normalize_url(url) == "https://example.com/article/1?id=5"

    def test_keep_meaningful_params(self):
        url = "https://example.com/p/123?page=2"
        assert _normalize_url(url) == "https://example.com/p/123?page=2"

    def test_no_query(self):
        assert _normalize_url("https://example.com/a") == "https://example.com/a"


class TestSimhash:
    def test_identical_text_same_hash(self):
        assert _simhash("今天天气很好") == _simhash("今天天气很好")

    def test_similar_text_small_distance(self):
        a = _simhash("这是一段用于测试的文本内容，长度需要足够")
        b = _simhash("这是一段用于测试的文本内容，长度需要足够")
        assert _hamming_distance(a, b) == 0

    def test_empty(self):
        assert _simhash("") == 0

    def test_hamming_distance_known(self):
        assert _hamming_distance(0, 0) == 0
        assert _hamming_distance(0b1111, 0b0000) == 4


class TestCrawlerHelpers:
    def setup_method(self):
        self.mgr = CrawlerManager.__new__(CrawlerManager)  # 不触发 __init__ 即可用静态方法

    def test_clean_title_strip_suffix(self):
        assert CrawlerManager._clean_title("  标题内容_某某网  ") == "标题内容"

    def test_clean_title_empty(self):
        assert CrawlerManager._clean_title("") == ""
        assert CrawlerManager._clean_title(None) == ""

    def test_score_content_empty(self):
        assert CrawlerManager._score_content("") == 0.0

    def test_score_content_prefers_structured(self):
        short = "短文本"
        long_text = "第一段内容。" * 50
        assert CrawlerManager._score_content(long_text) > CrawlerManager._score_content(short)

    def test_clean_content_removes_html(self):
        assert CrawlerManager._clean_content("<p>正文内容</p>") == "正文内容"

    def test_build_page_url_placeholder(self):
        assert CrawlerManager._build_page_url("https://x.com/list?page={page}", 3) == "https://x.com/list?page=3"

    def test_build_page_url_existing_page_param(self):
        assert CrawlerManager._build_page_url("https://x.com/list?page=1", 2) == "https://x.com/list?page=2"

    def test_build_page_url_no_query(self):
        assert CrawlerManager._build_page_url("https://x.com/list", 2) == "https://x.com/list?page=2"


class TestValidatePublicURL:
    """SSRF 防护：validate_public_url 的边界与内网阻断"""

    def test_public_url_allowed(self):
        assert validate_public_url("https://www.example.com/article/1")[0] is True
        assert validate_public_url("http://example.com")[0] is True

    def test_non_http_protocol_blocked(self):
        assert validate_public_url("file:///etc/passwd")[0] is False
        assert validate_public_url("ftp://example.com")[0] is False
        assert validate_public_url("gopher://example.com")[0] is False

    def test_loopback_blocked(self):
        assert validate_public_url("http://127.0.0.1:5432")[0] is False
        assert validate_public_url("http://localhost:8080")[0] is False
        assert validate_public_url("http://[::1]")[0] is False

    def test_private_ip_blocked(self):
        assert validate_public_url("http://192.168.1.1/admin")[0] is False
        assert validate_public_url("http://10.0.0.1")[0] is False
        assert validate_public_url("http://172.16.0.5")[0] is False
        assert validate_public_url("http://172.31.255.255")[0] is False

    def test_link_local_and_metadata_blocked(self):
        assert validate_public_url("http://169.254.169.254/latest/meta-data")[0] is False

    def test_empty_and_invalid(self):
        assert validate_public_url("")[0] is False
        assert validate_public_url(None)[0] is False
        assert validate_public_url("not a url")[0] is False

    def test_172_public_range_allowed(self):
        # 172.32.x.x 不在 172.16/12 内网范围，应放行
        assert validate_public_url("http://172.32.0.1")[0] is True


# ──────────────────────────────────────────────
# 集成测试（需外部服务，默认跳过）
# ──────────────────────────────────────────────

requires_integration = pytest.mark.skipif(
    os.environ.get("RAG_RUN_INTEGRATION") != "1",
    reason="设置 RAG_RUN_INTEGRATION=1 且准备好 PG + BGE-M3 后运行",
)


@requires_integration
class TestRAGIntegration:
    @pytest.fixture(scope="class")
    def engine(self):
        import yaml
        from rag_core import RAGEngine

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        eng = RAGEngine(config)
        yield eng
        eng.close()

    def test_ingest_and_query(self, engine):
        content = "PyTorch 是一个基于 Python 的科学计算库，支持 GPU 加速。"
        result = engine.ingest_document(content=content, title="测试", file_type="text")
        assert result["doc_id"]
        assert result["chunk_count"] >= 1

        results = engine.query("PyTorch 是什么", top_k=3)
        assert isinstance(results, list)

    def test_delete_document(self, engine):
        content = "待删除的文档内容，用于测试删除功能是否正常。"
        result = engine.ingest_document(content=content, title="删除测试")
        doc_id = result["doc_id"]
        engine.delete_document(doc_id)
        # 删除后再查，不应返回该 doc_id
        results = engine.query("待删除的文档", top_k=5)
        assert all(r["metadata"].get("doc_id") != doc_id for r in results)
