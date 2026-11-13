"""
RAG Knowledge Base - Web UI v2.4
Chat (SSE streaming) + RAG context injection + multi-threaded server

Start: python webui.py
Open:  http://localhost:8899
"""

import sys, os, json, yaml, logging, subprocess, time, threading, re, datetime, atexit, signal
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 离线模式：模型为本地路径（./data
# 可缩短 import 时间约 2 秒，同时避免网络波动影响启动。
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core import RAGEngine, LLMClient
from rag_core.crawler import CrawlerManager
from rag_core.wordcloud import get_top_keywords, generate_wordcloud

WORDS_CLOUD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "wordclouds")

# 日志系统：每次启动清空之前所有日志，仅保留本次启动记录
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "rag_webui.log")


def _clear_logs():
    """启动前清空 logs 目录下所有日志文件（含轮转残留、历史 server_*.log）。"""
    try:
        for name in os.listdir(LOG_DIR):
            # 只清理日志类文件：*.log 及 *.log.N 轮转备份、server_* 历史残留
            if name.endswith(".log") or name.startswith("server_") or ".log." in name:
                path = os.path.join(LOG_DIR, name)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError as e:
                    print(f"[warn] 清理日志失败 {name}: {e}", file=sys.stderr)
    except OSError as e:
        print(f"[warn] 扫描日志目录失败: {e}", file=sys.stderr)


_clear_logs()

# 覆盖写模式：每次启动只保留本次记录
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger("webui")
logger.info("=" * 50)
logger.info("RAG WebUI 启动: %s", datetime.datetime.now().isoformat())
logger.info("=" * 50)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
ALLOWED_INGEST_ROOTS = [
    "E:\\", "./data",
    "D:\\", "./data",
    os.path.expanduser("~") + os.sep,
    os.path.expanduser("~").replace("\\", "/") + "/",
]
engine: RAGEngine = None
llm: LLMClient = None

# 引擎加载状态（先启HTTP，后台加载）
_engine_ready = threading.Event()
_engine_error: str = None
_engine_load_time: float = None


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — SSE long-poll doesn't block other requests"""
    daemon_threads = True
    allow_reuse_address = True


PG_BIN = r"./data"
PG_DATA = r"./data"


def ensure_pg(config: dict = None):
    """Ensure PostgreSQL is running; reads credentials from config when available."""
    if config:
        pg = config["postgresql"]
        dsn = dict(host=pg["host"], port=pg["port"], dbname=pg["database"],
                     user=pg["user"], password=pg["password"])
    else:
        # 无配置时的兜底凭据：密码从环境变量读取，避免硬编码泄露；
        # 未设置时给出明确报错，而不是用默认弱口令连接。
        default_pw = os.environ.get("RAG_PG_PASSWORD", "")
        dsn = dict(host="localhost", port=5432, dbname="ragdb",
                   user=os.environ.get("RAG_PG_USER", "postgres"),
                   password=default_pw)

    try:
        import psycopg2
        conn = psycopg2.connect(**dsn, connect_timeout=3)
        conn.close()
        return True, "PostgreSQL already running"
    except Exception:
        pass

    # 未就绪：清理崩溃残留的进程锁文件（防止 pg_ctl 误判"服务器运行中"）
    pid_file = os.path.join(PG_DATA, "postmaster.pid")
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
            logger.info("Removed stale postmaster.pid")
        except OSError:
            logger.warning("postmaster.pid exists but could not be removed (PG may be starting)")

    # 启动 PG（日志由 postgresql.conf 的 logging_collector 管理，不再 -l 到数据目录）
    #
    # ⚠️ 关键：不能用 capture_output=True / stdout=PIPE。
    # pg_ctl 启动的 postgres 是常驻进程，会继承子进程的 stdout/stderr 管道并一直持有，
    # 导致 subprocess 的 communicate() 永远等不到管道 EOF 而挂起；且 Windows 上即使
    # timeout 触发，超时后回收管道的 communicate() 仍会挂起 → 引擎加载线程永久卡死。
    # 因此把 pg_ctl 输出重定向到文件（文件无 EOF 语义，subprocess 只等 pg_ctl 退出即返回）。
    pg_start_log = os.path.join(LOG_DIR, "pg_start.log")
    try:
        with open(pg_start_log, "w", encoding="utf-8", errors="replace") as _logf:
            r = subprocess.run(
                [os.path.join(PG_BIN, "pg_ctl.exe"), "start", "-D", PG_DATA, "-w", "-t", "60"],
                stdout=_logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                timeout=90,
            )
        if r.returncode == 0:
            return True, "PostgreSQL started"
        try:
            with open(pg_start_log, encoding="utf-8", errors="replace") as _f:
                detail = _f.read().strip()
        except OSError:
            detail = ""
        return False, detail or f"pg_ctl returned non-zero ({r.returncode})"
    except subprocess.TimeoutExpired:
        return False, "PostgreSQL start timed out"
    except Exception as e:
        return False, str(e)


class APIHandler(BaseHTTPRequestHandler):

    # 类级别爬取取消标志: {request_id: bool}
    _crawl_cancel_flags = {}
    _crawl_request_counter = 0
    # 并发控制：最多 2 个爬取 + 2 个摄入同时进行
    _crawl_semaphore = threading.Semaphore(2)
    _ingest_semaphore = threading.Semaphore(2)

    def log_message(self, fmt, *args): pass

    _ALLOWED_CORS_HOSTS = {"localhost", "127.0.0.1", "::1"}

    def _cors_headers(self):
        """Set CORS headers (restrict to localhost origins).

        用 hostname 精确匹配，避免 startswith 前缀绕过（如 http://localhost.evil.com）。
        """
        origin = self.headers.get("Origin", "")
        allowed = False
        if origin:
            try:
                host = urlparse(origin).hostname or ""
                # 只放行本机回环地址，且端口不限（避免 http://localhost:任意端口 被拒）
                allowed = host in self._ALLOWED_CORS_HOSTS
            except ValueError:
                allowed = False
        else:
            # 无 Origin 头（同源/非浏览器请求），无需 CORS 头，直接返回
            return
        if allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, error_type: str, detail: str = "", status: int = 500):
        """统一错误响应格式"""
        error_map = {
            "not_found": (404, "资源不存在"),
            "forbidden": (403, "权限不足"),
            "bad_request": (400, "请求参数错误"),
            "loading": (503, "引擎加载中"),
            "rate_limit": (429, "请求过于频繁"),
            "internal": (500, "服务器内部错误"),
        }
        code, msg = error_map.get(error_type, (status, detail or "未知错误"))
        self._send_json({
            "error": msg,
            "error_type": error_type,
            "detail": detail if detail else msg,
            "status": code
        }, code)

    def _send_sse_event(self, data: str, event: str = None):
        out = ""
        if event:
            out += f"event: {event}\n"
        out += f"data: {data}\n\n"
        try:
            self.wfile.write(out.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise SystemExit(0)

    def _read_body(self):
        length_str = self.headers.get("Content-Length", "")
        if not length_str:
            return {}
        try:
            length = int(length_str)
        except ValueError:
            return {}
        if length <= 0 or length > 5 * 1024 * 1024:  # 5MB max
            self._send_json({"error": "payload too large (max 5MB)"}, 413)
            return None  # sentinel
        raw = self.rfile.read(length).decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # -- Routing --

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._serve_static("index.html", "text/html")
        elif path.startswith("/static/"):
            self._serve_static(path[1:])
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/documents":
            self._api_list_docs()
        elif path == "/api/health":
            self._send_json({"ok": True})
        elif path == "/api/ready":
            self._api_ready()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/llm/status":
            self._api_llm_status()
        elif path == "/api/config":
            self._api_config()
        elif path == "/api/config/models":
            self._api_config_models()
        elif path == "/api/sources":
            self._api_sources()
        elif path == "/api/sources/uncategorized-docs":
            self._api_uncategorized_docs()
        elif path == "/api/categories":
            self._api_categories()
        elif path.startswith("/api/categories/") and "/docs" in path:
            # /api/categories/{id}/docs
            parts = path.split("/")
            if len(parts) >= 4:
                try:
                    cat_id = int(parts[3])
                    self._api_category_docs(cat_id)
                except ValueError:
                    self._send_json({"error": "invalid category id"}, 400)
            else:
                self._send_json({"error": "not found"}, 404)
        elif path.startswith("/api/documents/"):
            parts = path.split("/")
            if len(parts) >= 4:
                doc_id = parts[3]
                if len(parts) >= 5 and parts[4] == "keywords":
                    self._api_doc_keywords(doc_id)
                elif len(parts) >= 5 and parts[4] == "content":
                    self._api_doc_content(doc_id)
                elif len(parts) >= 5 and parts[4] == "wordcloud":
                    self._api_doc_wordcloud(doc_id)
                else:
                    self._api_doc_detail(doc_id)
            else:
                self._send_json({"error": "not found"}, 404)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()
        if body is None:
            return  # _read_body already sent 413

        if path == "/api/search":
            self._api_search(body)
        elif path == "/api/ingest/text":
            self._api_ingest_text(body)
        elif path == "/api/ingest/files":
            self._api_ingest_files(body)
        elif path == "/api/ingest/directory":
            self._api_ingest_directory(body)
        elif path == "/api/chat":
            self._api_chat(body)
        elif path == "/api/settings/save":
            self._api_save_settings(body)
        elif path == "/api/config/update":
            self._api_config_update(body)
        elif path == "/api/config/reset":
            self._api_config_reset(body)
        elif path == "/api/categories/create":
            self._api_create_category(body)
        elif path == "/api/categories/delete":
            self._api_delete_category(body)
        elif path == "/api/documents/update":
            self._api_update_doc_meta(body)
        elif path == "/api/documents/update-content":
            self._api_update_doc_content(body)
        elif path == "/api/documents/ai-summarize":
            self._api_ai_summarize(body)
        elif path == "/api/crawl/url":
            self._api_crawl_url(body)
        elif path == "/api/crawl/ingest":
            self._api_crawl_ingest(body)
        elif path == "/api/sources/add":
            self._api_add_source(body)
        elif path == "/api/sources/remove":
            self._api_remove_source(body)
        elif path == "/api/crawl/source":
            self._api_crawl_source(body)
        elif path == "/api/crawl/source-progress":
            self._api_crawl_source_progress(body)
        elif path == "/api/crawl/cancel":
            self._api_crawl_cancel(body)
        elif path == "/api/documents/batch-delete":
            self._api_batch_delete(body)
        elif path == "/api/crawl/batch-ingest":
            self._api_crawl_batch_ingest(body)
        elif path == "/api/documents/editor":
            self._api_doc_editor(body)
        elif path == "/api/categories/assign-doc":
            self._api_assign_doc_to_category(body)
        elif path == "/api/categories/remove-doc":
            self._api_remove_doc_from_category(body)
        elif path == "/api/sources/settings":
            self._api_source_settings(body)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/documents/"):
            self._api_delete_doc(path.split("/")[-1])
        elif path.startswith("/api/categories/"):
            try:
                cat_id = int(path.split("/")[-1])
                self._api_delete_cat_by_id(cat_id)
            except ValueError:
                self._send_json({"error": "invalid category id"}, 400)
        else:
            self._send_json({"error": "not found"}, 404)

    # -- Static Files --

    def _serve_static(self, rel_path, mime=None):
        base = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.realpath(os.path.join(base, rel_path))
        if not filepath.startswith(base + os.sep) and filepath != base:
            self._send_json({"error": "forbidden"}, 403)
            return
        if not os.path.isfile(filepath):
            self._send_json({"error": f"file not found: {rel_path}"}, 404)
            return

        mime_map = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }
        mime = mime or mime_map.get(os.path.splitext(filepath)[1], "application/octet-stream")
        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(content)

    # -- Status --

    def _api_status(self):
        import torch
        ready = _engine_ready.is_set()
        # 注意：不能用 engine.embedder.dimension（property 会触发 _load_model 阻塞加载大模型）
        # 改用缓存的 _dimension（未加载时为 None），避免 /api/status 意外触发模型加载
        model_dim = engine.embedder._dimension if engine else None
        self._send_json({
            "ready": ready,
            "error": _engine_error,
            "load_time": _engine_load_time,
            "postgresql": engine.db.is_connected if engine else False,
            "chromadb": True,
            "embedding": engine.embedder._model is not None if engine else False,
            "cuda": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device": engine.embedder.device if engine else "unknown",
            "model_dim": model_dim or 0,
        })

    def _api_ready(self):
        """前端轮询引擎是否就绪"""
        if _engine_error:
            self._send_json({"ready": False, "error": _engine_error}, 500)
        elif _engine_ready.is_set():
            self._send_json({"ready": True, "load_time": _engine_load_time})
        else:
            self._send_json({"ready": False, "status": "loading"})

    def _api_llm_status(self):
        self._send_json({
            "enabled": llm.enabled if llm else False,
            "model": llm.model if llm else "",
            "api_base": llm.api_base if llm else "",
        })

    # -- Config Display & Management --

    def _mask_key(self, key: str) -> str:
        """API key 脱敏"""
        if not key:
            return ""
        if len(key) <= 8:
            return key[:2] + "***" + key[-2:]
        return key[:4] + "..." + key[-4:]

    def _api_config(self):
        """返回完整配置信息（API key 脱敏）"""
        if not llm:
            self._send_json({"error": "LLM not initialized"}, 503)
            return
        self._send_json({
            "llm": {
                "api_base": llm.api_base,
                "api_key": self._mask_key(llm.api_key),
                "model": llm.model,
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "enabled": llm.enabled,
                "timeout": llm.timeout,
            }
        })

    def _api_config_models(self):
        """返回可用模型列表（供对话页下拉选择）"""
        models = []
        default_model = ""
        if llm and llm.enabled:
            provider = llm.api_base.rstrip("/").split("//")[-1].split("/")[0]
            models.append({
                "id": llm.model,
                "name": f"{llm.model} ({provider})",
                "provider": provider,
                "type": "api",
                "default": True,
            })
            default_model = llm.model
        self._send_json({
            "models": models,
            "default": default_model,
            "api_base": llm.api_base if llm else "",
        })

    def _api_config_update(self, body):
        """更新单个配置字段"""
        field = body.get("field", "").strip()
        value = body.get("value")
        if not field:
            self._send_json({"error": "field is required"}, 400)
            return

        allowed_fields = {"api_base", "api_key", "model", "temperature", "max_tokens", "timeout"}
        if field not in allowed_fields:
            self._send_json({"error": f"unknown field: {field}"}, 400)
            return

        try:
            import yaml as _yaml
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)

            llm_cfg = cfg.setdefault("llm", {})
            old_val = llm_cfg.get(field)

            if field in ("temperature", "timeout"):
                value = float(value)
            elif field == "max_tokens":
                value = int(value)
            else:
                value = str(value).strip()

            llm_cfg[field] = value

            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                _yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            logger.info("Config field %s: %s -> %s", field,
                        self._mask_key(str(old_val)) if field == "api_key" else old_val,
                        self._mask_key(str(value)) if field == "api_key" else value)
            self._send_json({"success": True, "field": field, "value": value})
        except Exception as e:
            logger.exception("Config update failed")
            self._send_json({"error": str(e)}, 500)

    def _api_config_reset(self, body):
        """重置单个配置字段为默认值"""
        field = body.get("field", "").strip()
        if not field:
            self._send_json({"error": "field is required"}, 400)
            return

        defaults = {
            "api_base": "https://api.deepseek.com",
            "api_key": "",
            "model": "deepseek-v4-flash",
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 120,
        }
        if field not in defaults:
            self._send_json({"error": f"unknown field: {field}"}, 400)
            return

        try:
            import yaml as _yaml
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)

            llm_cfg = cfg.setdefault("llm", {})
            old_val = llm_cfg.get(field)

            if defaults[field] == "" and field in llm_cfg:
                del llm_cfg[field]
            else:
                llm_cfg[field] = defaults[field]

            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                _yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            logger.info("Config field %s reset: %s -> %s", field,
                        self._mask_key(str(old_val)) if field == "api_key" else old_val,
                        self._mask_key(str(defaults[field])) if field == "api_key" else defaults[field])
            self._send_json({"success": True, "field": field, "value": defaults[field]})
        except Exception as e:
            logger.exception("Config reset failed")
            self._send_json({"error": str(e)}, 500)

    # -- Sources Management --

    def _get_crawler(self):
        if not hasattr(self, "_crawler"):
            self._crawler = CrawlerManager()
        return self._crawler

    def _api_sources(self):
        """GET /api/sources — 返回默认源列表"""
        crawler = self._get_crawler()
        sources = crawler.get_sources()
        self._send_json({"sources": sources})

    def _api_add_source(self, body):
        """POST /api/sources/add — 添加默认源"""
        name = body.get("name", "").strip()
        url = body.get("url", "").strip()
        category = body.get("category", "通用").strip()
        list_url = body.get("list_url", "").strip()
        article_pattern = body.get("article_pattern", "").strip()
        if not name or not url:
            self._send_json({"error": "名称和URL不能为空"}, 400)
            return
        crawler = self._get_crawler()
        crawler.add_source(name, url, category,
                          list_url=list_url or None,
                          article_pattern=article_pattern or None,
                          settings=body.get("settings", {}))
        self._send_json({"success": True})

    def _api_remove_source(self, body):
        """POST /api/sources/remove — 移除默认源"""
        name = body.get("name", "").strip()
        if not name:
            self._send_json({"error": "名称不能为空"}, 400)
            return
        crawler = self._get_crawler()
        ok = crawler.remove_source(name)
        self._send_json({"success": ok})

    # -- Categories Management --

    def _api_categories(self):
        """GET /api/categories — 返回分类列表"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            cats = engine.get_categories(schema)
            self._send_json({"categories": cats})
        except Exception as e:
            logger.exception("Categories get failed")
            self._send_json({"error": str(e)}, 500)

    def _api_create_category(self, body):
        """POST /api/categories/create — 创建分类"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        name = body.get("name", "").strip()
        if not name:
            self._send_json({"error": "名称不能为空"}, 400)
            return
        parent_id = body.get("parent_id")
        color = body.get("color")
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            cat_id = engine.create_category(
                schema, name, parent_id, color,
            )
            self._send_json({"success": True, "id": cat_id})
        except Exception as e:
            logger.exception("Category create failed")
            self._send_json({"error": str(e)}, 500)

    def _api_delete_category(self, body):
        """POST /api/categories/delete — 删除分类"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        cat_id = body.get("category_id")
        if not cat_id:
            self._send_json({"error": "category_id required"}, 400)
            return
        try:
            engine.delete_category(engine._get_schema_name("default"), int(cat_id))
            self._send_json({"success": True})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_delete_cat_by_id(self, cat_id: int):
        """DELETE /api/categories/{id}"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            engine.delete_category(engine._get_schema_name("default"), cat_id)
            self._send_json({"success": True})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_category_docs(self, cat_id: int):
        """GET /api/categories/{id}/docs"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            docs = engine.get_docs_by_category(schema, cat_id)
            self._send_json({"documents": docs})
        except Exception as e:
            logger.exception("Category docs failed")
            self._send_json({"error": str(e)}, 500)

    def _api_uncategorized_docs(self):
        """GET /api/sources/uncategorized-docs"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            docs = engine.get_all_uncategorized_docs(schema)
            self._send_json({"documents": docs})
        except Exception as e:
            logger.exception("Uncategorized docs failed")
            self._send_json({"error": str(e)}, 500)

    def _api_assign_doc_to_category(self, body):
        """POST /api/categories/assign-doc"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        doc_id = body.get("doc_id", "").strip()
        cat_id = body.get("category_id")
        if not doc_id or not cat_id:
            self._send_json({"error": "doc_id和category_id不能为空"}, 400)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            engine.add_doc_to_category(schema, doc_id, int(cat_id))
            self._send_json({"success": True})
        except Exception as e:
            logger.exception("Assign doc to category failed")
            self._send_json({"error": str(e)}, 500)

    def _api_remove_doc_from_category(self, body):
        """POST /api/categories/remove-doc"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        doc_id = body.get("doc_id", "").strip()
        cat_id = body.get("category_id")
        if not doc_id or not cat_id:
            self._send_json({"error": "doc_id和category_id不能为空"}, 400)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            engine.remove_doc_from_category(schema, doc_id, int(cat_id))
            self._send_json({"success": True})
        except Exception as e:
            logger.exception("Remove doc from category failed")
            self._send_json({"error": str(e)}, 500)

    def _api_source_settings(self, body):
        """POST /api/sources/settings — 更新单个源的设置"""
        name = body.get("name", "").strip()
        settings = body.get("settings", {})
        if not name:
            self._send_json({"error": "name required"}, 400)
            return
        try:
            crawler = self._get_crawler()
            sources = crawler.get_sources()
            found = False
            for i, s in enumerate(sources):
                if s["name"] == name:
                    s.setdefault("settings", {})
                    s["settings"].update(settings)
                    found = True
                    break
            if not found:
                self._send_json({"error": f"源不存在: {name}"}, 404)
                return
            crawler._sources = sources
            crawler._save_sources()
            self._send_json({"success": True})
        except Exception as e:
            logger.exception("Source settings failed")
            self._send_json({"error": str(e)}, 500)

    # -- Document Enhanced Methods --

    def _api_doc_detail(self, doc_id: str):
        """GET /api/documents/{doc_id} — 文档详情"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            doc = engine.get_document_info(schema, doc_id)
            if not doc:
                self._send_json({"error": "document not found"}, 404)
                return
            self._send_json(doc)
        except Exception as e:
            logger.exception("Doc detail failed")
            self._send_json({"error": str(e)}, 500)

    def _api_doc_content(self, doc_id: str):
        """GET /api/documents/{doc_id}/content — 文档全文（按 chunk 顺序拼接）

        供外部系统（如小说创作平台）把 RAG 文档导入为自己的知识库使用。
        只读、无副作用；返回 chunks 顺序拼接的正文与用量信息。
        """
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中", "loading": True}, 503)
            return
        try:
            schema = engine._get_schema_name("default")
            engine.db.ensure_schema(schema)
            doc = engine.get_document_info(schema, doc_id)
            if not doc:
                self._send_json({"error": "document not found"}, 404)
                return
            chunks = engine.db.get_chunks_by_doc(schema, doc_id)
            # 按 chunk_index 排序，保证拼接顺序与原文一致
            chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
            content = "\n\n".join(c["content"] for c in chunks if c.get("content"))
            self._send_json({
                "doc_id": doc_id,
                "title": doc.get("title") or doc_id[:8],
                "source": doc.get("source") or "",
                "file_type": doc.get("file_type") or "",
                "word_count": doc.get("word_count") or len(content),
                "chunk_count": len(chunks),
                "summary": doc.get("summary") or "",
                "description": doc.get("description") or "",
                "content": content,
            })
        except Exception as e:
            logger.exception("Doc content failed")
            self._send_json({"error": str(e)}, 500)

    def _api_update_doc_meta(self, body):
        """POST /api/documents/update — 更新文档元数据"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        doc_id = body.get("doc_id", "").strip()
        if not doc_id:
            self._send_json({"error": "doc_id required"}, 400)
            return
        updates = {k: v for k, v in body.items()
                   if k in ("description", "tags", "summary", "source_url", "category_id", "title") and v is not None}
        if not updates:
            self._send_json({"error": "no valid fields to update"}, 400)
            return
        try:
            engine.update_document_metadata(
                engine._get_schema_name("default"), doc_id, updates,
            )
            self._send_json({"success": True})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_update_doc_content(self, body):
        """POST /api/documents/update-content — 更新文档内容（重新分块+向量化）"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        doc_id = body.get("doc_id", "").strip()
        content = body.get("content", "").strip()
        if not doc_id or not content:
            self._send_json({"error": "doc_id和content不能为空"}, 400)
            return
        title = body.get("title")
        try:
            engine.update_document_content(
                engine._get_schema_name("default"),
                doc_id, content, title,
            )
            self._send_json({"success": True})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_doc_keywords(self, doc_id: str):
        """GET /api/documents/{doc_id}/keywords — 文档中文 TOP20 关键词"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            doc = engine.get_document_info(engine._get_schema_name("default"), doc_id)
            if not doc:
                self._send_json({"error": "document not found"}, 404)
                return
            # Get content from chunks
            chunks = engine.db.get_chunks_by_doc(engine._get_schema_name("default"), doc_id)
            full_text = " ".join(c["content"] for c in chunks)
            keywords = get_top_keywords(full_text, top_n=20)
            self._send_json({"doc_id": doc_id, "keywords": keywords, "total_words": len(full_text), "content": full_text})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_doc_wordcloud(self, doc_id: str):
        """GET /api/documents/{doc_id}/wordcloud — 生成文档词云图，返回图片URL"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        try:
            # Check cache
            os.makedirs(WORDS_CLOUD_DIR, exist_ok=True)
            cached = os.path.join(WORDS_CLOUD_DIR, f"{doc_id}.png")
            if os.path.isfile(cached):
                img_url = f"/static/wordclouds/{doc_id}.png"
                self._send_json({"wordcloud_url": img_url, "cached": True})
                return

            doc = engine.get_document_info(engine._get_schema_name("default"), doc_id)
            if not doc:
                self._send_json({"error": "document not found"}, 404)
                return
            chunks = engine.db.get_chunks_by_doc(engine._get_schema_name("default"), doc_id)
            full_text = " ".join(c["content"] for c in chunks)
            result_path = generate_wordcloud(full_text, cached, max_words=100)
            if result_path:
                img_url = f"/static/wordclouds/{doc_id}.png"
                self._send_json({"wordcloud_url": img_url, "cached": False})
            else:
                self._send_json({"error": "生成词云失败"}, 500)
        except Exception as e:
            logger.exception("Wordcloud failed")
            self._send_json({"error": str(e)}, 500)

    def _api_ai_summarize(self, body):
        """POST /api/documents/ai-summarize — AI 文档摘要"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        doc_id = body.get("doc_id", "").strip()
        if not doc_id:
            self._send_json({"error": "doc_id required"}, 400)
            return
        if not llm or not llm.enabled:
            self._send_json({"error": "LLM 未配置"}, 400)
            return
        try:
            doc = engine.get_document_info(engine._get_schema_name("default"), doc_id)
            if not doc:
                self._send_json({"error": "document not found"}, 404)
                return
            chunks = engine.db.get_chunks_by_doc(engine._get_schema_name("default"), doc_id)
            full_text = "\n".join(c["content"] for c in chunks)
            summary = engine.ai_summarize(
                full_text, llm,
                title=doc.get("title", ""),
            )
            if summary:
                # Save summary
                engine.update_document_metadata(
                    engine._get_schema_name("default"),
                    doc_id, {"summary": summary},
                )
            self._send_json({"success": True, "summary": summary or "摘要生成失败"})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    # -- Crawl Endpoints --

    def _api_crawl_url(self, body):
        """POST /api/crawl/url — 抓取单个URL（预览，不存入库）"""
        url = body.get("url", "").strip()
        timeout = int(body.get("timeout", 20))
        min_quality = body.get("min_quality", "C")
        if not url:
            self._send_json({"error": "URL不能为空"}, 400)
            return
        try:
            crawler = self._get_crawler()
            result = crawler.crawl_url(url, timeout=timeout)
            if not result.get("success") or (min_quality == "B" and result.get("quality") == "C") or (min_quality == "A" and result.get("quality") in ("B", "C")):
                if not result.get("success"):
                    result["error"] = result.get("error", "爬取失败")
                    self._send_json(result)
                    return
                self._send_json({"error": f"内容质量 {result.get('quality')} 级，低于要求的 {min_quality} 级", "success": False})
                return
            self._send_json(result)
        except Exception as e:
            import traceback, logging
            logging.getLogger(__name__).error("爬取出错", exc_info=True)
            self._send_json({"error": f"爬取出错: {e}", "success": False}, 500)

    def _api_crawl_ingest(self, body):
        """POST /api/crawl/ingest — 爬取URL并摄入知识库"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        source_name = body.get("source_name", "临时爬取").strip()
        url = body.get("url", "").strip()
        if not url:
            self._send_json({"error": "url不能为空"}, 400)
            return
        try:
            crawler = self._get_crawler()
            result = crawler.crawl_url(url)
            if not result.get("success"):
                self._send_json(result)
                return
            content = result.get("content", "")
            if len(content) < 20:
                self._send_json({"error": "内容过短，无法摄入"}, 400)
                return
            entities = result.get("entities", {})
            meta = {
                "source_category": source_name,
                "source_url": url,
                "entities": entities,
            }
            ingest_result = engine.ingest_document(
                content=content,
                title=result.get("title", ""),
                source=url,
                file_type="crawled",
                metadata=meta,
            )
            ingest_result["title"] = result.get("title", "")
            ingest_result["entities"] = entities
            self._send_json(ingest_result)
        except Exception as e:
            logging.getLogger(__name__).error("爬取摄入出错", exc_info=True)
            self._send_json({"error": f"爬取摄入出错: {e}", "success": False}, 500)

    def _api_crawl_source(self, body):
        """POST /api/crawl/source — 从源名称读取设置并执行翻页爬取"""
        source_name = body.get("name", "").strip()
        if not source_name:
            self._send_json({"error": "name 不能为空"}, 400)
            return
        try:
            crawler = self._get_crawler()
            sources = crawler.get_sources()
            source = None
            for s in sources:
                if s["name"] == source_name:
                    source = s
                    break
            if not source:
                self._send_json({"error": f"源不存在: {source_name}"}, 404)
                return

            settings = source.get("settings", {})
            max_pages = int(settings.get("max_pages", 3))
            max_articles = int(settings.get("max_articles", 50))

            logger.info("爬取源 '%s': pages=%d, max=%d, timeout=%s, delay=%s",
                        source_name, max_pages, max_articles,
                        settings.get("timeout", 20), settings.get("delay", 1.5))
            articles = crawler.crawl_source_advanced(
                source,
                settings=settings,
                max_pages=max_pages,
                max_articles=max_articles,
            )
            # 限制每篇文章预览长度
            for a in articles:
                if a.get("success") and len(a.get("content", "")) > 2000:
                    a["preview"] = a["content"][:2000] + "..."
                else:
                    a["preview"] = a.get("content", "")[:2000]
            self._send_json({
                "articles": articles,
                "total": len(articles),
                "success_count": sum(1 for a in articles if a.get("success")),
            })
        except Exception as e:
            logging.getLogger(__name__).error("爬取源出错", exc_info=True)
            self._send_json({"error": f"爬取出错: {e}", "success": False}, 500)

    def _api_crawl_cancel(self, body):
        """POST /api/crawl/cancel — 取消指定源的爬取"""
        source_name = body.get("name", "").strip()
        if not source_name:
            self._send_json({"error": "name 不能为空"}, 400)
            return
        # 按源名标记取消，所有该源的进行中请求都会检查到
        APIHandler._crawl_cancel_flags[source_name] = True
        logger.info("Crawl cancel requested for: %s", source_name)
        self._send_json({"success": True, "message": f"取消请求已发送: {source_name}"})

    def _api_batch_delete(self, body):
        """POST /api/documents/batch-delete — 批量删除文档"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        doc_ids = body.get("doc_ids", [])
        if not doc_ids or not isinstance(doc_ids, list):
            self._send_json({"error": "doc_ids 列表不能为空"}, 400)
            return
        schema = engine._get_schema_name("default")
        success_count = 0
        failed = []
        for doc_id in doc_ids:
            try:
                engine.delete_document(doc_id)
                # delete_document already cleans wordcloud via rag_engine
                success_count += 1
            except Exception as e:
                failed.append({"doc_id": doc_id, "error": str(e)})
        self._send_json({
            "success_count": success_count,
            "failed": failed,
            "total": len(doc_ids)
        })

    def _api_crawl_source_progress(self, body):
        """POST /api/crawl/source-progress — 带SSE进度推送的翻页爬取"""
        source_name = body.get("name", "").strip()
        if not source_name:
            self._send_json({"error": "name 不能为空"}, 400)
            return

        # 并发控制：非阻塞获取信号量
        if not APIHandler._crawl_semaphore.acquire(blocking=False):
            self._send_json({"error": "已有爬取任务进行中，请等待完成或取消后再试"}, 429)
            return

        # 重置取消标志
        APIHandler._crawl_cancel_flags[source_name] = False

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors_headers()
        self.end_headers()

        def emit(data, event="progress"):
            self._send_sse_event(json.dumps(data, ensure_ascii=False, default=str), event=event)

        def cancel_check():
            return APIHandler._crawl_cancel_flags.get(source_name, False)

        try:
            crawler = self._get_crawler()
            sources = crawler.get_sources()
            source = None
            for s in sources:
                if s["name"] == source_name:
                    source = s
                    break
            if not source:
                emit({"error": f"源不存在: {source_name}"}, event="error")
                self._send_sse_event("[DONE]", event="done")
                return

            settings = source.get("settings", {})
            max_pages = int(settings.get("max_pages", 3))
            max_articles = int(settings.get("max_articles", 50))

            emit({
                "type": "start", "source_name": source_name,
                "max_pages": max_pages, "max_articles": max_articles
            }, event="status")

            articles = crawler.crawl_source_advanced_with_progress(
                source,
                settings=settings,
                max_pages=max_pages,
                max_articles=max_articles,
                progress_callback=lambda data: emit(data, event="progress"),
                cancel_check=cancel_check
            )

            for a in articles:
                if a.get("success") and len(a.get("content", "")) > 2000:
                    a["preview"] = a["content"][:2000] + "..."
                else:
                    a["preview"] = a.get("content", "")[:2000]

            emit({
                "type": "complete", "total": len(articles),
                "success_count": sum(1 for a in articles if a.get("success")),
                "articles": articles[:200]
            }, event="result")
        except Exception as e:
            logging.getLogger(__name__).error("爬取源出错(SSE)", exc_info=True)
            emit({"error": str(e)}, event="error")
        finally:
            self._send_sse_event("[DONE]", event="done")
            APIHandler._crawl_semaphore.release()

    def _api_crawl_batch_ingest(self, body):
        """POST /api/crawl/batch-ingest — 批量摄入选中文章（支持 min_quality 过滤）"""
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return
        articles = body.get("articles", [])
        source_name = body.get("source_name", "")
        min_quality = body.get("min_quality", "C")
        if not articles:
            self._send_json({"error": "请选择文章"}, 400)
            return
        quality_order = {"A": 3, "B": 2, "C": 1}
        min_q_val = quality_order.get(min_quality, 0)
        results = []
        skipped = 0
        for art in articles:
            if not art.get("content") or len(art["content"].strip()) < 20:
                continue
            # 质量过滤
            art_q = quality_order.get(art.get("quality", "C"), 1)
            if art_q < min_q_val:
                skipped += 1
                continue
            content = art["content"]
            entities = art.get("entities", {})
            meta = {
                "source_category": source_name,
                "source_url": art.get("url", ""),
                "entities": entities,
            }
            try:
                r = engine.ingest_document(
                    content=content,
                    title=art.get("title", ""),
                    source=art.get("url", ""),
                    file_type="crawled",
                    metadata=meta,
                )
                r["title"] = art.get("title", "")
                r["success"] = True
                results.append(r)
            except Exception as e:
                results.append({
                    "success": False,
                    "title": art.get("title", ""),
                    "error": str(e),
                })
        self._send_json({
            "results": results,
            "total": len(results),
            "success_count": sum(1 for r in results if r.get("success")),
            "skipped": skipped,
        })

    # -- Document Editor --

    def _api_doc_editor(self, body):
        """POST /api/documents/editor — 编辑文档内容（含完整编辑器）"""
        doc_id = body.get("doc_id", "").strip()
        content = body.get("content", "")
        title = body.get("title", "")
        description = body.get("description", "")
        tags = body.get("tags", [])

        if not doc_id:
            self._send_json({"error": "doc_id required"}, 400)
            return

        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中"}, 503)
            return

        schema = engine._get_schema_name("default")
        engine.db.ensure_schema(schema)

        try:
            if content:
                # 更新内容（重新切分+向量化）
                engine.update_document_content(schema, doc_id, content, title=title)
            if description or tags:
                updates = {}
                if description:
                    updates["description"] = description
                if tags:
                    updates["tags"] = tags
                if updates:
                    engine.update_document_metadata(schema, doc_id, updates)
            self._send_json({"success": True})
        except Exception as e:
            logger.exception("Document editor save failed")
            self._send_json({"error": str(e)}, 500)

    # -- Documents --

    def _api_list_docs(self):
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中", "loading": True}, 503)
            return
        try:
            schema = engine._get_schema_name("default")
            with engine.db.cursor() as cur:
                engine.db._set_search_path(cur, schema)
                cur.execute(
                    "SELECT d.doc_id, d.title, d.source, d.file_type, d.word_count, d.created_at, "
                    "COUNT(c.id) as chunk_count "
                    "FROM documents d LEFT JOIN chunks c ON d.doc_id = c.doc_id "
                    "GROUP BY d.doc_id, d.title, d.source, d.file_type, d.word_count, d.created_at "
                    "ORDER BY d.created_at DESC LIMIT 200"
                )
                rows = cur.fetchall()
            docs = []
            for r in rows:
                docs.append({
                    "doc_id": r["doc_id"],
                    "title": r["title"] or r["doc_id"][:8],
                    "source": r["source"] or "-",
                    "file_type": r["file_type"] or "-",
                    "word_count": r["word_count"] or 0,
                    "chunk_count": r["chunk_count"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else "",
                })
            self._send_json({"documents": docs})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_delete_doc(self, doc_id):
        try:
            engine.delete_document(doc_id)
            self._send_json({"success": True, "deleted": doc_id})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    # -- Search (thread-safe: per-request params, no global config mutation) --

    def _api_search(self, body):
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中", "loading": True}, 503)
            return
        query = body.get("query", "").strip()
        if not query:
            self._send_json({"error": "query is required"}, 400)
            return
        if len(query) > 4000:
            self._send_json({"error": "query too long (max 4000 chars)"}, 400)
            return
        top_k = body.get("top_k", 5)
        threshold = body.get("threshold", 0.4)
        results = engine.query(query, top_k=top_k, threshold=threshold)
        self._send_json({"query": query, "results": results, "count": len(results)})

    # -- Ingest --

    def _api_ingest_text(self, body):
        if not _engine_ready.is_set():
            self._send_json({"error": "引擎加载中", "loading": True}, 503)
            return
        if not APIHandler._ingest_semaphore.acquire(blocking=False):
            self._send_json({"error": "已有摄入任务进行中，请稍后再试"}, 429)
            return
        try:
            content = body.get("content", "").strip()
            if not content:
                self._send_json({"error": "content is required"}, 400)
                return
            title = body.get("title", "\u624b\u52a8\u7c98\u8d34")
            source = body.get("source", "web_ui")
            try:
                result = engine.ingest_document(content=content, title=title, source=source, file_type="text")
                self._send_json({"success": True, **result})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        finally:
            APIHandler._ingest_semaphore.release()

    def _is_allowed_path(self, p) -> bool:
        """检查路径是否在允许的摄入白名单内"""
        s = str(p).replace("/", "\\")
        for root in ALLOWED_INGEST_ROOTS:
            r = root.replace("/", "\\")
            if s.startswith(r):
                return True
        return False

    def _api_ingest_files(self, body):
        files = body.get("files", [])
        if not files:
            self._send_json({"error": "files list is required"}, 400)
            return
        success = fail = 0
        for fpath in files:
            try:
                p = Path(fpath).resolve()
                if not p.exists() or not p.is_file():
                    fail += 1
                    continue
                if not self._is_allowed_path(p):
                    fail += 1
                    continue
                content = p.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    fail += 1
                    continue
                engine.ingest_document(content=content, title=p.stem, source=str(p), file_type=p.suffix.lstrip("."))
                success += 1
            except Exception:
                fail += 1
        self._send_json({"success": True, "imported": success, "failed": fail, "total": len(files)})

    def _api_ingest_directory(self, body):
        directory = body.get("directory", "").strip()
        if not directory:
            self._send_json({"error": "directory is required"}, 400)
            return
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            self._send_json({"error": "directory not found"}, 400)
            return
        if not self._is_allowed_path(dir_path):
            self._send_json({"error": "ingest allowed only under allowed paths (E:\\, D:\\, user home)"}, 403)
            return
        extensions = body.get("extensions", [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".log", ".csv"])
        files = []
        for ext in extensions:
            files.extend(dir_path.rglob(f"*{ext}"))
        success = fail = 0
        for p in files:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    fail += 1
                    continue
                engine.ingest_document(content=content, title=p.stem, source=str(p), file_type=p.suffix.lstrip("."))
                success += 1
            except Exception:
                fail += 1
        self._send_json({"success": True, "imported": success, "failed": fail, "total": len(files)})

    # -- Settings (regex-safe: lambda avoids backreference issues in replacement) --

    def _api_save_settings(self, body):
        """保存设置：YAML 安全读写，不做危险的正则文本替换"""
        try:
            import yaml as _yaml
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = _yaml.safe_load(f)

            llm_cfg = config.setdefault("llm", {})
            changed = 0

            for key in ("api_base", "api_key", "model"):
                val = body.get(key, "").strip()
                if val and llm_cfg.get(key) != val:
                    llm_cfg[key] = val
                    changed += 1
                    logger.info("Settings: %s updated", key)

            for key in ("temperature", "max_tokens"):
                if key in body and body[key] is not None:
                    new_val = float(body[key]) if key == "temperature" else int(body[key])
                    if llm_cfg.get(key) != new_val:
                        llm_cfg[key] = new_val
                        changed += 1
                        logger.info("Settings: %s = %s", key, new_val)

            if changed == 0:
                self._send_json({"success": True, "message": "\u914d\u7f6e\u672a\u53d8\u66f4\uff0c\u65e0\u9700\u4fdd\u5b58"})
                return

            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                _yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            # 热重载 LLM 配置
            reloaded = False
            if llm:
                try:
                    reloaded = llm.reload(config)
                    logger.info("LLM hot-reloaded: %s @ %s", llm.model, llm.api_base)
                except Exception as e:
                    logger.warning("LLM reload failed: %s", e)

            msg = "设置已保存"
            if reloaded:
                msg += "，LLM 配置已热重载"
            elif llm and llm.enabled:
                msg += "，LLM 配置已更新"
            else:
                msg += "，重启服务后生效"
            self._send_json({"success": True, "message": msg})
        except Exception as e:
            logger.exception("Save settings failed")
            self._send_json({"error": str(e)}, 500)

    # -- Chat (SSE streaming + RAG) --

    def _api_chat(self, body):
        query = body.get("query", "").strip()
        if not query:
            self._send_json({"error": "query is required"}, 400)
            return

        if not _engine_ready.is_set():
            self._send_json({"error": "引擎正在加载中，请稍候"}, 503)
            return

        use_rag = body.get("use_rag", True)
        top_k = body.get("top_k", 5)
        history_raw = body.get("history") or []
        # Validate history structure
        if not isinstance(history_raw, list):
            history_raw = []
        history = [
            h for h in history_raw
            if isinstance(h, dict) and isinstance(h.get("role"), str) and isinstance(h.get("content"), str)
        ][-20:]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors_headers()
        self.end_headers()

        if not llm or not llm.enabled:
            self._send_sse_event(
                json.dumps({"error": "LLM not configured. Set API Key in Settings."}),
                event="error"
            )
            self._send_sse_event("[DONE]", event="done")
            return

        context_docs = []
        if use_rag:
            try:
                context_docs = engine.query(query, top_k=top_k, threshold=0.3)
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)

        self._send_sse_event(json.dumps({
            "context_count": len(context_docs),
            "contexts": [{"content": d["content"][:120] + "...", "score": d["score"]} for d in context_docs[:3]],
        }, ensure_ascii=False), event="context")

        if context_docs:
            parts = []
            for i, doc in enumerate(context_docs, 1):
                parts.append(f"[Ref {i}] (score: {doc['score']:.2f})\n{doc['content']}")
            context_block = "\n\n---\n\n".join(parts)
            sys_prompt = (
                "You are a knowledge base assistant. Answer based on the provided reference documents. "
                "If the documents are insufficient, say so honestly. "
                "Be concise and accurate. Reply in the user's language."
            )
            user_msg = f"Reference:\n\n{context_block}\n\n---\n\nQuestion: {query}"
        else:
            sys_prompt = "You are a helpful assistant. Reply in the user's language."
            user_msg = query

        msgs = [{"role": "system", "content": sys_prompt}]
        msgs.extend(history)
        msgs.append({"role": "user", "content": user_msg})

        try:
            full_text = []
            deadline = time.time() + 300  # 5-min timeout
            for chunk in llm._stream_request("chat/completions", {
                "model": body.get("model") or llm.model,
                "messages": msgs,
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "stream": True,
            }):
                if time.time() > deadline:
                    self._send_sse_event(json.dumps({"error": "响应超时"}), event="error")
                    break
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text.append(token)
                    self._send_sse_event(token, event="token")

            full = "".join(full_text)
            self._send_sse_event(json.dumps({
                "full_text": full,
                "context_count": len(context_docs),
            }, ensure_ascii=False), event="result")

        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send_sse_event(json.dumps({"error": str(e)}), event="error")
            except BrokenPipeError:
                pass

        self._send_sse_event("[DONE]", event="done")


def main():
    global engine, llm, _engine_load_time

    print("=" * 50)
    print("  RAG Knowledge Base v2.4")
    print("=" * 50)

    logger.info("=" * 50)
    logger.info("  RAG Knowledge Base v2.4 starting")
    logger.info("=" * 50)

    # 端口与绑定地址
    # 默认绑定本机回环地址，防止局域网/公网任意访问（信息泄露、SSRF、路径读取等风险）。
    # 如确需局域网访问，可设置环境变量 RAG_HOST=0.0.0.0 覆盖。
    port = 8899
    host = os.environ.get("RAG_HOST", "127.0.0.1")

    # 1. 先在主线程加载引擎（此时不能有其它线程运行）。
    # ⚠️ torch / sentence_transformers 的 CUDA 初始化有严格的线程要求：
    #   - 在「非主线程」加载模型 → 段错误（segfault），整个进程崩溃；
    #   - 在有其它线程（如 HTTP 服务线程）同时运行时加载 → CUDA 初始化失败
    #     （报 "Found no NVIDIA driver"）并崩溃。
    # 两种情况下 HTTP 服务器都会随之消失，浏览器报 ERR_CONNECTION_REFUSED。
    # 因此必须先在主线程、无其它线程的情况下把模型加载完，再启动 HTTP 服务器。
    def _load_engine():
        global engine, llm, _engine_load_time, _engine_error
        t0 = time.time()
        _engine_load_time = time.time()  # marker for elapsed tracking
        try:
            logger.info("Loading config from %s", CONFIG_PATH)
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logger.info("Checking PostgreSQL...")
            ok, msg = ensure_pg(config)
            logger.info("PG status: %s", msg)
            print(f"  PG: {msg}")

            logger.info("Loading RAG engine (BGE-M3)...")
            print("  Loading RAG engine...")
            engine = RAGEngine(config)
            logger.info("Engine loaded: %dd, device=%s", engine.embedder.dimension, engine.embedder.device)
            print(f"  BGE-M3: {engine.embedder.dimension}d, device={engine.embedder.device}")

            logger.info("Initializing LLM client...")
            llm = LLMClient(config)
            if llm.enabled:
                logger.info("LLM configured: %s @ %s", llm.model, llm.api_base)
                print(f"  LLM: {llm.model} @ {llm.api_base}")
            else:
                logger.warning("LLM not configured")
                print("  LLM: not configured")

            _engine_load_time = round(time.time() - t0, 1)
            _engine_ready.set()
            logger.info("Engine ready in %.1fs", _engine_load_time)
            print(f"  Ready in {_engine_load_time}s")
        except Exception as e:
            _engine_error = str(e)
            logger.exception("Engine load failed: %s", e)
            print(f"  ERROR: {e}")

    _load_engine()

    # 2. 引擎就绪后启动 HTTP 服务器（主线程）
    server = ThreadingHTTPServer((host, port), APIHandler)
    print(f"\nHTTP server listening on http://localhost:{port} (bind={host})")
    logger.info(f"HTTP server on {host}:{port}")

    # 服务已就绪，自动打开浏览器（避免用户在服务未启动时看到连接被拒绝）
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        logger.warning("自动打开浏览器失败", exc_info=True)

    def _graceful_shutdown(signum=None, frame=None):
        logger.info("Signal %s received, shutting down", signum)
        try:
            server.shutdown()
        except Exception:
            pass

    atexit.register(_graceful_shutdown)
    try:
        signal.signal(signal.SIGTERM, _graceful_shutdown)
    except (ValueError, OSError):
        # 非主线程或平台不支持时忽略（正常情况 main() 即主线程）
        pass

    print(f"  http://localhost:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        logger.info("Shutdown signal received")
        try:
            server.shutdown()
        except Exception:
            pass
        if engine:
            engine.close()
            logger.info("Engine closed")
        logger.info("Shutdown complete")
        print("Done.")


if __name__ == "__main__":
    main()
