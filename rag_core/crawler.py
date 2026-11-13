"""
通用网页爬虫模块 — 多引擎融合提取，零站点特化

核心架构（全通用，不区分「新闻站」「教程站」）：
  URL → fetch → 并行提取管道:
    1. BS4 直接提取 <meta>/<title>/<h1>（基础元数据，不依赖 structured data）
    2. trafilatura favor_recall（正文主力，宁可多取不少取）
    3. readability-lxml（备选，trafilatura 失败或结果过短时启用）
    4. extruct 如果站点有结构化元数据则补充（OG/JSON-LD/Microdata）
       → 结果融合（最长正文优先）→ 清洗 → 质量分级 → 类型识别 → 关键词 → 实体
"""

import os, re, logging, time, random, hashlib
from typing import Optional, List, Dict
from urllib.parse import urlparse, urljoin, urlunparse

logger = logging.getLogger(__name__)

logging.getLogger("trafilatura").setLevel(logging.WARNING)
logging.getLogger("readability").setLevel(logging.WARNING)
logging.getLogger("newspaper").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "sources.yaml")

_DEFAULT_SOURCES = {
    "sources": [
        {"name": "界面新闻", "url": "https://www.jiemian.com", "category": "新闻",
         "list_url": "https://www.jiemian.com/lists/4.html",
         "article_pattern": r"/article/\d+\.html"},
        {"name": "36氪快讯", "url": "https://36kr.com", "category": "科技",
         "list_url": "https://36kr.com/newsflashes",
         "article_pattern": r"/newsflashes/\d+"},
        {"name": "IT之家", "url": "https://www.ithome.com", "category": "科技",
         "list_url": "https://www.ithome.com/",
         "article_pattern": r"https?://www\.ithome\.com/0/\d+/\d+\.htm"},
    ]
}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
]

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

CONTENT_TYPE_RULES = [
    ("新闻资讯", re.compile(r"报道|记者|快讯|消息来源|通讯员|本报", re.IGNORECASE)),
    ("技术教程", re.compile(r"步骤[一二三四五]|首先[,，].*然后[,，]|代码|安装|配置|运行|教程|指南", re.IGNORECASE)),
    ("问答讨论", re.compile(r"问题[:：]|回答[:：]|提问|@\w+|赞同\d|踩|评论", re.IGNORECASE)),
    ("产品评测", re.compile(r"评分|优缺点|性价比|对比|评测|测评|推荐指数", re.IGNORECASE)),
    ("官方公告", re.compile(r"通知|声明|公告|文件号|公章|批准|决定|函", re.IGNORECASE)),
    ("学术论文", re.compile(r"摘要|引言|参考文献|DOI|结论|致谢|引用", re.IGNORECASE)),
]

TAG_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "这个", "那个",
    "可以", "什么", "怎么", "如果", "因为", "所以", "但是", "而且", "或者",
}

JUNK_LINE_PATTERNS = [
    re.compile(r"^(版权所有|未经许可|不得转载|侵权必究|All Rights? Reserved)", re.IGNORECASE),
    re.compile(r"^(分享到|举报|推荐阅读|相关文章|热门文章|猜你喜欢|大家都在看)"),
    re.compile(r"^(广告|推广|赞助|Advert)"),
    re.compile(r"^(阅读原文|查看更多|点击上方|扫码|长按识别)"),
    re.compile(r"^(\d+人赞|\d+人阅读|\d+评论|\d+收藏)"),
    re.compile(r"^(免责声明|风险提示|投资有风险)"),
    re.compile(r"^\s*$"),
    re.compile(r"^(首页|科技|金融|证券|地产|汽车|健康|地方|硬科技|文旅|数据|ESG|双碳|电厂|商业|消费|工业|时尚|交通|医药|互联网|创投|能源|数码|教育|食品|新能源|家电|酒业|物流|零售|美妆|体育|楼市|家居|餐饮|日用|企服|珠宝|腕表|智库|农业|出海|财经|投资|IPO|宏观|股市|财富|天下|中国|评论|职场|国是|文娱|影像|营销|大湾区|长三角|中西部|文化生活|文化|生活方式|美食美酒|艺术|游戏|正午|箭厂|VIP|快报|今日热点|公司头条|股市前沿|监管通报|财经速览|时事追踪|界面号|财经号|城市号|媒体号|视频|专题|Global|登录|消息|我的面点|我的关注|我的文章|投稿|报料|账号设置|退出账号|旧版搜索|新版搜索|历史搜索|全部删除|热门搜索|金融快讯)$"),
    re.compile(r"^(点赞|收藏|看评论|分享至|微博分享|微信分享|QQzone|评论|暂无评论|快来评价一下吧)$"),
    re.compile(r"^未经正式授权严禁转载本文，侵权必究$"),
    re.compile(r"^界面快报$"),
    re.compile(r"^来源：.+$"),
    re.compile(r"^\d+月\d+日$"),
    re.compile(r"^热门排行$"),
    re.compile(r"^(关于我们|联系方式|加入我们|人才招聘|商务合作|广告合作|媒体合作|合作伙伴|友情链接|网站地图|使用条款|隐私政策|法律声明|免责条款|服务协议|用户协议)$"),
    re.compile(r"^(注册|登录|登出|注销|忘记密码|重置密码|切换账号|绑定账号|会员中心|个人中心|用户中心|账号管理)$"),
    re.compile(r"^(搜索|高级搜索|热门搜索|搜索推荐|历史搜索|清除记录|全部删除|确认删除|取消)$"),
    re.compile(r"^(首页|上一页|下一页|末页|尾页|返回顶部|返回列表|返回首页|回到顶部|加载更多|查看更多|展开全部|收起全部|显示更多|更多>>|更多\.\.\.)$"),
    re.compile(r"^(订阅|关注|点赞|分享|转发|收藏|举报|投诉|反馈|建议|帮助|帮助中心|常见问题|在线客服|联系我们|意见反馈)$"),
    re.compile(r"^(下载APP|扫码下载|关注公众号|关注微信|关注微博|扫码关注|二维码|小程序|立即下载|免费下载|点击下载)$"),
    re.compile(r"^(栏目导航|网站导航|频道导航|内容导航|快捷导航|导航菜单|主导航|子导航|侧边栏|边栏|面包屑)$"),
    re.compile(r"^(您当前的位置|当前位置|所在位置|文章位置|频道位置)$"),
    re.compile(r"^(责任编辑|编辑|责任编辑[:：].*|实习编辑|责任编辑\s.+|编辑\s*[:：].*|记者\s*[:：].*|作者\s*[:：].*|来源\s*[:：].*|摄影\s*[:：].*|视频\s*[:：].*)$"),
    re.compile(r"^(字体|大|中|小|默认|打印|关闭|放大|缩小|分享|复制|转发|收藏|保存|下载)$"),
    re.compile(r"^(本文\s*[:：]|本文来源于|本文来自|本文转自|文章来源|文章来自|稿件来源|图文来源|综合来源|内容来源|数据来源)$"),
    re.compile(r"^(相关推荐|热点推荐|热门推荐|最新文章|相关阅读|延伸阅读|深度阅读|推荐阅读|猜您喜欢|看了还看|为您推荐|你可能感兴趣|大家都在看|热门排行|最新发布|最热发布|精华推荐)$"),
    re.compile(r"^(评论|留言|我要评论|发表评论|查看评论|全部评论|热门评论|最新评论|条评论|人参与|人评论)$"),
    re.compile(r"^(加载中|加载失败|暂无数据|暂无内容|暂无评论|暂无更多|没有更多了|已经到底了|到底了|没有数据)$"),
    re.compile(r"^\[[图文视频音频]+\]$"),
    re.compile(r"^(图\s*[:：]|表\s*[:：]|图\d+|表\d+|图\s*\d+[:：]|表\s*\d+[:：])$"),
    re.compile(r"^(上一条|下一条|上一篇|下一篇|上一页|下一页|上一个|下一个|上一张|下一张|上一章|下一章|上一节|下一节)$"),
    re.compile(r"^(微信|微博|QQ|空间|豆瓣|贴吧|知乎|豆瓣|FaceBook|Twitter|WhatsApp|Telegram|Line|微信好友|朋友圈|QQ空间|新浪微博|腾讯微博|复制链接|分享到)$"),
    re.compile(r"^(扫描二维码|扫码关注|二维码|微信扫码|支付宝扫码|付款码|收款码)$"),
    re.compile(r"^( copyright | all rights reserved | powered by | designed by | made with )$", re.IGNORECASE),
    re.compile(r"^(回复|查看|展开|收起|折叠|展开全文|查看全文|阅读全文|显示全文|完整内容|查看详情|查看详细信息|了解更多|详细信息|详情请见|详情请点击|点击查看|点击了解更多)$"),
    re.compile(r"^(本文约\d+字|阅读约需\d+分钟|预计阅读\d+分钟|阅读时间\d+分钟|全文约\d+字|字数\s*[:：]\s*\d+)$"),
    re.compile(r"^(首页.*首页|频道.*频道|专题.*专题)$"),
]

CUSTOM_ENTITY_DICT = [
    "习近平", "李克强", "李强", "王沪宁", "赵乐际", "丁薛祥", "李希", "蔡奇",
    "何立峰", "王毅", "布林肯", "普京", "拜登", "特朗普", "马斯克",
    "雷军", "马云", "马化腾", "任正非", "张一鸣", "王传福", "曾毓群",
    "李彦宏", "周鸿祎", "刘强东", "董明珠", "黄仁勋", "库克",
    "马克龙", "朔尔茨", "内塔尼亚胡", "埃尔多安", "莫迪",
    "中共中央", "国务院", "全国人大", "全国政协", "中央军委",
    "发改委", "商务部", "财政部", "外交部", "教育部", "证监会", "银保监会",
    "新华社", "人民日报", "中央电视台",
    "新质生产力", "高质量发展", "中国式现代化", "一带一路", "碳达峰", "碳中和",
    "长三角", "大湾区", "京津冀", "粤港澳",
]


def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                           "utm_content", "from", "source", "ref", "spm", "scm"}
        query_parts = []
        if parsed.query:
            for pair in parsed.query.split("&"):
                if "=" in pair:
                    key = pair.split("=", 1)[0]
                    if key.lower() not in tracking_params:
                        query_parts.append(pair)
                else:
                    query_parts.append(pair)
        new_query = "&".join(query_parts)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path,
                           parsed.params, new_query, ""))
    except Exception:
        return url


def _simhash(text: str, hash_bits: int = 64) -> int:
    if not text:
        return 0
    v = [0] * hash_bits
    tokens = text.split()
    if len(tokens) < 10:
        tokens = list(text)
    for token in tokens[:500]:
        h = int(hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
        for i in range(hash_bits):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    result = 0
    for i in range(hash_bits):
        if v[i] > 0:
            result |= (1 << i)
    return result


def _hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    d = 0
    while x:
        d += 1
        x &= x - 1
    return d


def validate_public_url(url: str) -> tuple:
    """校验 URL 是否允许抓取，防止 SSRF（内网探测 / 读取本地文件等）。

    规则：
      1. 仅允许 http / https 协议；
      2. 解析后的 hostname 必须存在；
      3. 阻断内网 / 回环 / 链路本地等地址。

    Returns:
        (是否安全: bool, 错误信息: str)；安全时错误信息为空字符串。
    """
    if not url or not isinstance(url, str):
        return False, "URL 为空或非法"

    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "URL 解析失败"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"仅支持 http/https 协议，当前为 {scheme!r}"

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL 缺少有效主机名"

    # 阻断内网 / 回环 / 链路本地 / 保留地址，防止 SSRF
    import ipaddress
    blocked_prefixes = [
        "127.",          # 127.0.0.0/8 回环
        "10.",           # 10.0.0.0/8 内网
        "0.",            # 0.0.0.0/8
        "169.254.",      # 链路本地
        "192.0.0.",
        "192.0.2.",
        "198.18.",
        "198.51.100.",
        "203.0.113.",
        "100.64.",       # CGNAT
    ]
    if host == "localhost" or host.endswith(".local") or host == "::1":
        return False, "禁止访问本地/回环地址"
    if any(host.startswith(p) for p in blocked_prefixes):
        return False, "禁止访问内网/保留地址"
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return False, "禁止访问内网地址（172.16.0.0/12）"
        except (ValueError, IndexError):
            pass
    if host.startswith("192.168."):
        return False, "禁止访问内网地址（192.168.0.0/16）"
    # IPv6 及十进制/十六进制 IP 形式统一交给 ipaddress 判定
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified or ip.is_multicast:
            return False, "禁止访问内网/保留 IP 地址"
    except ValueError:
        pass  # 非 IP 字面量（正常域名），放行

    return True, ""


class CrawlerManager:
    """通用网页爬虫管理器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or CONFIG_PATH
        self._sources = None

    def _get_headers(self) -> dict:
        h = DEFAULT_HEADERS.copy()
        h["User-Agent"] = random.choice(_USER_AGENTS)
        return h

    def _create_session(self, max_retries: int = 2) -> "requests.Session":
        """创建带连接池和重试的 requests.Session"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_sources(self) -> list:
        if self._sources is not None:
            return self._sources
        try:
            import yaml
            if os.path.isfile(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self._sources = data.get("sources", [])
            else:
                self._sources = _DEFAULT_SOURCES["sources"]
                self._save_sources()
        except Exception as e:
            logger.warning("Failed to load sources config: %s", e)
            self._sources = _DEFAULT_SOURCES["sources"]
        return self._sources

    def add_source(self, name: str, url: str, category: str = "通用",
                   list_url: str = None, article_pattern: str = None,
                   content_selector: str = None, settings: dict = None):
        sources = self.get_sources()
        entry = {"name": name, "url": url.rstrip("/"), "category": category,
                 "settings": settings or {}}
        if list_url:
            entry["list_url"] = list_url
        if article_pattern:
            entry["article_pattern"] = article_pattern
        if content_selector:
            entry["content_selector"] = content_selector
        for i, s in enumerate(sources):
            if s["name"] == name:
                entry["settings"] = settings or s.get("settings", {})
                sources[i] = entry
                self._save_sources()
                return
        sources.append(entry)
        self._save_sources()

    def remove_source(self, name: str) -> bool:
        sources = self.get_sources()
        orig = len(sources)
        self._sources = [s for s in sources if s["name"] != name]
        if len(self._sources) < orig:
            self._save_sources()
            return True
        return False

    def _save_sources(self):
        try:
            import yaml
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump({"sources": self._sources}, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            logger.error("Failed to save sources: %s", e)

    @staticmethod
    def _is_list_page(html: str, url: str, soup_obj) -> bool:
        """Detect list/archive/index pages. Article detail pages are NOT flagged."""
        article_url_patterns = [
            r'/article/', r'/news/\d', r'/story/', r'/post/',
            r'/detail/', r'/\d{5,}\.html',
            r'/p/\d+', r'content_\d+', r'/a/\d+',
            r'/newsflashes/\d+',
        ]
        if any(re.search(p, url) for p in article_url_patterns):
            return False

        list_url_patterns = [
            r'/lists?/', r'/channel/', r'/category/', r'/archive/',
            r'/tag/', r'/topic/', r'/column/', r'/index\.',
            r'/newsflashes(?:/|$)', r'/telegraph', r'/explore',
        ]
        if any(re.search(p, url) for p in list_url_patterns):
            return True

        if soup_obj and soup_obj.title and soup_obj.title.string:
            t = soup_obj.title.string.strip().lower()
            if t in ('快报', '首页', '列表', '目录', 'index', 'home', 'archive',
                       '发现', 'explore'):
                return True

        text_len = len(soup_obj.get_text() if soup_obj else html)
        if text_len > 0:
            link_count = len(soup_obj.find_all('a', href=True)) if soup_obj else html.count('<a ')
            if link_count > 80 and text_len / max(link_count, 1) < 25:
                return True

        body_text = soup_obj.get_text() if soup_obj else html
        timestamp_lines = len(re.findall(r'\d{1,2}:\d{2}\s', body_text))
        if timestamp_lines >= 8:
            return True

        return False

    def _fetch_with_retry(self, url: str, timeout: int = 20, max_retries: int = 2) -> "requests.Response":
        """封装带重试的 HTTP 请求"""
        import requests
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, headers=self._get_headers(),
                                    timeout=timeout, allow_redirects=True)
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt < max_retries:
                    wait = 0.5 * (attempt + 1)
                    logger.warning("Fetch retry %d/%d for %s: %s, waiting %.1fs",
                                   attempt + 1, max_retries, url, e, wait)
                    time.sleep(wait)
                else:
                    raise
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if 500 <= status < 600 and attempt < max_retries:
                    wait = 0.5 * (attempt + 1)
                    logger.warning("HTTP %d retry %d/%d for %s, waiting %.1fs",
                                   status, attempt + 1, max_retries, url, wait)
                    time.sleep(wait)
                    last_err = e
                else:
                    raise
        raise last_err

    def crawl_url(self, url: str, timeout: int = 20, content_selector: str = None) -> dict:
        """
        通用多引擎融合提取单个URL。

        提取管线（不区分站点类型）：
          1. BS4 扫描 <meta>/<title>/<h1> → 基础元数据
          2. trafilatura favor_recall → 正文
          3. readability-lxml → 备选正文（trafilatura 结果 <100 字时）
          4. extruct → 结构化元数据增强（如果站点提供）
          5. 结果融合 → 清洗 → 分级 → 分类 → 标签 → 实体

        如果 content_selector 提供，先用 BeautifulSoup select 提取正文区域。
        列表页检测改为标记而非阻断：如果检测到列表页，标记 is_list_page 但仍尝试提取。
        """
        # SSRF 防护：拒绝非 http/https、内网/回环地址
        safe, err = validate_public_url(url)
        if not safe:
            logger.warning("Rejected unsafe URL: %s (%s)", url, err)
            return {
                "url": url, "title": "", "content": "", "preview": "",
                "date": "", "domain": "", "content_type": "未分类",
                "quality": "C", "tags": [], "entities": {},
                "word_count": 0, "content_hash": "", "success": False,
                "error": err, "is_list_page": False,
            }

        result = {
            "url": _normalize_url(url),
            "title": "", "content": "", "preview": "",
            "date": "", "domain": "",
            "content_type": "未分类", "quality": "C",
            "tags": [], "entities": {},
            "word_count": 0, "content_hash": "",
            "success": False, "error": "",
            "is_list_page": False,
        }

        try:
            result["domain"] = urlparse(url).netloc
        except Exception:
            pass

        # Step 1: Fetch（带重试 + 自动编码检测）
        try:
            resp = self._fetch_with_retry(url, timeout=timeout, max_retries=2)
            # 使用 requests 内置的 apparent_encoding 自动检测编码
            resp.encoding = resp.apparent_encoding or 'utf-8'
            html = resp.text
        except Exception as e:
            result["error"] = f"请求失败: {e}"
            return result

        try:
            result["url"] = _normalize_url(resp.url or url)
        except Exception:
            pass

        # Step 2: BS4 基础元数据提取
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Step 2b: 列表页检测 — 标记但不阻断，仍尝试提取
        if self._is_list_page(html, result["url"], soup):
            result["is_list_page"] = True
            logger.warning("URL appears to be a list page, will still attempt extraction: %s", url)

        meta = self._extract_meta_bs4(soup)

        # Step 2c: 如果提供了 content_selector，用 BS4 select 提取正文区域
        selector_content = ""
        if content_selector and content_selector.strip():
            try:
                elements = soup.select(content_selector)
                if elements:
                    for el in elements:
                        selector_content += el.get_text(separator="\n", strip=True) + "\n"
                    logger.info("content_selector '%s' extracted %d chars", content_selector, len(selector_content))
                else:
                    logger.warning("content_selector '%s' matched nothing", content_selector)
            except Exception as e:
                logger.warning("content_selector error: %s", e)

        # Step 3: trafilatura → 正文提取 (favor_recall: 宁可多取，不少取)
        import trafilatura
        content = ""
        title = meta.get("title", "")
        try:
            t = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                output_format="txt",
                favor_recall=True,
            )
            if t:
                content = t.strip()
        except Exception as e:
            logger.debug("trafilatura failed: %s", e)

        # Step 4: readability-lxml → 备选正文
        content_readability = ""
        title_readability = ""
        try:
            from readability import Document
            doc = Document(html)
            title_readability = doc.title()
            raw = doc.summary()
            soup2 = BeautifulSoup(raw, "html.parser")
            content_readability = soup2.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.debug("readability failed: %s", e)

        # Step 5: newspaper3k → 第三备选引擎
        content_newspaper = ""
        try:
            from newspaper import Article as NPArticle
            np_art = NPArticle(url)
            np_art.download(input_html=html)
            np_art.parse()
            if np_art.text:
                content_newspaper = np_art.text
            if not title and np_art.title:
                title = np_art.title
        except Exception as e:
            logger.debug("newspaper3k failed: %s", e)

        # Step 6: 正文融合 — 优先 content_selector，其次按质量评分选最佳
        if selector_content and len(selector_content) >= 50:
            content = selector_content
        else:
            candidates = [
                (content, title),
                (content_readability, title_readability),
                (content_newspaper, title),
            ]
            scored = [(txt, t, self._score_content(txt)) for txt, t in candidates]
            best = max(scored, key=lambda x: x[2])
            if best[2] > 0:
                content = best[0]
                if not title and best[1]:
                    title = best[1]

        # Step 7: extruct → 结构化元数据增强
        if not meta.get("date"):
            try:
                import extruct, json
                data = extruct.extract(html, base_url=result["url"],
                                       syntaxes=["opengraph", "json-ld"])
                og_list = data.get("opengraph", [])
                og = og_list[0] if isinstance(og_list, list) and og_list else (
                    og_list if isinstance(og_list, dict) else {}
                )
                for ld in data.get("json-ld", []):
                    if isinstance(ld, dict):
                        for k in ("headline", "datePublished", "dateModified", "author"):
                            if ld.get(k) and not meta.get(k):
                                meta[k] = ld[k]
                if og.get("title") and not title:
                    title = og["title"]
                if og.get("description") and not meta.get("description"):
                    meta["description"] = og["description"]
                if og.get("article:published_time"):
                    meta.setdefault("date", og["article:published_time"])
            except Exception as e:
                logger.debug("extruct failed: %s", e)

        # Step 8: 内容清洗
        content = self._clean_content(content)
        title = self._clean_title(title or meta.get("title", ""))
        if not title:
            title = self._clean_title(soup.title.string if soup.title else "")

        result["title"] = title or "无标题"
        result["content"] = content
        result["preview"] = content[:300] if content else ""
        result["date"] = meta.get("date", "")
        result["word_count"] = len(content)

        # 质量分级
        wc = result["word_count"]
        has_title = bool(title and title != "无标题")
        has_date = bool(result["date"])
        if wc >= 500 and has_title:
            result["quality"] = "A"
        elif wc >= 200:
            result["quality"] = "B"
        else:
            result["quality"] = "C"

        result["content_type"] = self._identify_type(content, title)
        if content:
            result["tags"] = self._extract_tags(content, top_n=5)
            result["entities"] = self._extract_entities(content)
        if content:
            result["content_hash"] = str(_simhash(content))

        result["success"] = bool(content.strip())
        if not result["success"]:
            result["error"] = "未能提取有效内容"

        return result

    def _extract_meta_bs4(self, soup) -> dict:
        """BS4 通用元数据提取 — 扫描 <meta>/<title>/<h1>，不依赖结构化标记"""
        meta = {}
        if soup.title and soup.title.string:
            meta["title"] = soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and not meta.get("title"):
            meta["title"] = h1.get_text(strip=True)
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publishdate"]',
            'meta[name="pubdate"]',
            'meta[name="pub_date"]',
            'meta[name="date"]',
            'meta[name="DC.date"]',
            'meta[name="weibo:article:create_at"]',
            'meta[itemprop="datePublished"]',
            'time[datetime]',
            'time[pubdate]',
        ]
        for sel in date_selectors:
            el = soup.select_one(sel)
            if el:
                val = el.get("content") or el.get("datetime") or el.get_text(strip=True)
                if val:
                    meta["date"] = val.strip()
                    break
        desc_sel = soup.select_one('meta[name="description"], meta[property="og:description"]')
        if desc_sel:
            meta["description"] = desc_sel.get("content", "")
        author_sel = soup.select_one('meta[name="author"], meta[property="article:author"]')
        if author_sel:
            meta["author"] = author_sel.get("content", "")
        return meta

    def crawl_source_advanced(
        self, source: dict, settings: dict = None,
        max_pages: int = 3, max_articles: int = 50,
        with_content: bool = True,
        progress_callback=None, cancel_check=None,
    ) -> List[dict]:
        """
        翻页爬取：发现文章链接 → 逐篇多引擎提取。

        这是唯一实现，`crawl_source_advanced_with_progress` 是其薄封装。
        可选参数：
          progress_callback: 接收进度数据字典的回调
          cancel_check: 可选 callable，返回 True 时取消爬取
        """
        settings = settings or {}
        timeout = int(settings.get("timeout", 20))
        delay = float(settings.get("delay", 1.5))
        min_quality = settings.get("min_quality", "C")
        content_selector = settings.get("content_selector") or source.get("content_selector", "")
        max_pages = int(settings.get("max_pages", max_pages))
        max_articles = int(settings.get("max_articles", max_articles))

        from bs4 import BeautifulSoup

        raw_list = settings.get("list_url") or source.get("list_url", "")
        source_url = source.get("url", "")
        parsed = urlparse(raw_list or source_url)
        if not parsed.path or parsed.path in ("/", ""):
            list_url = source_url
        else:
            list_url = raw_list
        article_pattern = source.get("article_pattern", "") or settings.get("article_pattern", "")
        all_urls: List[str] = []

        # 使用带连接池和重试的 Session
        session = self._create_session(max_retries=2)

        if progress_callback:
            progress_callback({
                "type": "phase", "phase": "discovering",
                "message": f"正在发现链接（最多 {max_pages} 页）...",
            })

        cancelled = False
        for page in range(1, max_pages + 1):
            if cancel_check and cancel_check():
                cancelled = True
                break
            if progress_callback:
                progress_callback({
                    "type": "page_discover", "current_page": page,
                    "total_pages": max_pages, "found_so_far": len(all_urls),
                })
            pl = self._build_page_url(list_url, page)
            # SSRF 防护：list_url 可能来自用户添加的源，同样校验
            safe, err = validate_public_url(pl)
            if not safe:
                logger.warning("Rejected unsafe list_url: %s (%s)", pl, err)
                if progress_callback:
                    progress_callback({
                        "type": "page_error", "current_page": page, "error": err,
                    })
                break
            try:
                r = session.get(pl, headers=self._get_headers(), timeout=timeout)
                r.encoding = r.apparent_encoding or 'utf-8'
                html = r.text
            except Exception as e:
                logger.warning("List page %s failed: %s", pl, e)
                if progress_callback:
                    progress_callback({
                        "type": "page_error", "current_page": page, "error": str(e),
                    })
                break

            if article_pattern:
                found = list(set(re.findall(article_pattern, html)))
                if not found:
                    logger.info("No article links found on page %d, stopping", page)
                    break
                base_url = re.match(r"(https?://[^/]+)", list_url)
                base = base_url.group(1) if base_url else source.get("url", "")
                urls = [urljoin(base, u) for u in found]
            else:
                soup = BeautifulSoup(html, "html.parser")
                urls = list(set(
                    a["href"] for a in soup.find_all("a", href=True)
                    if re.search(r"/article/\d+|/\d+\.html|/news/\d+|newsDetail_forward_|[a-z]+Detail", a["href"])
                ))
                base_url = source.get("url", "").rstrip("/")
                urls = [u if u.startswith("http") else urljoin(base_url, u) for u in urls]

            new_urls = [u for u in urls if u not in all_urls]
            all_urls.extend(new_urls)
            logger.info("Page %d: %d new links (total %d)", page, len(new_urls), len(all_urls))
            if len(all_urls) >= max_articles:
                all_urls = all_urls[:max_articles]
                break
            time.sleep(delay + random.uniform(0, 0.5))

        if cancelled:
            if progress_callback:
                progress_callback({"type": "cancelled", "message": "用户取消爬取"})
            if not all_urls:
                return [{"success": False, "error": "用户取消爬取"}]
            # 有部分结果时继续返回已发现的文章

        if not all_urls:
            return [{"success": False, "error": "未能发现文章链接"}]

        if progress_callback:
            progress_callback({
                "type": "phase", "phase": "crawling",
                "message": f"开始抓取 {len(all_urls)} 篇文章...",
                "total_articles": len(all_urls),
            })

        results = []
        seen_hashes = set()
        total_urls = len(all_urls)

        for idx, url in enumerate(all_urls):
            if cancel_check and cancel_check():
                cancelled = True
                if progress_callback:
                    progress_callback({"type": "cancelled", "message": "用户取消爬取"})
                break
            if progress_callback:
                progress_callback({
                    "type": "article_progress", "current": idx + 1,
                    "total": total_urls, "url": url,
                    "percent": round((idx + 1) / total_urls * 100, 1),
                })
            if len(results) >= max_articles:
                break
            # 每篇文章用 crawl_url 抓取（内部已带重试）
            article = self.crawl_url(url, timeout=timeout, content_selector=content_selector or None)
            if article.get("success"):
                ch = article.get("content_hash", "")
                if ch:
                    is_dup = any(_hamming_distance(int(ch), int(e)) <= 3 for e in seen_hashes)
                    if is_dup:
                        if progress_callback:
                            progress_callback({"type": "article_skipped", "reason": "duplicate", "url": url})
                        continue
                    seen_hashes.add(ch)
                article["source_name"] = source.get("name", "")
                article["source_category"] = source.get("category", "通用")
                results.append(article)
                if progress_callback:
                    progress_callback({
                        "type": "article_success", "title": article.get("title", ""),
                        "quality": article.get("quality", ""), "word_count": article.get("word_count", 0),
                    })
            else:
                results.append({"url": url, "success": False,
                                "error": article.get("error", "解析失败")})
                if progress_callback:
                    progress_callback({
                        "type": "article_failed", "url": url,
                        "error": article.get("error", "解析失败"),
                    })
            time.sleep(delay)

        # 质量过滤
        quality_order = {"A": 3, "B": 2, "C": 1}
        min_q_val = quality_order.get(min_quality, 0)
        if min_q_val > 0:
            filtered = []
            for r in results:
                if not r.get("success"):
                    filtered.append(r)  # 保留失败项用于统计
                    continue
                q = quality_order.get(r.get("quality", "C"), 1)
                if q >= min_q_val:
                    filtered.append(r)
                else:
                    logger.info("Filtered by quality: %s (quality=%s, min=%s)",
                                r.get("url", ""), r.get("quality", ""), min_quality)
            results = filtered

        if progress_callback:
            summary = {
                "type": "summary", "total_found": total_urls,
                "success_count": sum(1 for r in results if r.get("success")),
                "filtered_count": len(results),
            }
            if cancelled:
                summary["cancelled"] = True
            progress_callback(summary)

        logger.info("Crawl complete: %d/%d successful%s",
                     sum(1 for r in results if r.get("success")), len(all_urls),
                     " (cancelled)" if cancelled else "")
        return results

    def crawl_source_advanced_with_progress(
        self, source: dict, settings: dict = None,
        max_pages: int = 3, max_articles: int = 50,
        progress_callback=None, cancel_check=None
    ) -> List[dict]:
        """翻页爬取（带进度回调）—— `crawl_source_advanced` 的薄封装，保持向后兼容。"""
        return self.crawl_source_advanced(
            source, settings=settings,
            max_pages=max_pages, max_articles=max_articles,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )

    @staticmethod
    def _build_page_url(list_url: str, page: int) -> str:
        """构建翻页 URL，支持 {page} 占位符、?page= 模式、&page= 模式"""
        if "{page}" in list_url:
            return list_url.replace("{page}", str(page))
        # 如果 URL 已有 page 参数，替换其值
        page_match = re.search(r'([?&]page=)\d+', list_url)
        if page_match:
            return list_url[:page_match.start(1) + len(page_match.group(1))] + str(page) + list_url[page_match.end():]
        # 如果 URL 有查询参数，追加 &page=
        if "?" in list_url:
            return f"{list_url}&page={page}"
        # 否则追加 ?page=
        return f"{list_url}?page={page}"

    @staticmethod
    def _score_content(text: str) -> float:
        """评估正文质量分数：段落结构、平均段落长度、标点密度、中文占比"""
        if not text:
            return 0.0
        score = 0.0
        paragraphs = [p for p in text.split('\n') if p.strip()]
        score += min(len(paragraphs), 10) * 2
        avg_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if 100 < avg_len < 800:
            score += 10
        punct_count = len(re.findall(r'[，。！？；：、]', text))
        score += min(punct_count, 20)
        cn_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        if cn_count / max(len(text), 1) > 0.4:
            score += 5
        return score

    @staticmethod
    def _clean_content(text: str) -> str:
        """
        通用内容清洗 — 过滤导航菜单、互动元素、垃圾行。
        """
        if not text:
            return ""

        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\u00a0\u3000]", " ", text)
        text = re.sub(r"\r\n|\r", "\n", text)

        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if any(pat.match(stripped) for pat in JUNK_LINE_PATTERNS):
                continue
            cn_words = re.findall(r'[\u4e00-\u9fff]{2,4}', stripped)
            if len(cn_words) >= 6 and len(stripped) < 40:
                continue
            if re.match(r'^[a-zA-Z\s\.\-]{2,30}$', stripped):
                continue
            lines.append(stripped)

        text = "\n".join(lines)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r" {3,}", "  ", text)
        return text.strip()

    @staticmethod
    def _clean_title(title: str) -> str:
        if not title:
            return ""
        title = re.sub(r"\s+", " ", title).strip()
        title = re.sub(r"\s*[|_—–-]\s*(.{2,20}网|.{2,15}新闻|.{2,10}首页)$", "", title)
        title = re.sub(r"^\s*(.{2,20}网|.{2,15}新闻)\s*[|_—–-]\s*", "", title)
        return title.strip()[:200]

    @staticmethod
    def _identify_type(text: str, title: str = "") -> str:
        combined = f"{title}\n{text[:2000]}"
        scores = {}
        for ctype, pattern in CONTENT_TYPE_RULES:
            scores[ctype] = len(pattern.findall(combined))
        if not scores:
            return "未分类"
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "未分类"

    @staticmethod
    def _extract_tags(text: str, top_n: int = 5) -> List[str]:
        try:
            import jieba.analyse
            tags = jieba.analyse.extract_tags(text, topK=top_n * 2, withWeight=False)
            return [t for t in tags if t not in TAG_STOPWORDS and len(t) >= 2][:top_n]
        except ImportError:
            words = re.findall(r"[\u4e00-\u9fff\w]{2,}", text)
            freq = {}
            for w in words:
                if w not in TAG_STOPWORDS:
                    freq[w] = freq.get(w, 0) + 1
            return [t[0] for t in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def _extract_entities(self, text: str) -> dict:
        """通用实体抽取 — jieba 词性标注 + 自定义词典"""
        result = {"persons": [], "organizations": [], "locations": []}
        try:
            import jieba.posseg as pseg
            import jieba
        except ImportError:
            return result

        for w in CUSTOM_ENTITY_DICT:
            jieba.add_word(w, freq=100)

        seen = {"nr": set(), "nt": set(), "ni": set(), "ns": set()}
        skip_suffixes = {"的", "和", "是", "在", "了", "也", "都", "就", "但", "而", "与",
                         "等", "从", "对", "向", "给", "被", "把", "将", "以", "所",
                         "这", "那", "很", "更", "最", "只", "又", "再", "才",
                         "工作", "服务", "管理", "技术", "发展", "建设", "经济",
                         "项目", "领域", "产业", "市场", "企业", "金融", "文化",
                         "社会", "科学", "研究", "活动", "安全", "体系", "系统",
                         "机制", "模式", "目标", "任务", "方案", "战略",
                         "强调", "指出", "宣布", "公布", "发布", "报道", "关注",
                         "全面", "深入", "重点", "大力", "加强", "推进", "推动",
                         "省", "市", "县", "区", "镇", "乡", "村", "路", "街",
                         "局", "面", "度", "性", "化", "点", "力", "法", "率", "证",
                         "件", "片", "器", "机", "业", "体", "型",
                         "新华社", "人民日报", "高质量发展", "中国式现代化"}

        for tok in pseg.cut(text[:5000]):
            w = tok.word.strip()
            f = tok.flag
            if len(w) < 2 or len(w) > 10:
                continue
            if f in ("nr", "nt", "ni", "ns") and w not in seen[f]:
                if any(w.endswith(s) for s in skip_suffixes):
                    continue
                if f == "nr":
                    result["persons"].append(w)
                elif f in ("nt", "ni"):
                    result["organizations"].append(w)
                elif f == "ns":
                    result["locations"].append(w)
                seen[f].add(w)

        for key in result:
            result[key] = list(dict.fromkeys(result[key]))
        return result
