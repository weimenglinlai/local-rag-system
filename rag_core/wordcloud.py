"""
词云与词频统计模块
依赖: wordcloud, jieba, matplotlib (Pillow-backed)
"""

import os
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# 中文停用词表（常用虚词、代词、介词等）
_STOPWORDS = set([
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "等", "以", "与", "之", "而", "但", "从", "被",
    "把", "对", "将", "为", "能", "所", "如", "因", "于", "及",
    "或", "并", "其", "中", "又", "让", "向", "用", "以", "我们",
    "他们", "她们", "它们", "这个", "那个", "这些", "那些", "什么",
    "怎么", "如何", "为什么", "可以", "因为", "所以", "如果", "虽然",
    "但是", "而且", "然后", "这样", "那样", "其中", "已经", "可能",
    "通过", "进行", "使用", "包括", "具有", "需要", "应该", "可以",
    "通过", "进入", "出来", "以后", "之前", "问题", "时候", "以及",
    "关于", "有关", "就是", "不是", "只是", "还是", "还要", "因为",
    "所以", "因此", "于是", "关于", "部分", "很多", "其他", "一样",
    "一种", "这些", "那些", "一些", "目前", "当前", "本次", "本",
    "被", "把", "比", "对", "从", "向", "在", "为", "以", "与",
    "和", "跟", "同", "除", "除了", "按照", "根据", "将", "可以",
    "能够", "可能", "需要", "应该", "已经", "还", "更", "最",
    "比较", "非常", "很", "极", "十分", "较", "都", "总", "共",
    "每", "各", "某", "该", "别", "某", "另", "旁", "一切",
    "一些", "任何", "所有", "什么", "谁", "哪", "哪里", "何时",
    "多少", "几", "多", "少", "全", "大", "小", "新", "老",
    "第一", "第二", "首先", "其次", "最后", "同时", "此外",
    "以及", "此外", "另外", "此外", "啊", "吧", "吗", "呢",
    "呀", "哇", "哈", "嗯", "哦", "喂", "嘛", "呵",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "这是", "不是", "没有", "可以", "什么", "怎么",
    "一些", "很多", "这个", "那个", "我们", "他们",
])

# WordCloud 字体路径
_FONT_CANDIDATES = [
    "./data",      # Microsoft YaHei
    "./data",    # SimHei
    "./data",   # Microsoft YaHei Bold
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
]


def _find_font() -> Optional[str]:
    for fp in _FONT_CANDIDATES:
        if os.path.isfile(fp):
            return fp
    logger.warning("No Chinese font found for wordcloud")
    return None


def get_top_keywords(text: str, top_n: int = 20) -> list:
    """提取 TOP N 关键词及其出现次数，返回 [{word, count}]"""
    if not text.strip():
        return []

    import jieba

    words = list(jieba.cut(text))
    counter = Counter()
    for w in words:
        w = w.strip()
        if len(w) < 2:
            continue
        if w in _STOPWORDS:
            continue
        if w.isdigit():
            continue
        counter[w] += 1

    result = [{"word": w, "count": c} for w, c in counter.most_common(top_n)]
    return result


def generate_wordcloud(text: str, output_path: str,
                       max_words: int = 100,
                       width: int = 800, height: int = 400) -> str:
    """生成词云图，保存到 output_path，返回路径

    使用单色灰度生成（遵循无颜色规则），如果需要彩色请设置 colormap 参数。
    """
    if not text.strip():
        logger.warning("Empty text for wordcloud")
        return ""

    import jieba
    from wordcloud import WordCloud

    font_path = _find_font()

    # 分词
    words = list(jieba.cut(text))
    filtered = [w for w in words
                if len(w.strip()) >= 2
                and w.strip() not in _STOPWORDS
                and not w.strip().isdigit()]

    # 直接用 WordCloud 的 generate 处理
    text_clean = " ".join(filtered)

    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        colormap="plasma",       # 使用鲜艳彩色（用户明确要求）
        prefer_horizontal=0.7,
        random_state=42,
    )

    wc.generate(text_clean)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wc.to_file(output_path)
    logger.info("Wordcloud saved to %s", output_path)
    return output_path
