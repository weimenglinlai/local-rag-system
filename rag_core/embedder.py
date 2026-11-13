"""
Embedding 向量化模块

使用 BGE-M3 模型（BAAI/bge-m3）：
- 1024 维稠密向量
- 支持 100+ 语言
- RTX 4060 8GB 显存友好（~2.5GB）
- 稠密 + 稀疏混合检索能力

支持 CUDA 加速，无 CUDA 时自动回退 CPU
"""

import logging
import threading
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """文本向量化器（线程安全延迟加载）"""

    def __init__(self, model_name: str = "BAAI/bge-m3",
                 device: str = "cuda", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension = None
        self._lock = threading.Lock()

    def _load_model(self):
        """延迟加载模型（线程安全）"""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            # 进度提示：导入模型库是耗时最长的一步（约 15s，纯 CPU/磁盘开销，
            # 无法避免），给出明确提示，避免用户误以为程序卡死而中断（Ctrl+C）。
            print("  [1/3] 导入模型库 sentence_transformers（首次约 15 秒，请耐心等待）...", flush=True)
            from sentence_transformers import SentenceTransformer
            import torch
            import time as _time

            # CUDA 可用性检测（带重试）。
            # GPU 驱动偶发的短暂不可用（如 CUDA 上下文 / 显存未及时释放、WDDM 驱动恢复期）
            # 会导致 is_available() 返回 False、甚至 device_count()==0；稍等重试通常即可恢复，
            # 避免误回退到 CPU 导致推理性能大幅下降。
            cuda_ok = torch.cuda.is_available()
            if self.device == "cuda" and not cuda_ok:
                for _ in range(3):
                    _time.sleep(1.5)
                    if torch.cuda.is_available():
                        cuda_ok = True
                        logger.info("CUDA 重试后恢复可用")
                        break
            if self.device == "cuda" and not cuda_ok:
                logger.warning("CUDA 不可用（已重试 3 次），回退到 CPU")
                self.device = "cpu"

            print(f"  [2/3] 加载 BGE-M3 权重到显存（device={self.device}）...", flush=True)
            logger.info("加载模型 %s (device=%s)...", self.model_name, self.device)
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            try:
                self._dimension = self._model.get_embedding_dimension()
            except AttributeError:
                self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("模型加载完成，维度: %d", self._dimension)

            # 预热：首次推理会初始化 CUDA kernel / 分配临时显存，提前做掉，
            # 避免第一次真实查询时用户感知到明显卡顿。
            print("  [3/3] 模型预热中...", flush=True)
            try:
                self._model.encode(
                    ["预热"],
                    batch_size=1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                logger.info("模型预热完成")
            except Exception as e:
                logger.warning("模型预热失败（不影响使用）: %s", e)
            print("  模型就绪。", flush=True)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def encode(self, texts: List[str],
               normalize: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """
        将文本列表编码为向量

        Args:
            texts: 文本列表
            normalize: 是否 L2 归一化（用于余弦相似度）
            show_progress: 是否显示进度条

        Returns:
            shape=(len(texts), dimension) 的 numpy 数组
        """
        self._load_model()

        # BGE 模型推荐给查询添加前缀以提升效果
        # 对于文档段落不添加前缀
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode([prefixed], normalize=True)[0]

    def encode_documents(self, texts: List[str],
                         show_progress: bool = False) -> np.ndarray:
        """编码文档段落"""
        return self.encode(texts, normalize=True, show_progress=show_progress)
