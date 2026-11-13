"""
LLM API 客户端

支持所有 OpenAI 兼容 API（DeepSeek / OpenAI / vLLM / Ollama / Qwen 等）
"""

import json
import logging
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import List, Dict, Optional, Generator

logger = logging.getLogger(__name__)


class LLMClient:
    """通用 LLM API 客户端（OpenAI 兼容协议）"""

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self.api_base = llm_cfg.get("api_base", "https://api.openai.com/v1").rstrip("/")
        self.api_key = llm_cfg.get("api_key", "")
        self.model = llm_cfg.get("model", "gpt-4o-mini")
        self.max_tokens = llm_cfg.get("max_tokens", 2048)
        self.temperature = llm_cfg.get("temperature", 0.7)
        self.timeout = llm_cfg.get("timeout", 60)
        self.enabled = llm_cfg.get("enabled", True) and bool(self.api_key)

    def reload(self, config: dict):
        """热重载 LLM 配置，无需重启服务"""
        llm_cfg = config.get("llm", {})
        old_enabled = self.enabled
        self.api_base = llm_cfg.get("api_base", self.api_base).rstrip("/")
        self.api_key = llm_cfg.get("api_key", self.api_key)
        self.model = llm_cfg.get("model", self.model)
        self.max_tokens = llm_cfg.get("max_tokens", self.max_tokens)
        self.temperature = llm_cfg.get("temperature", self.temperature)
        self.timeout = llm_cfg.get("timeout", self.timeout)
        self.enabled = llm_cfg.get("enabled", True) and bool(self.api_key)
        return self.enabled != old_enabled

    def _request(self, endpoint: str, payload: dict, stream: bool = False) -> dict:
        """发送 API 请求 (OpenAI 兼容协议, 默认 /v1/{endpoint})"""
        url = f"{self.api_base}/v1/{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API 错误 {e.code}: {body[:200]}")
        except URLError as e:
            raise RuntimeError(f"连接失败: {e.reason}")

    def _stream_request(self, endpoint: str, payload: dict):
        """流式请求（生成器）。注意：会复制 payload 避免修改调用方，并添加 stream=True。"""
        url = f"{self.api_base}/v1/{endpoint}"
        request_payload = {**payload, "stream": True}
        data = json.dumps(request_payload).encode("utf-8")

        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                buffer = b""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if line.startswith(b"data: "):
                            data_str = line[6:].decode("utf-8", errors="ignore")
                            if data_str == "[DONE]":
                                return
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
        except (HTTPError, URLError) as e:
            raise RuntimeError(f"流式请求失败: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """发送聊天请求，返回回复文本"""
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        result = self._request("chat/completions", payload)
        return result["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        on_token: callable = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """流式聊天，每收到 token 回调 on_token"""
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        full_text = []
        for chunk in self._stream_request("chat/completions", payload):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_text.append(content)
                if on_token:
                    on_token(content)

        return "".join(full_text)

    def rag_chat(
        self,
        query: str,
        context_docs: List[dict],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None,
    ) -> str:
        """
        \u5e26 RAG \u4e0a\u4e0b\u6587\u7684\u804a\u5929
        """
        if not context_docs:
            return self.chat(
                [{"role": "user", "content": query}],
                system_prompt=system_prompt,
            )

        # \u6784\u9020\u4e0a\u4e0b\u6587
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[\u53c2\u8003\u6587\u6863 {i}] (\u76f8\u4f3c\u5ea6: {doc['score']:.2f})\n{doc['content']}")

        context_block = "\n\n---\n\n".join(context_parts)

        if system_prompt is None:
            system_prompt = (
                "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u77e5\u8bc6\u5e93\u52a9\u624b\u3002\u6839\u636e\u4e0b\u9762\u63d0\u4f9b\u7684\u53c2\u8003\u6587\u6863\u56de\u7b54\u7528\u6237\u7684\u95ee\u9898\u3002"
                "\u56de\u7b54\u89c4\u5219\uff1a\n"
                "1. \u4f18\u5148\u4f7f\u7528\u53c2\u8003\u6587\u6863\u4e2d\u7684\u4fe1\u606f\u56de\u7b54\n"
                "2. \u5982\u679c\u53c2\u8003\u6587\u6863\u4e0d\u8db3\u4ee5\u56de\u7b54\u95ee\u9898\uff0c\u8bda\u5b9e\u544a\u77e5\u7528\u6237\n"
                "3. \u5f15\u7528\u53c2\u8003\u6587\u6863\u65f6\u6ce8\u660e\u6765\u6e90\n"
                "4. \u56de\u7b54\u7b80\u6d01\u3001\u51c6\u786e\uff0c\u4f7f\u7528\u4e2d\u6587"
            )

        user_message = f"\u53c2\u8003\u6587\u6863\uff1a\n\n{context_block}\n\n---\n\n\u7528\u6237\u95ee\u9898\uff1a{query}"

        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        return self.chat(messages, system_prompt=system_prompt)

    def rag_chat_stream(
        self,
        query: str,
        context_docs: List[dict],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        on_token: callable = None,
    ) -> str:
        """带 RAG 上下文的流式聊天"""
        if not context_docs:
            return self.chat_stream(
                [{"role": "user", "content": query}],
                system_prompt=system_prompt,
                on_token=on_token,
            )

        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[参考文档 {i}] (相似度: {doc['score']:.2f})\n{doc['content']}")

        context_block = "\n\n---\n\n".join(context_parts)

        if system_prompt is None:
            system_prompt = (
                "你是一个智能知识库助手。根据下面提供的参考文档回答用户的问题。"
                "回答规则：\n"
                "1. 优先使用参考文档中的信息回答\n"
                "2. 如果参考文档不足以回答问题，诚实告知用户\n"
                "3. 引用参考文档时注明来源\n"
                "4. 回答简洁、准确，使用中文"
            )

        user_message = f"参考文档：\n\n{context_block}\n\n---\n\n用户问题：{query}"

        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        return self.chat_stream(messages, system_prompt=system_prompt, on_token=on_token)
