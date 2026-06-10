"""OpenAI 兼容 API 适配器 — 发送请求、接收响应、流式支持。"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional

import aiohttp

from core.schema import ModelConfig


class ModelAdapter:
    """OpenAI 兼容 API 适配器"""

    def __init__(self, config: ModelConfig, timeout: int = 120):
        self.config = config
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._build_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_chat_body(self, messages: list[dict], **overrides) -> dict:
        body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
            "top_p": overrides.get("top_p", self.config.top_p),
        }
        extra = overrides.get("extra_body", {})
        body.update(extra)
        return body

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict[str, Any]:
        """发送聊天请求，返回完整响应"""
        session = await self._get_session()
        body = self._build_chat_body(messages, **kwargs)
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"API 请求失败 [HTTP {resp.status}]: {text[:500]}"
                )
            result = await resp.json()

        return result

    async def chat_with_usage(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict[str, Any]:
        """发送聊天请求，返回含 token 用量和延迟的增强响应"""
        start = time.time()
        result = await self.chat(messages, **kwargs)
        elapsed = (time.time() - start) * 1000

        usage = result.get("usage", {})
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        return {
            "content": content,
            "finish_reason": choice.get("finish_reason", ""),
            "latency_ms": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "raw": result,
        }

    async def test_connection(self) -> dict:
        """测试模型连通性 — 发一个最小请求，自动关闭 session"""
        messages = [
            {"role": "user", "content": "1+1=?\n请只输出答案。"}
        ]
        try:
            result = await self.chat_with_usage(messages, max_tokens=10)
            await self.close()
            return {
                "success": True,
                "model": self.config.model_name,
                "response": result["content"][:200],
                "latency_ms": result["latency_ms"],
            }
        except Exception as e:
            await self.close()
            return {
                "success": False,
                "model": self.config.model_name,
                "error": str(e),
            }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
