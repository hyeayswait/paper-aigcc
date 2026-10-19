"""DeepSeek Chat Completions 客户端（OpenAI 兼容）。

所有失败统一抛 RewriteError，消息是可以直接展示给用户的中文。
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# 这些状态码视为可恢复，按 cfg.retries 退避重试
_RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


class RewriteError(Exception):
    """改写相关错误；str(exc) 为可直接展示的中文提示。"""


class DeepSeekClient:
    """DeepSeek /chat/completions 的最小客户端。"""

    def __init__(self, cfg) -> None:
        key = (getattr(cfg, "api_key", "") or "").strip()
        if not key:
            raise RewriteError("未配置 DeepSeek API Key")
        if not all(ord(ch) < 128 for ch in key):
            raise RewriteError(
                "API Key 含有非 ASCII 字符（中文/全角），请检查是否误用了示例占位符 "
                "「sk-你的真实key」。真实 key 形如 sk-<your-key>，"
                "在 platform.deepseek.com 的 API keys 页面创建。"
            )
        self.cfg = cfg
        self.api_key = key
        base = (getattr(cfg, "base_url", "") or "").strip() or DEFAULT_BASE_URL
        self.base_url = base.rstrip("/")
        self.model = (getattr(cfg, "model", "") or "").strip() or DEFAULT_MODEL
        try:
            self.timeout = int(getattr(cfg, "timeout", 90) or 90)
        except (TypeError, ValueError):
            self.timeout = 90
        try:
            self.retries = max(0, int(getattr(cfg, "retries", 0) or 0))
        except (TypeError, ValueError):
            self.retries = 0

    # ------------------------------------------------------------------ 公开接口
    def chat(self, messages: List[Dict[str, str]], temperature: float = None,
             max_tokens: int = None) -> str:
        """发一次对话补全请求，返回助手回复文本。失败抛 RewriteError。"""
        if not messages:
            raise RewriteError("消息为空，无法调用 DeepSeek")
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": float(
                getattr(self.cfg, "temperature", 0.7) if temperature is None else temperature
            ),
            "max_tokens": int(
                getattr(self.cfg, "max_tokens", 4096) if max_tokens is None else max_tokens
            ),
        }
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

        attempts = self.retries + 1
        last_error = ""
        for attempt in range(attempts):
            if attempt > 0:
                time.sleep(min(2.0, 0.4 * (2 ** (attempt - 1))))
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.exceptions.Timeout:
                last_error = "请求 DeepSeek 超时（超过 {0} 秒），请检查网络后重试。".format(
                    self.timeout
                )
                continue
            except requests.exceptions.RequestException:
                last_error = "网络不可达，无法连接 DeepSeek（{0}），请检查网络或代理设置。".format(
                    self.base_url
                )
                continue
            except Exception as exc:  # 极少数非 requests 异常，兜底成中文
                raise RewriteError("调用 DeepSeek 时发生未知错误：{0}".format(exc))

            status = int(getattr(response, "status_code", 0) or 0)
            if 200 <= status < 300:
                return self._extract_content(response)
            if status in _RETRYABLE_STATUS:
                last_error = self._message_for_status(status, response)
                continue
            raise RewriteError(self._message_for_status(status, response))

        raise RewriteError(last_error or "调用 DeepSeek 失败，请稍后重试。")

    def check_connection(self) -> Tuple[bool, str]:
        """做一次最小请求确认 Key 与网络可用；不抛异常。"""
        try:
            self.chat(
                [{"role": "user", "content": "请回复两个字：正常"}],
                temperature=0.0,
                max_tokens=8,
            )
            return True, "连接正常，当前模型：{0}".format(self.model)
        except RewriteError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, "连接检查失败：{0}".format(exc)

    # ------------------------------------------------------------------ 内部工具
    @staticmethod
    def _extract_content(response) -> str:
        try:
            data = response.json()
        except Exception:
            raise RewriteError("无法解析 DeepSeek 返回内容，请稍后重试。")
        if not isinstance(data, dict):
            raise RewriteError("无法解析 DeepSeek 返回内容，请稍后重试。")
        choices = data.get("choices") or []
        if not choices or not isinstance(choices, list):
            raise RewriteError("DeepSeek 返回了空内容，请稍后重试。")
        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if content is None:
            content = first.get("text")
        if not isinstance(content, str) or not content.strip():
            raise RewriteError("DeepSeek 返回了空内容，请稍后重试。")
        return content

    @staticmethod
    def _error_detail(response) -> str:
        detail = ""
        try:
            data = response.json()
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
                elif error:
                    detail = str(error)
        except Exception:
            detail = ""
        if not detail:
            try:
                detail = str(getattr(response, "text", "") or "")
            except Exception:
                detail = ""
        detail = " ".join(detail.split())
        if len(detail) > 160:
            detail = detail[:160] + "…"
        return detail

    def _message_for_status(self, status: int, response) -> str:
        detail = self._error_detail(response)
        if status in (401, 403):
            base = "API Key 无效或无权限，请在设置中检查 DeepSeek API Key。"
        elif status == 402:
            base = "余额不足，请前往 DeepSeek 平台充值后重试。"
        elif status == 429:
            base = "触发限流，请稍后重试或降低改写并发数。"
        elif status == 404:
            base = "接口地址不存在，请检查 Base URL 配置（HTTP 404）。"
        elif 400 <= status < 500:
            base = "请求被 DeepSeek 拒绝（HTTP {0}），请检查模型名或输入内容。".format(status)
        elif status >= 500:
            base = "DeepSeek 服务端错误（HTTP {0}），请稍后重试。".format(status)
        else:
            base = "调用 DeepSeek 失败（HTTP {0}）。".format(status)
        if detail:
            return base + "详情：" + detail
        return base
