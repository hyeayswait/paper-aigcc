"""配置与 DeepSeek API Key 存储。

API Key 优先级：环境变量 DEEPSEEK_API_KEY > 本地配置文件。
配置目录按以下顺序取第一个可写的：
  1. 环境变量 PAPERAIGC_CONFIG_DIR
  2. %APPDATA%\\PaperAIGC
  3. <项目目录>\\.config
  4. 系统临时目录\\PaperAIGC
（某些受限运行环境不允许写 AppData，会自动降级，不会抛异常。）
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterator, List

APP_NAME = "PaperAIGC"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_dir_cache: List[str] = []


def _candidates() -> Iterator[str]:
    env_dir = os.environ.get("PAPERAIGC_CONFIG_DIR", "").strip()
    if env_dir:
        yield env_dir
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        yield os.path.join(appdata, APP_NAME)
    yield os.path.join(_PROJECT_ROOT, ".config")
    yield os.path.join(tempfile.gettempdir(), APP_NAME)


def _writable(directory: str) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, ".write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def config_dir() -> str:
    """返回第一个可写的配置目录；全部失败时退回临时目录。"""
    if _config_dir_cache:
        return _config_dir_cache[0]
    for cand in _candidates():
        if _writable(cand):
            _config_dir_cache.append(cand)
            return cand
    fallback = tempfile.gettempdir()
    _config_dir_cache.append(fallback)
    return fallback


def config_path() -> str:
    return os.path.join(config_dir(), "config.json")


@dataclass
class AppConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 1.15
    max_tokens: int = 4096
    timeout: int = 90
    retries: int = 2

    # 评分与改写阈值（0-100 启发式指数，非知网刻度）
    high_threshold: float = 45.0      # >= 该值视为待改写段落（45 = 「偏高」起点）
    rewrite_target: float = 35.0      # 改写后希望落到该值以下
    max_rewrite_attempts: int = 3
    min_rewrite_chars: int = 40       # 过短的段落不送改
    concurrency: int = 3

    # 评分引擎：calibrated = 公开数据校准（默认，AUC 0.826）；heuristic = 手工锚点
    score_engine: str = "calibrated"

    # 目标值：把整篇指数压到该百分比以下（50 = 压到 50% 以下；30 为常用下限）
    target_rate: float = 50.0

    # 查重 provider：local = 本地库；web:<后端名> = 联网搜索（需先注册搜索后端）
    similarity_provider: str = "local"
    similarity_ngram: int = 13

    # 本地库查重：比对库目录（放 .txt/.md/.docx 参考文档）
    corpus_dir: str = ""

    # 全文优化（optimize 模块）
    optimize_target: float = 18.0     # 整篇目标指数，低于该值即停止
    optimize_rounds: int = 2          # 全局迭代轮数
    optimize_max_calls: int = 300     # API 调用总预算，防止失控

    export_dir: str = ""


def load_config(path: str = "") -> AppConfig:
    """读取配置；任何异常都降级为默认配置，绝不抛出。"""
    cfg = AppConfig()
    try:
        p = path or config_path()
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            known = {f.name for f in fields(AppConfig)}
            for key, value in raw.items():
                if key in known:
                    setattr(cfg, key, value)
    except Exception:
        pass
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        cfg.api_key = env_key
    return cfg


def save_config(cfg: AppConfig, path: str = "") -> str:
    p = path or config_path()
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data: Dict[str, Any] = asdict(cfg)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def mask_key(key: str) -> str:
    if not key:
        return "(未设置)"
    if len(key) <= 8:
        return key[:2] + "***"
    return key[:4] + "****" + key[-4:]
