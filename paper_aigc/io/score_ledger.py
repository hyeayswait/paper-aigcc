"""T7 多系统分数台账（SPEC2 第 4 节）。

把朱雀 / PaperPass / 知网 / 本工具等不同系统的检测分数集中记录到一个 JSON
文件里，供报告展示各版本、各系统的最新分数。纯标准库实现，兼容 Python 3.9。

- ``ledger_path(base_dir="")``：默认 ``<项目根>/data/score_ledger.json``
- ``add_record(record, path="")``：追加一条记录（自动建父目录、自动补时间），
  返回实际写入的路径
- ``load_records(path="")``：读回记录；文件不存在或 JSON 损坏一律返回 ``[]``，
  绝不抛异常
- ``to_meta(records)``：生成报告用的 ``{"rows": [...], "latest": {...}}``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

TIME_FORMAT = "%Y-%m-%d %H:%M"
LEDGER_RELATIVE = os.path.join("data", "score_ledger.json")

# 本文件位于 <项目根>/paper_aigc/io/score_ledger.py，因此上溯三层拿到项目根。
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@dataclass
class LedgerRecord:
    """一条多系统检测分数记录。

    ``value`` 是百分比数值（例如 34.47 表示 34.47%）。
    ``time`` 允许留空，由 :func:`add_record` 补当前时间。
    """

    time: str = ""
    version: str = ""
    system: str = ""
    metric: str = ""
    value: float = 0.0
    note: str = ""


def _project_root() -> str:
    return _PROJECT_ROOT


def ledger_path(base_dir: str = "") -> str:
    """返回台账文件路径。

    ``base_dir`` 为空时使用项目根（``<项目根>/data/score_ledger.json``）；
    传入 ``base_dir`` 时把它当作项目根看待，返回
    ``<base_dir>/data/score_ledger.json``。返回值始终是绝对路径。
    """
    root = (base_dir or "").strip() or _project_root()
    return os.path.abspath(os.path.join(root, LEDGER_RELATIVE))


def now_str() -> str:
    """当前本地时间，格式 ``YYYY-MM-DD HH:MM``。"""
    return datetime.now().strftime(TIME_FORMAT)


def _as_float(value: Any) -> float:
    """尽量把值转成 float，失败时退回 0.0（台账读取不容错失败）。"""
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip() if isinstance(value, str) else str(value)


def _resolve(path: str) -> str:
    raw = (path or "").strip()
    return os.path.abspath(raw) if raw else ledger_path()


def _record_to_dict(record: LedgerRecord) -> Dict[str, Any]:
    return {
        "time": _as_str(record.time),
        "version": _as_str(record.version),
        "system": _as_str(record.system),
        "metric": _as_str(record.metric),
        "value": _as_float(record.value),
        "note": _as_str(record.note),
    }


def _record_from_dict(item: Any) -> Any:
    """把一个 JSON 元素转成 LedgerRecord；无法识别时返回 None。"""
    if isinstance(item, LedgerRecord):
        return item
    if not isinstance(item, dict):
        return None
    return LedgerRecord(
        time=_as_str(item.get("time")),
        version=_as_str(item.get("version")),
        system=_as_str(item.get("system")),
        metric=_as_str(item.get("metric")),
        value=_as_float(item.get("value")),
        note=_as_str(item.get("note")),
    )


def load_records(path: str = "") -> List[LedgerRecord]:
    """读取全部台账记录。

    文件不存在、内容不是合法 JSON、顶层不是列表等情况统一返回 ``[]``，
    任何情况下都不抛异常。
    """
    target = _resolve(path)
    try:
        if not os.path.isfile(target):
            return []
        with open(target, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError, TypeError):
        return []

    if isinstance(raw, dict):
        raw = raw.get("records") or raw.get("rows") or []
    if not isinstance(raw, list):
        return []

    records: List[LedgerRecord] = []
    for item in raw:
        record = _record_from_dict(item)
        if record is not None:
            records.append(record)
    return records


def add_record(record: LedgerRecord, path: str = "") -> str:
    """追加一条记录并返回实际写入的文件路径。

    自动创建父目录；``record.time`` 为空时补当前时间；``value`` 统一转 float。
    已有的记录会先读回来再整体写回（损坏的旧文件按空台账处理）。
    """
    target = _resolve(path)

    if not _as_str(record.time):
        record.time = now_str()
    else:
        record.time = _as_str(record.time)
    record.version = _as_str(record.version)
    record.system = _as_str(record.system)
    record.metric = _as_str(record.metric)
    record.note = _as_str(record.note)
    record.value = _as_float(record.value)

    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)

    records = load_records(target)
    records.append(record)
    payload = [_record_to_dict(item) for item in records]

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return target


def _time_key(value: str) -> datetime:
    """把时间字符串转成可比较的 datetime；无法解析时视为最早。"""
    text = _as_str(value)
    for fmt in (TIME_FORMAT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.min


def to_meta(records: List[LedgerRecord]) -> dict:
    """生成报告用的元数据。

    返回结构::

        {
          "rows": [{"time","version","system","metric","value","note"}, ...],
          "latest": {system: {metric: {"value","version","time"}}}
        }

    ``rows`` 保持传入顺序；``latest`` 对每个 system + metric 取时间最新的一条，
    时间相同或无法解析时后出现的记录优先。
    """
    rows: List[Dict[str, Any]] = []
    latest: Dict[str, Dict[str, Dict[str, Any]]] = {}
    latest_key: Dict[Tuple[str, str], datetime] = {}

    for record in records or []:
        rows.append(_record_to_dict(record))
        system = _as_str(record.system)
        metric = _as_str(record.metric)
        marker = (system, metric)
        key = _time_key(record.time)
        current = latest_key.get(marker)
        if current is None or key >= current:
            latest_key[marker] = key
            latest.setdefault(system, {})[metric] = {
                "value": _as_float(record.value),
                "version": _as_str(record.version),
                "time": _as_str(record.time),
            }

    return {"rows": rows, "latest": latest}
