"""让 vendor/ 里的第三方包可被导入（本项目无网络 pip，包是手动装到 vendor 的）。"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR = os.path.join(_PROJECT_ROOT, "vendor")


def ensure_vendor() -> bool:
    """把 vendor 目录加入 sys.path；不存在则静默返回 False。"""
    if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
        return True
    return _VENDOR in sys.path
