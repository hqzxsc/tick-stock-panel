"""MINIQMT (xtquant) 依赖与可用性检测。"""
from __future__ import annotations


def availability() -> tuple[bool, str]:
    """返回 (是否可用, 原因)。不抛异常。

    availability 仅用于检测插件依赖 (xtquant) 是否已安装。
    只要成功导入 xtquant 即视为已安装可用 (True, "ok")。
    """
    try:
        import xtquant.xtdata  # noqa: F401  # 导入成功即视为依赖就绪
        return True, "ok"
    except ImportError as e:
        return False, f"未安装 xtquant 库 ({e}), 请运行: pip install xtquant"
    except Exception as e:
        return False, f"依赖检测异常: {e}"


