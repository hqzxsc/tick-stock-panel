"""TDX (通达信 TdxQuant) 依赖与可用性检测。"""
from __future__ import annotations

import locale
import os
import subprocess
import sys


def get_process_dir_by_name(exe_name: str) -> list[str]:
    """返回进程可执行文件所在目录列表。

    用 PowerShell CIM 查询进程路径: wmic 在 Win11 24H2 起默认不再提供,
    继续依赖它会出现"装了通达信却检测不到"的问题。
    """
    path_list: list[str] = []
    script = (
        f"Get-CimInstance Win32_Process -Filter \"name='{exe_name}'\" "
        "| Select-Object -ExpandProperty ExecutablePath"
    )
    p = None
    try:
        p = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        stdout_bytes, _ = p.communicate(timeout=10)
    except (OSError, subprocess.SubprocessError):
        if p is not None:
            p.kill()
        return path_list

    for line in stdout_bytes.decode(locale.getpreferredencoding(), errors="replace").splitlines():
        line = line.strip()
        if not line or line == "ExecutablePath":
            continue
        path = os.path.dirname(line)
        if path and path not in path_list:
            path_list.append(path)
    return path_list


def availability() -> tuple[bool, str]:
    """返回 (是否可用, 原因)。不抛异常。"""
    import platform

    if platform.system() != "Windows":
        return False, "仅支持 Windows 系统"

    tdx_root = get_process_dir_by_name("tdxw.exe")
    if not tdx_root:
        return False, "未检测到通达信安装,请先安装通达信金融终端"
    tdx_root = tdx_root[0]
    
    import os

    tqcenter_path = os.path.join(tdx_root, "PYPlugins", "user", "tqcenter.py")
    if not os.path.isfile(tqcenter_path):
        return False, f"未找到 tqcenter.py ({tqcenter_path}),请升级通达信客户端至支持 TQ 策略的版本"

    sys.path.insert(0, os.path.join(tdx_root, "PYPlugins", "user"))
    try:
        from tqcenter import tq  # noqa: F401

        return True, "ok"
    except ImportError as e:
        return False, f"tqcenter 导入失败 ({e}),请确认通达信客户端已运行并登录"
    except Exception as e:
        return False, f"依赖检测异常: {e}"
