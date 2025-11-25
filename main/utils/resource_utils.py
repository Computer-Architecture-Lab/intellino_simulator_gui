# utils/resource_utils.py
import sys
import os
from pathlib import Path

def resource_path(name: str) -> str:
    """
    PyInstaller(onefile) + 개발 환경 모두에서 리소스를 안전하게 찾는 공통 함수.
    모든 custom_x.py, existing_mode_window.py, main.py에서 동일하게 import해서 사용 가능.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()

    candidates = [
        base / name,                  # PyInstaller dest='.'
        base / "main" / name,         # PyInstaller dest='main'
        base.parent / name,           # 일부 구조 보정
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    return str(candidates[0])
