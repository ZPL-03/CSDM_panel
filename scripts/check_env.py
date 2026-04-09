"""CSDM 环境自检脚本。"""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import ensure_project_dirs


def check_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        return True, f"{name} {version}"
    except Exception as exc:
        return False, f"{name} 导入失败: {exc}"


def torch_runtime_detail() -> tuple[bool, str]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        detail = (
            f"torch {torch.__version__}, cuda={torch.version.cuda}, "
            f"available={cuda_available}, device={device_name}"
        )
        return cuda_available, detail
    except Exception as exc:
        return False, f"torch 运行态检查失败: {exc}"


def main() -> int:
    ensure_project_dirs()
    checks = []
    checks.append(("Python", True, sys.version.split()[0]))
    checks.append(("ABAQUS", shutil.which("abaqus") is not None, shutil.which("abaqus") or "未找到"))
    checks.append(("Ollama", True, "当前默认使用 Ollama 云端 API，本地 Ollama 是否拉模型不影响主流程"))

    for module_name in [
        "PyQt6",
        "jinja2",
        "yaml",
        "jsonschema",
        "openai",
        "chromadb",
        "langchain",
        "langchain_community",
        "sentence_transformers",
        "torch",
        "sklearn",
        "matplotlib",
        "pyvista",
        "reportlab",
    ]:
        ok, detail = check_module(module_name)
        checks.append((module_name, ok, detail))

    torch_ok, torch_detail = torch_runtime_detail()
    checks.append(("TorchCUDA", torch_ok, torch_detail))

    exit_code = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
