"""项目支持的铺层表达式、角度序列与比例。"""

from __future__ import annotations

import re
from typing import Any


def normalize_layup(value: Any) -> dict:
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
        symmetric = bool(parts and parts[-1].lower() == "s")
        value = "[" + "/".join(parts[:-1] if symmetric else parts) + "]" + ("s" if symmetric else "")
    text = re.sub(r"^skin_layup\s*[:：]\s*", "", str(value or "").strip(), flags=re.IGNORECASE)
    match = re.fullmatch(r"\[([^\[\]]+)\]([sS]?)", text)
    if not match:
        raise ValueError("铺层必须使用 [角度/角度/...]s 格式")
    base = []
    for token in match[1].split("/"):
        token = token.strip().replace("deg", "").replace("°", "")
        if token.startswith("±"):
            angle = float(token[1:])
            values = [angle, -angle]
        else:
            values = [float(token)]
        if any(angle not in (0.0, 45.0, -45.0, 90.0) for angle in values):
            raise ValueError("铺层角度只允许 0、45、-45、90 度")
        base.extend(values)
    if base.count(45.0) != base.count(-45.0):
        raise ValueError("正负 45 度铺层数量不相等，不满足平衡铺层约束")
    symmetric = bool(match[2])
    sequence = base + list(reversed(base)) if symmetric else base
    total = len(sequence)
    return {
        "skin_layup": "[" + "/".join(f"{angle:g}" for angle in base) + "]" + ("s" if symmetric else ""),
        "skin_f0": round(sequence.count(0.0) / total, 3),
        "skin_f45": round((sequence.count(45.0) + sequence.count(-45.0)) / total, 3),
        "skin_f90": round(sequence.count(90.0) / total, 3),
        "ply_count": total,
    }
