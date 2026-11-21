"""使用案例数据训练 SCREENER 代理模型。"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.surrogate_model import SurrogateModelManager  # noqa: E402
from core.io_utils import read_json  # noqa: E402
from core.stiffener_profile import STIFFENER_TYPES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="逐层代理训练及独立 FEM 记录验收")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--stiffener-type", choices=STIFFENER_TYPES, required=True)
    parser.add_argument("--cases-dir", type=Path)
    parser.add_argument("--validate-cases", type=Path)
    parser.add_argument("--max-mape", type=float)
    parser.add_argument("--max-relative-error", type=float)
    parser.add_argument("--min-per-slice", type=int)
    args = parser.parse_args()
    manager = SurrogateModelManager(models_dir=args.models_dir, case_dir=args.cases_dir, stiffener_type=args.stiffener_type)
    if args.validate_cases:
        if any(value is None for value in (args.max_mape, args.max_relative_error, args.min_per_slice)):
            parser.error("独立验收必须显式指定 max-mape、max-relative-error 和 min-per-slice")
        records = [read_json(path) for path in sorted(args.validate_cases.glob("CASE_*.json"))]
        summary = manager.validate_from_records(records, max_mape=args.max_mape,
                                                max_relative_error=args.max_relative_error,
                                                min_per_slice=args.min_per_slice)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "passed" else 1
    records = manager.load_training_records()
    if not records:
        raise RuntimeError("未找到可用于训练的成功案例。")

    summary = manager.train_from_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
