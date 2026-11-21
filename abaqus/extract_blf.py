# -*- coding: utf-8 -*-
"""宿主 Python 的 ODB 提取入口，数值计算复用 Abaqus 运行时实现。"""

from __future__ import annotations

import argparse
from pathlib import Path

from abaqus import runtime_extract_blf as runtime
from core.io_utils import read_json, write_json
from core.task_contract import describe_boundary_conditions, describe_load_conditions


def extract_blf(odb_path: Path, result_json: Path, input_json: Path | None = None) -> None:
    runtime.extract_blf(str(odb_path), str(result_json), str(input_json) if input_json else None)
    result = read_json(result_json)
    candidate = runtime.load_candidate(str(input_json) if input_json else None,
                                       result["candidate_id"], str(result_json))
    result["load_summary"] = describe_load_conditions(candidate.get("load_conditions", {}))
    result["boundary_summary"] = describe_boundary_conditions(candidate.get("boundary_conditions", {}))
    result["diagnosis_summary"] = (
        "首个正屈曲特征值提取完成；整体或局部模态尚未分类。"
        if result["status"] == "success" else "结果提取失败，需检查 ODB 和作业日志。"
    )
    write_json(result_json, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odb", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--input")
    args = parser.parse_args()
    extract_blf(Path(args.odb), Path(args.result), Path(args.input) if args.input else None)


if __name__ == "__main__":
    main()
