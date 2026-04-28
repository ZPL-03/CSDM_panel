from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.literature_ingest import LiteratureIngestor
from core.paths import CONFIG_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--download-oa-pdfs", action="store_true")
    parser.add_argument("--download-existing-pdfs", action="store_true")
    parser.add_argument("--parse-pdfs", action="store_true")
    parser.add_argument("--parse-existing-pdfs", action="store_true")
    parser.add_argument("--parse-backend", choices=["pymupdf", "mineru", "nougat"], default="")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--check-parsers", action="store_true")
    parser.add_argument("--force-pdfs", action="store_true")
    parser.add_argument("--import-pdf-dir", default="")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--query-group", action="append", default=[])
    parser.add_argument("--max-results", type=int, default=0)
    args = parser.parse_args()

    ingestor = LiteratureIngestor()
    if args.max_results and args.max_results > 0:
        ingestor.max_results_per_query = int(args.max_results)

    if args.check_parsers:
        import shutil

        print(
            {
                "pymupdf": True,
                "mineru": bool(shutil.which("mineru")),
                "nougat": bool(shutil.which("nougat")),
            }
        )
        return 0

    if args.reindex:
        summary = ingestor.reindex_from_records()
        print(summary)
        return 0

    if args.import_pdf_dir:
        summary = ingestor.import_pdf_directory(
            Path(args.import_pdf_dir),
            parse_pdfs=args.parse_pdfs,
            force=args.force_pdfs,
            backend=args.parse_backend or None,
            ocr=args.ocr,
        )
        print(summary)
        return 0

    if args.download_existing_pdfs:
        summary = ingestor.download_open_access_pdfs(force=args.force_pdfs)
        if args.parse_pdfs:
            parse_summary = ingestor.parse_pdfs(force=args.force_pdfs, backend=args.parse_backend or None, ocr=args.ocr)
            index_summary = ingestor.reindex_from_records()
            summary = {**summary, **{f"parse_{key}": value for key, value in parse_summary.items()}}
            summary["chunk_count"] = index_summary.get("chunk_count", 0)
        print(summary)
        return 0

    if args.parse_existing_pdfs:
        summary = ingestor.parse_pdfs(force=args.force_pdfs, backend=args.parse_backend or None, ocr=args.ocr)
        index_summary = ingestor.reindex_from_records()
        summary["chunk_count"] = index_summary.get("chunk_count", 0)
        print(summary)
        return 0

    query_groups = ingestor.load_query_groups(CONFIG_DIR / "literature_queries.yaml")
    if args.seed and not args.query_group:
        selected_groups = list(query_groups.keys())
    elif args.query_group:
        selected_groups = args.query_group
    elif args.query:
        selected_groups = []
    else:
        selected_groups = list(query_groups.keys())

    queries = []
    queries.extend(args.query)
    for group in selected_groups:
        queries.extend(query_groups.get(group, []))

    if args.refresh and not queries:
        raise SystemExit("未找到可刷新的文献查询组，请检查 literature_queries.yaml 或 --query-group 参数。")

    summary = ingestor.ingest_queries(
        queries,
        download_pdfs=args.download_oa_pdfs,
        parse_pdfs=args.parse_pdfs,
        force_pdfs=args.force_pdfs,
        parse_backend=args.parse_backend or None,
        parse_ocr=args.ocr,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
