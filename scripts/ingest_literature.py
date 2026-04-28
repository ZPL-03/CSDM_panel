from __future__ import annotations

import argparse

from core.literature_ingest import LiteratureIngestor
from core.paths import CONFIG_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--query-group", action="append", default=[])
    parser.add_argument("--max-results", type=int, default=0)
    args = parser.parse_args()

    ingestor = LiteratureIngestor()
    if args.max_results and args.max_results > 0:
        ingestor.max_results_per_query = int(args.max_results)

    if args.reindex:
        summary = ingestor.reindex_from_records()
        print(summary)
        return 0

    query_groups = ingestor.load_query_groups(CONFIG_DIR / "literature_queries.yaml")
    if args.seed and not args.query_group:
        selected_groups = list(query_groups.keys())
    elif args.query_group:
        selected_groups = args.query_group
    else:
        selected_groups = list(query_groups.keys())

    queries = []
    for group in selected_groups:
        queries.extend(query_groups.get(group, []))

    if args.refresh and not queries:
        raise SystemExit("未找到可刷新的文献查询组，请检查 literature_queries.yaml 或 --query-group 参数。")

    summary = ingestor.ingest_queries(queries)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
