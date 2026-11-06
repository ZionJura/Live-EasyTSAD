#!/usr/bin/env python3
"""Export EasyTSAD avg.json files for the leaderboard, using only the stdlib.

Protocol settings not encoded in avg.json (e.g. EventF1PA's logarithm base)
must be verified by the experiment owner before publishing the snapshot.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


METRICS = {
    "event_f1": ("event-based f1 under pa with mode log", "f1"),
    "point_f1": ("best f1 under pa", "f1"),
    "auprc": ("point-based auprc", None),
}
SCHEMAS = {"mts", "naive", "all_in_one", "zero_shot", "zero_shot_cross_ds"}
TRADITIONAL = {"AR", "LOF", "OCSVM", "SubLOF", "SubOCSVM", "SAND", "MatrixProfile", "Spot", "Diff"}


def export_results(root: Path) -> dict:
    methods, datasets, results = {}, {}, []
    for path in sorted(root.glob("*/*/*/avg.json")):
        method, schema, dataset, _ = path.relative_to(root).parts
        # Legacy summaries may use one_by_one for Naive.
        schema = "naive" if schema == "one_by_one" else schema
        if schema not in SCHEMAS:
            raise ValueError(f"{path}: unsupported training schema {schema!r}")
        source = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError(f"{path}: expected a JSON object")
        scores = {}
        for metric, (key, field) in METRICS.items():
            value = source.get(key)
            if field and isinstance(value, dict):
                value = value.get(field)
            elif field and value is not None:
                raise ValueError(f"{path}: {key!r} must contain an {field!r} field")
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                scores[metric] = None
            elif isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"{path}: {key!r} must be a number in [0, 1] or null")
            else:
                scores[metric] = value
        if all(v is None for v in scores.values()):
            continue
        methods[method] = {
            "id": method, "name": method,
            "family": "传统方法" if method in TRADITIONAL else "预训练模型" if method == "OFA" else "深度学习",
            "description": "本地实验方法 · 分类可在导出文件中调整", "color": "#738f78",
        }
        task = "mts" if schema == "mts" else "uts"
        if dataset in datasets and datasets[dataset]["task"] != task:
            raise ValueError(f"{dataset!r} occurs in both UTS and MTS; use distinct dataset identifiers")
        datasets[dataset] = {"id": dataset, "task": task, "domain": "本地评测数据集"}
        if any(row["method"] == method and row["schema"] == schema and row["dataset"] == dataset for row in results):
            raise ValueError(f"{path}: duplicate method/schema/dataset after legacy schema normalization")
        results.append({"method": method, "dataset": dataset, "schema": schema, "metrics": scores})
    if not results:
        raise ValueError(f"No supported metric results found under {root}/<method>/<schema>/<dataset>/avg.json")
    return {
        "version": 1, "demo": False,
        "updatedAt": datetime.now(timezone.utc).date().isoformat(),
        "methods": list(methods.values()), "datasets": list(datasets.values()), "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("Results/Evals"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        benchmark = export_results(args.results)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(benchmark['results'])} results to {args.output}")
    print("Before publication, verify protocol parameters, margins, preprocessing and experiment provenance.")


if __name__ == "__main__":
    main()
