#!/usr/bin/env python3
"""Score document-level translation JSON files with a local COMET model."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


GROUP_ALIASES = {
    "acl_dev": "acl_dev",
    "acldev": "acl_dev",
    "acl6060dev": "acl_dev",
    "acl_eval": "acl_eval",
    "acleval": "acl_eval",
    "acl6060eval": "acl_eval",
    "realsi_en2zh": "realsi_en2zh",
    "realsien2zh": "realsi_en2zh",
    "en2zh": "realsi_en2zh",
    "realsi_zh2en": "realsi_zh2en",
    "realsizh2en": "realsi_zh2en",
    "zh2en": "realsi_zh2en",
}


def canonical_group(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("/", "_").replace("-", "_")
    compact = re.sub(r"[^a-z0-9]", "", text)
    return GROUP_ALIASES.get(text) or GROUP_ALIASES.get(compact)


def read_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{}: invalid JSON: {}".format(path, line_number, exc))
            if not isinstance(row, dict):
                raise ValueError("{}:{}: expected a JSON object".format(path, line_number))
            rows.append(row)
        return rows
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("data", "results", "predictions", "outputs", "items"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            rows = [payload]
    else:
        raise ValueError("{}: expected a JSON array/object or JSONL".format(path))
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("{}: every row must be a JSON object".format(path))
    return list(rows)


def resolve_checkpoint(model_path: Path) -> Path:
    model_path = model_path.expanduser()
    if model_path.is_file():
        return model_path
    candidates = (
        model_path / "checkpoints" / "model.ckpt",
        model_path / "model.ckpt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "COMET model path must be a checkpoint file or a directory containing checkpoints/model.ckpt: {}".format(
            model_path
        )
    )


def value(row: Dict[str, Any], keys: Sequence[str], label: str, allow_empty: bool = False) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            result = str(row[key]).strip()
            if result or allow_empty:
                return result
    if allow_empty:
        return ""
    raise ValueError("row is missing non-empty {} (tried {})".format(label, ", ".join(keys)))


def parse_input_spec(spec: str) -> Tuple[Optional[str], Path]:
    # GROUP=PATH is convenient when a path has no informative filename.
    if "=" in spec:
        group_text, path_text = spec.split("=", 1)
        group = canonical_group(group_text)
        if group is None:
            raise ValueError("unknown group in --input specification: {}".format(group_text))
        return group, Path(path_text)
    path = Path(spec)
    return canonical_group(path.stem), path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="COMET checkpoint or model directory")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="document JSON/JSONL files; optionally use GROUP=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs passed to COMET; use 0 for CPU")
    return parser.parse_args()


def score_rows(model: Any, rows: Sequence[Dict[str, Any]], batch_size: int, gpus: int) -> List[float]:
    comet_data = []
    for row_number, row in enumerate(rows, 1):
        source = value(row, ("src", "source"), "source", allow_empty=False)
        reference = value(row, ("ref", "reference"), "reference", allow_empty=False)
        hypothesis = value(row, ("mt", "hypothesis", "prediction", "translation"), "hypothesis", allow_empty=True)
        comet_data.append({"src": source, "mt": hypothesis, "ref": reference})
    if not comet_data:
        return []
    result = model.predict(comet_data, batch_size=batch_size, gpus=gpus)
    scores = getattr(result, "scores", None)
    if scores is None and isinstance(result, dict):
        scores = result.get("scores")
    if scores is None:
        raise TypeError("unsupported COMET result type: {}".format(type(result).__name__))
    values = [float(score) for score in scores]
    if len(values) != len(rows):
        raise ValueError("COMET returned {} scores for {} documents".format(len(values), len(rows)))
    return values


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.gpus < 0:
        raise ValueError("--gpus must be >= 0")

    checkpoint = resolve_checkpoint(args.model_path)
    try:
        from comet import load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError("COMET is not installed; install the unbabel-comet package in this environment") from exc

    print("loading COMET checkpoint: {}".format(checkpoint))
    model = load_from_checkpoint(str(checkpoint))
    model.eval()

    group_results: Dict[str, Dict[str, Any]] = {}
    all_scores: List[float] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.input:
        inferred_group, path = parse_input_spec(spec)
        rows = read_records(path)
        group = inferred_group or "input_{}".format(len(group_results) + 1)
        scores = score_rows(model, rows, args.batch_size, args.gpus)
        scored_rows = []
        for row, score in zip(rows, scores):
            scored_rows.append(
                {
                    "group": group,
                    "audio": row.get("audio", row.get("name")),
                    "comet": score,
                    "num_segments": row.get("num_segments"),
                }
            )
        mean_score = mean(scores) if scores else None
        group_results[group] = {
            "input": str(path),
            "num_audios": len(scores),
            "mean_comet": mean_score,
            "scores_file": str(args.output_dir / (group + "_comet.json")),
        }
        all_scores.extend(scores)
        write_json(args.output_dir / (group + "_comet.json"), scored_rows)
        print("{}: {} audios, mean COMET={}".format(group, len(scores), mean_score))

    summary = {
        "schema": "semantic_group.document_comet.v1",
        "model_path": str(args.model_path),
        "checkpoint": str(checkpoint),
        "batch_size": args.batch_size,
        "gpus": args.gpus,
        "groups": group_results,
        "all_audio_count": len(all_scores),
        "all_audio_macro_mean": mean(all_scores) if all_scores else None,
        "four_group_mean": mean(
            [item["mean_comet"] for item in group_results.values() if item["mean_comet"] is not None]
        )
        if any(item["mean_comet"] is not None for item in group_results.values())
        else None,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
