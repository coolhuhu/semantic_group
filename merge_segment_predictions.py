#!/usr/bin/env python3
"""Merge segment-level translation predictions into document-level COMET inputs.

The input may be a JSON array, a JSON object containing a list, or JSONL.  Each
prediction row must contain a segment name and a predicted translation.  The
segment name is matched against files produced by ``segment_data.py``; matching
is by the normalized filename, not by an unsafe substring search.

For every source recording this script writes one row containing ``src``,
``mt`` and ``ref`` fields suitable for COMET.  Four output files are always
created: ``acl_dev.json``, ``acl_eval.json``, ``realsi_en2zh.json`` and
``realsi_zh2en.json``.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GROUP_DIRS = {
    "acl_dev": Path("ACL6060/dev"),
    "acl_eval": Path("ACL6060/eval"),
    "realsi_en2zh": Path("RealSI/en2zh"),
    "realsi_zh2en": Path("RealSI/zh2en"),
}
GROUP_FILES = {
    "acl_dev": "acl_dev.json",
    "acl_eval": "acl_eval.json",
    "realsi_en2zh": "realsi_en2zh.json",
    "realsi_zh2en": "realsi_zh2en.json",
}
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

SEGMENT_RE = re.compile(r"^(?P<recording>.+)-(?P<segment_id>\d+)-(?P<start>\d+)-(?P<end>\d+)$")
NAME_KEYS = ("name", "segment_name", "audio_segment", "filename", "file", "path", "id")
PREDICTION_KEYS = (
    "translation",
    "pred_translation",
    "prediction",
    "hypothesis",
    "mt",
    "output",
    "generated_text",
    "text",
)


def canonical_group(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("/", "_").replace("-", "_")
    compact = re.sub(r"[^a-z0-9]", "", text)
    return GROUP_ALIASES.get(text) or GROUP_ALIASES.get(compact)


def read_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSON, JSONL, or an object containing a conventional row list."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}")
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
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
        raise ValueError(f"{path}: expected a JSON array/object or JSONL")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every prediction row must be a JSON object")
    return list(rows)


def normalized_name(value: Any) -> str:
    """Normalize a prediction name to the generated segment basename."""
    text = str(value).strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    for suffix in (".translation", ".wav", ".txt", ".json"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def parse_segment_name(name: str) -> Tuple[str, int, int, int]:
    match = SEGMENT_RE.fullmatch(name)
    if not match:
        raise ValueError(
            "cannot parse segment name {!r}; expected <audio>-<segment_id>-<start_ms>-<end_ms>".format(name)
        )
    return (
        match.group("recording"),
        int(match.group("segment_id")),
        int(match.group("start")),
        int(match.group("end")),
    )


def segment_meta(group: str, translation_path: Path) -> Dict[str, Any]:
    suffix = ".translation"
    basename = translation_path.name[: -len(suffix)]
    recording, segment_id, start_ms, end_ms = parse_segment_name(basename)
    source_path = translation_path.with_name(basename + ".txt")
    if not source_path.is_file():
        raise FileNotFoundError("missing source transcript for {}: {}".format(basename, source_path))
    return {
        "group": group,
        "key": basename,
        "recording": recording,
        "segment_id": segment_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source_path": source_path,
        "reference_path": translation_path,
    }


def build_segment_index(data_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for group, relative_dir in GROUP_DIRS.items():
        full_wavs = data_root / relative_dir / "full_wavs"
        for segment_dir in sorted(full_wavs.glob("*.segments")):
            for translation_path in sorted(segment_dir.glob("*.translation")):
                meta = segment_meta(group, translation_path)
                index[meta["key"]].append(meta)
    return dict(index)


def get_row_value(row: Dict[str, Any], requested: str, candidates: Sequence[str], label: str) -> Any:
    if requested != "auto":
        if requested not in row:
            raise KeyError("{} key {!r} not found in row keys {}".format(label, requested, sorted(row)))
        return row[requested]
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError("could not find {} in row; tried {}".format(label, ", ".join(candidates)))


def text_from_file(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def join_parts(parts: Sequence[str], separator: str) -> str:
    return separator.join(part.strip() for part in parts if part.strip()).strip()


def separators(group: str) -> Tuple[str, str, str]:
    """Return source, hypothesis, reference separators for the language pair."""
    if group in ("acl_dev", "acl_eval", "realsi_en2zh"):
        return " ", "", ""
    return "", " ", " "


def write_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_input_spec(spec: str) -> Tuple[Optional[str], Path]:
    """Parse PATH or GROUP=PATH, allowing separate files per dataset split."""
    if "=" in spec:
        group_text, path_text = spec.split("=", 1)
        group = canonical_group(group_text)
        if group is None:
            raise ValueError("unknown group in --input specification: {}".format(group_text))
        return group, Path(path_text)
    return None, Path(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="prediction JSON/JSONL file(s); optionally use GROUP=PATH",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="dataset root containing ACL6060/ and RealSI/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=sorted(GROUP_DIRS), help="assign every input row to this group")
    parser.add_argument("--name-key", default="auto", help="prediction row field containing segment name")
    parser.add_argument("--translation-key", default="auto", help="prediction row field containing translation")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="allow a document to have fewer predicted segments than the dataset metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    index = build_segment_index(data_root)
    if not index:
        raise FileNotFoundError(
            "no generated segment metadata found below {}; run segment_data.py first".format(data_root)
        )

    # (group, recording) -> normalized segment name -> prediction row
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    input_rows = 0
    for input_spec in args.input:
        input_group, input_path = parse_input_spec(input_spec)
        rows = read_records(input_path)
        for row_number, row in enumerate(rows, 1):
            input_rows += 1
            raw_name = get_row_value(row, args.name_key, NAME_KEYS, "segment name")
            key = normalized_name(raw_name)
            candidates = index.get(key, [])
            explicit_group = canonical_group(row.get("group")) or input_group or args.group
            if explicit_group:
                candidates = [item for item in candidates if item["group"] == explicit_group]
            if not candidates:
                raise KeyError(
                    "{} row {}: segment {!r} did not match data-root metadata".format(
                        input_path, row_number, raw_name
                    )
                )
            if len(candidates) > 1:
                groups = sorted({item["group"] for item in candidates})
                raise ValueError(
                    "{} row {}: segment {!r} is ambiguous across groups {}; use --group or a group field".format(
                        input_path, row_number, raw_name, groups
                    )
                )
            meta = candidates[0]
            prediction = get_row_value(row, args.translation_key, PREDICTION_KEYS, "translation")
            if isinstance(prediction, (dict, list)):
                raise ValueError("{} row {}: translation must be a string".format(input_path, row_number))
            prediction_text = str(prediction if prediction is not None else "").strip()
            doc_key = (meta["group"], meta["recording"])
            if key in grouped[doc_key]:
                raise ValueError("duplicate prediction for segment {!r}".format(key))
            grouped[doc_key][key] = {
                "meta": meta,
                "prediction": prediction_text,
            }

    output_rows: Dict[str, List[Dict[str, Any]]] = {group: [] for group in GROUP_DIRS}
    for (group, recording), predictions in sorted(grouped.items()):
        expected = [item for item in index.values() for item in item if item["group"] == group and item["recording"] == recording]
        expected_by_key = {item["key"]: item for item in expected}
        missing = sorted(set(expected_by_key) - set(predictions))
        if missing and not args.allow_missing:
            raise ValueError(
                "{} {}: missing {} segment predictions, first missing: {}".format(
                    group, recording, len(missing), missing[0]
                )
            )
        ordered = sorted(
            predictions.values(),
            key=lambda item: (item["meta"]["segment_id"], item["meta"]["start_ms"], item["meta"]["key"]),
        )
        source_separator, prediction_separator, reference_separator = separators(group)
        sources = [text_from_file(item["meta"]["source_path"]) for item in ordered]
        references = [text_from_file(item["meta"]["reference_path"]) for item in ordered]
        hypotheses = [item["prediction"] for item in ordered]
        output_rows[group].append(
            {
                "schema": "semantic_group.document_translation.v1",
                "group": group,
                "audio": recording,
                "src": join_parts(sources, source_separator),
                "mt": join_parts(hypotheses, prediction_separator),
                "ref": join_parts(references, reference_separator),
                "num_segments": len(ordered),
                "num_missing_segments": len(missing),
                "segments": [
                    {
                        "name": item["meta"]["key"],
                        "segment_id": item["meta"]["segment_id"],
                        "start_ms": item["meta"]["start_ms"],
                        "end_ms": item["meta"]["end_ms"],
                        "prediction": item["prediction"],
                    }
                    for item in ordered
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group, filename in GROUP_FILES.items():
        write_json(args.output_dir / filename, output_rows[group])
        print("{}: {} documents -> {}".format(group, len(output_rows[group]), args.output_dir / filename))
    print("input rows: {}".format(input_rows))


if __name__ == "__main__":
    main()
