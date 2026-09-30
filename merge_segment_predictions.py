#!/usr/bin/env python3
"""Merge segment-level inference JSON into document-level COMET inputs.

The inference rows are expected to contain ``wav_id``, ``src``, ``ref`` and
``mt``.  ``mt`` may be a JSON string such as
``{"transcription": "source<SepPanGuPi>translation"}``; the text after the
separator is used as the model translation.

The script does not need the dataset directory.  It groups rows by the source
recording parsed from ``wav_id`` and orders them by segment id/start time.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


GROUPS = ("acl_dev", "acl_eval", "realsi_en2zh", "realsi_zh2en")
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
SEPARATOR = "<SepPanGuPi>"


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


def row_value(row: Dict[str, Any], requested: str, candidates: Sequence[str], label: str) -> Any:
    if requested != "auto":
        if requested not in row:
            raise KeyError("{} key {!r} not found; available keys: {}".format(label, requested, sorted(row)))
        return row[requested]
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError("could not find {}; tried {}".format(label, ", ".join(candidates)))


def normalize_wav_id(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    for suffix in (".wav", ".json", ".txt"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def parse_segment_name(name: str) -> Tuple[str, int, int, int]:
    match = SEGMENT_RE.fullmatch(name)
    if not match:
        raise ValueError(
            "cannot parse wav_id {!r}; expected <audio>-<segment_id>-<start_ms>-<end_ms>".format(name)
        )
    return (
        match.group("recording"),
        int(match.group("segment_id")),
        int(match.group("start")),
        int(match.group("end")),
    )


def extract_translation(value: Any) -> str:
    """Extract the target translation from a plain or serialized MT result."""
    current = value
    if isinstance(current, str):
        text = current.strip()
        # Most inference rows store mt as an escaped JSON object.
        if text.startswith("{") or text.startswith("["):
            try:
                current = json.loads(text)
            except json.JSONDecodeError:
                current = text
        else:
            current = text

    if isinstance(current, dict):
        for key in ("transcription", "translation", "prediction", "hypothesis", "text", "mt"):
            if key in current:
                return extract_translation(current[key])
        raise ValueError("mt JSON object has no transcription/translation field")
    if isinstance(current, list):
        raise ValueError("mt must contain one translation, not a list")

    text = str(current if current is not None else "").strip()
    if SEPARATOR in text:
        text = text.rsplit(SEPARATOR, 1)[1].strip()
    return text


def infer_group(recording: str) -> Optional[str]:
    lower = recording.lower()
    if lower.startswith("en2zh-") or lower.startswith("en2zh_"):
        return "realsi_en2zh"
    if lower.startswith("zh2en-") or lower.startswith("zh2en_"):
        return "realsi_zh2en"
    # ACL dev/eval have the same filename convention and cannot be inferred.
    return None


def parse_input_spec(spec: str) -> Tuple[Optional[str], Path]:
    """Parse PATH or GROUP=PATH."""
    if "=" in spec:
        group_text, path_text = spec.split("=", 1)
        group = canonical_group(group_text)
        if group is None:
            raise ValueError("unknown group in --input specification: {}".format(group_text))
        return group, Path(path_text)
    return None, Path(spec)


def separators(group: str) -> Tuple[str, str, str]:
    """Return source, hypothesis, reference separators."""
    if group in ("acl_dev", "acl_eval", "realsi_en2zh"):
        return " ", "", ""
    return "", " ", " "


def join_parts(parts: Sequence[str], separator: str) -> str:
    return separator.join(str(part).strip() for part in parts if str(part).strip()).strip()


def write_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="inference JSON/JSONL file(s); optionally use GROUP=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, help="assign every row to this group")
    parser.add_argument("--name-key", default="wav_id", help="field containing segment wav_id")
    parser.add_argument("--source-key", default="src", help="field containing source text")
    parser.add_argument("--reference-key", default="ref", help="field containing reference translation")
    parser.add_argument("--translation-key", default="mt", help="field containing model translation/result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # (group, recording) -> segment basename -> segment data
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    input_rows = 0

    for input_spec in args.input:
        input_group, input_path = parse_input_spec(input_spec)
        rows = read_records(input_path)
        for row_number, row in enumerate(rows, 1):
            input_rows += 1
            raw_wav_id = row_value(row, args.name_key, ("wav_id", "name", "segment_name"), "wav_id")
            segment_name = normalize_wav_id(raw_wav_id)
            recording, segment_id, start_ms, end_ms = parse_segment_name(segment_name)

            row_group = canonical_group(row.get("group"))
            group = input_group or row_group or args.group or infer_group(recording)
            if group is None:
                raise ValueError(
                    "{} row {}: cannot distinguish ACL dev/eval from wav_id {!r}; "
                    "use --group acl_dev/acl_eval or GROUP=PATH".format(input_path, row_number, raw_wav_id)
                )

            source = row_value(row, args.source_key, ("src", "source"), "source")
            reference = row_value(row, args.reference_key, ("ref", "reference"), "reference")
            prediction_value = row_value(row, args.translation_key, ("mt", "translation", "prediction"), "mt")
            if isinstance(source, (dict, list)) or isinstance(reference, (dict, list)):
                raise ValueError("{} row {}: src/ref must be strings".format(input_path, row_number))

            key = (group, recording)
            if segment_name in grouped[key]:
                raise ValueError("duplicate prediction for {} row {}: {}".format(input_path, row_number, segment_name))
            grouped[key][segment_name] = {
                "segment_name": segment_name,
                "segment_id": segment_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "src": str(source if source is not None else "").strip(),
                "ref": str(reference if reference is not None else "").strip(),
                "mt": extract_translation(prediction_value),
            }

    output_rows: Dict[str, List[Dict[str, Any]]] = {group: [] for group in GROUPS}
    for (group, recording), segment_map in sorted(grouped.items()):
        ordered = sorted(
            segment_map.values(),
            key=lambda item: (item["segment_id"], item["start_ms"], item["segment_name"]),
        )
        source_separator, prediction_separator, reference_separator = separators(group)
        source = join_parts([item["src"] for item in ordered], source_separator)
        prediction = join_parts([item["mt"] for item in ordered], prediction_separator)
        reference = join_parts([item["ref"] for item in ordered], reference_separator)
        output_rows[group].append(
            {
                "schema": "semantic_group.document_translation.v2",
                "group": group,
                "audio": recording,
                "src": source,
                "mt": prediction,
                "ref": reference,
                "num_segments": len(ordered),
                "segments": [
                    {
                        "wav_id": item["segment_name"],
                        "segment_id": item["segment_id"],
                        "start_ms": item["start_ms"],
                        "end_ms": item["end_ms"],
                        "src": item["src"],
                        "mt": item["mt"],
                        "ref": item["ref"],
                    }
                    for item in ordered
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group, filename in GROUP_FILES.items():
        output_path = args.output_dir / filename
        write_json(output_path, output_rows[group])
        print("{}: {} documents -> {}".format(group, len(output_rows[group]), output_path))
    print("input rows: {}".format(input_rows))


if __name__ == "__main__":
    main()
