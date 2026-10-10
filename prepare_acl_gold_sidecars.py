#!/usr/bin/env python3
"""Generate .txt and .translation sidecars for official ACL6060 gold WAVs.

Official ACL6060 segments are named ``sent_<global_segment_id>.wav``.  The
English and translated XML files use the same global ``<seg id>`` values, so
they can be joined without reconstructing timestamps.

By default, sidecars and ``acl_gold_manifest.json`` are written beside the
WAVs.  Existing files are never changed unless ``--overwrite`` is supplied.
Only Python's standard library is required.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WAV_RE = re.compile(r"^sent_(\d+)\.wav$")
BARE_AMPERSAND_RE = re.compile(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]+;)")


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def parse_xml(path: Path) -> Dict[int, Dict[str, Any]]:
    """Return segment metadata keyed by the XML's global numeric segment id."""
    raw = path.read_text(encoding="utf-8-sig")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        # Some official ACL XML abstracts contain bare '&' (for example V&L).
        # Escaping only invalid ampersands keeps valid entities unchanged.
        root = ET.fromstring(BARE_AMPERSAND_RE.sub("&amp;", raw))

    segments: Dict[int, Dict[str, Any]] = {}
    for doc in root.iter("doc"):
        docid = str(doc.get("docid") or "").strip()
        if not docid:
            raise ValueError("{}: <doc> without docid".format(path))
        local_index = 0
        for seg in doc.findall("./seg"):
            local_index += 1
            raw_id = seg.get("id")
            if raw_id is None or not raw_id.isdigit():
                raise ValueError("{}: invalid segment id {!r}".format(path, raw_id))
            segment_id = int(raw_id)
            text = normalize_text("".join(seg.itertext()))
            if not text:
                raise ValueError("{}: empty text for segment {}".format(path, segment_id))
            if segment_id in segments:
                raise ValueError("{}: duplicate segment id {}".format(path, segment_id))
            segments[segment_id] = {
                "docid": docid,
                "document_segment_index": local_index,
                "text": text,
            }
    if not segments:
        raise ValueError("{}: no <seg> elements found".format(path))
    return segments


def collect_wavs(wav_dir: Path) -> Dict[int, Path]:
    wavs: Dict[int, Path] = {}
    for path in sorted(wav_dir.glob("sent_*.wav")):
        match = WAV_RE.fullmatch(path.name)
        if not match:
            continue
        segment_id = int(match.group(1))
        if segment_id in wavs:
            raise ValueError("duplicate WAV segment id {} in {}".format(segment_id, wav_dir))
        wavs[segment_id] = path
    if not wavs:
        raise FileNotFoundError("no sent_<id>.wav files found in {}".format(wav_dir))
    return wavs


def validate_alignment(
    wavs: Dict[int, Path],
    sources: Dict[int, Dict[str, Any]],
    translations: Dict[int, Dict[str, Any]],
) -> List[int]:
    wav_ids = set(wavs)
    source_ids = set(sources)
    translation_ids = set(translations)
    if wav_ids != source_ids or wav_ids != translation_ids:
        problems = []
        if wav_ids - source_ids:
            problems.append("WAV ids missing from source XML: {}".format(sorted(wav_ids - source_ids)[:20]))
        if source_ids - wav_ids:
            problems.append("source XML ids missing WAVs: {}".format(sorted(source_ids - wav_ids)[:20]))
        if wav_ids - translation_ids:
            problems.append(
                "WAV ids missing from translation XML: {}".format(sorted(wav_ids - translation_ids)[:20])
            )
        if translation_ids - wav_ids:
            problems.append(
                "translation XML ids missing WAVs: {}".format(sorted(translation_ids - wav_ids)[:20])
            )
        raise ValueError("segment-id sets do not match:\n  " + "\n  ".join(problems))

    for segment_id in sorted(wav_ids):
        source_docid = sources[segment_id]["docid"]
        translation_docid = translations[segment_id]["docid"]
        if source_docid != translation_docid:
            raise ValueError(
                "segment {} has different docids: source={!r}, translation={!r}".format(
                    segment_id, source_docid, translation_docid
                )
            )
    return sorted(wav_ids)


def infer_translation_xml(source_xml: Path, target_language: str) -> Path:
    suffix = ".en.xml"
    if not source_xml.name.endswith(suffix):
        raise ValueError(
            "cannot infer translation XML from {}; pass --translation-xml".format(source_xml)
        )
    return source_xml.with_name(source_xml.name[: -len(suffix)] + ".{}.xml".format(target_language))


def write_text_file(path: Path, content: str, overwrite: bool) -> str:
    payload = content + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8-sig")
        if existing == payload:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                "{} already exists with different content; use --overwrite to replace it".format(path)
            )
    path.write_text(payload, encoding="utf-8")
    return "written"


def write_json_file(path: Path, payload: Any, overwrite: bool) -> str:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8-sig")
        if existing == content:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                "{} already exists with different content; use --overwrite to replace it".format(path)
            )
    path.write_text(content, encoding="utf-8")
    return "written"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", type=Path, required=True, help="directory containing sent_<id>.wav")
    parser.add_argument("--source-xml", type=Path, required=True, help="English source XML")
    parser.add_argument(
        "--translation-xml",
        type=Path,
        help="translation XML; default: infer the sibling .zh.xml file",
    )
    parser.add_argument("--target-language", default="zh", help="used when inferring translation XML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="sidecar output directory; default: write beside the WAVs",
    )
    parser.add_argument(
        "--manifest-name",
        default="acl_gold_manifest.json",
        help="manifest filename inside output-dir",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace differing existing sidecars")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_dir = args.wav_dir.expanduser().resolve()
    source_xml = args.source_xml.expanduser().resolve()
    translation_xml = (
        args.translation_xml.expanduser().resolve()
        if args.translation_xml
        else infer_translation_xml(source_xml, args.target_language).resolve()
    )
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else wav_dir

    if not wav_dir.is_dir():
        raise NotADirectoryError(wav_dir)
    if not source_xml.is_file():
        raise FileNotFoundError(source_xml)
    if not translation_xml.is_file():
        raise FileNotFoundError(translation_xml)

    wavs = collect_wavs(wav_dir)
    sources = parse_xml(source_xml)
    translations = parse_xml(translation_xml)
    segment_ids = validate_alignment(wavs, sources, translations)

    manifest: List[Dict[str, Any]] = []
    for segment_id in segment_ids:
        source = sources[segment_id]
        translation = translations[segment_id]
        stem = wavs[segment_id].stem
        manifest.append(
            {
                "wav_id": stem,
                "wav": str(wavs[segment_id]),
                "global_segment_id": segment_id,
                "docid": source["docid"],
                "document_segment_index": source["document_segment_index"],
                "src": source["text"],
                "ref": translation["text"],
            }
        )

    docids = sorted({row["docid"] for row in manifest})
    print("validated {} WAV/XML segment pairs across {} documents".format(len(manifest), len(docids)))
    print("source XML: {}".format(source_xml))
    print("translation XML: {}".format(translation_xml))
    print("output directory: {}".format(output_dir))
    if args.dry_run:
        print("dry run: no files written")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "unchanged": 0}
    for row in manifest:
        stem = row["wav_id"]
        status = write_text_file(output_dir / (stem + ".txt"), row["src"], args.overwrite)
        counts[status] += 1
        status = write_text_file(
            output_dir / (stem + ".translation"), row["ref"], args.overwrite
        )
        counts[status] += 1

    manifest_path = output_dir / args.manifest_name
    manifest_status = write_json_file(manifest_path, manifest, args.overwrite)
    print(
        "sidecars: {} written, {} unchanged; manifest: {} ({})".format(
            counts["written"], counts["unchanged"], manifest_path, manifest_status
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise
