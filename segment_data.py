#!/usr/bin/env python3
"""Split ACL6060 and RealSI recordings using their millisecond timestamps.

Outputs live beside each source WAV, in ``<recording>.segments/``. Each
segment has a WAV, source-language transcript (.txt), and translation
(.translation) with the same ``<recording>-<id>-<start>-<end>`` basename.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import sys
import wave


GROUPS = ("ACL6060/dev", "ACL6060/eval", "RealSI/en2zh", "RealSI/zh2en")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def input_paths(group: Path, wav: Path) -> tuple[Path, Path, Path]:
    stem = wav.stem
    if group.parent.name == "ACL6060":
        number = stem.rsplit(".", 1)[-1]
        text_root = group / "text"
        prefix = f"ACL.6060.{group.name}.segments.en-xx"
        transcript = text_root / f"{prefix}.en" / f"2022.acl-segments.{number}.txt"
        translation = text_root / f"{prefix}.zh" / f"2022.acl-segments.{number}.txt"
        timestamps = group / "segments_timestamp" / f"{stem}.txt"
    else:
        source, target = ("en", "zh") if group.name == "en2zh" else ("zh", "en")
        transcript = group / "text_segments" / f"{stem}.{source}.txt"
        translation = group / "text_segments" / f"{stem}.{target}.txt"
        timestamp_name = f"{stem}.timestamps.txt" if group.name == "zh2en" else f"{stem}.txt"
        timestamps = group / "gold_timestamp" / timestamp_name
    return timestamps, transcript, translation


def mono_pcm16(stereo_bytes: bytes) -> bytes:
    """Average the two signed 16-bit channels, truncating toward zero."""
    samples = array("h")
    samples.frombytes(stereo_bytes)
    if sys.byteorder != "little":
        samples.byteswap()
    mono = array("h", (int((samples[i] + samples[i + 1]) / 2)
                       for i in range(0, len(samples), 2)))
    if sys.byteorder != "little":
        mono.byteswap()
    return mono.tobytes()


def process_wav(group: Path, wav: Path) -> int:
    timestamps_path, transcript_path, translation_path = input_paths(group, wav)
    timestamps = read_lines(timestamps_path)
    transcripts = read_lines(transcript_path)
    translations = read_lines(translation_path)
    if not (len(timestamps) == len(transcripts) == len(translations)):
        raise ValueError(
            f"{wav}: timestamp/transcript/translation line counts differ: "
            f"{len(timestamps)}/{len(transcripts)}/{len(translations)}"
        )

    with wave.open(str(wav), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.getnframes()
        real_si = group.parent.name == "RealSI"
        if real_si and (channels != 2 or sample_width != 2):
            raise ValueError(f"{wav}: RealSI requires stereo 16-bit PCM")
        if not real_si and channels != 1:
            raise ValueError(f"{wav}: expected mono ACL6060 audio")

        segments = []
        for segment_id, line in enumerate(timestamps, 1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{timestamps_path}:{segment_id}: expected start end")
            start_ms, end_ms = map(int, parts)
            start_frame = (start_ms * rate + 500) // 1000
            end_frame = (end_ms * rate + 500) // 1000
            if start_ms < 0 or end_ms <= start_ms or end_frame > frames or end_frame <= start_frame:
                raise ValueError(f"{timestamps_path}:{segment_id}: invalid range {line!r}")
            segments.append((start_ms, end_ms, start_frame, end_frame))

        output_dir = wav.parent / f"{wav.stem}.segments"
        output_dir.mkdir(exist_ok=True)
        for segment_id, (start_ms, end_ms, start_frame, end_frame) in enumerate(segments, 1):
            basename = f"{wav.stem}-{segment_id}-{start_ms}-{end_ms}"
            source.setpos(start_frame)
            raw = source.readframes(end_frame - start_frame)
            if real_si:
                raw = mono_pcm16(raw)
            with wave.open(str(output_dir / f"{basename}.wav"), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(sample_width)
                target.setframerate(rate)
                target.writeframes(raw)
            (output_dir / f"{basename}.txt").write_text(transcripts[segment_id - 1] + "\n", encoding="utf-8")
            (output_dir / f"{basename}.translation").write_text(
                translations[segment_id - 1] + "\n", encoding="utf-8"
            )
    return len(segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--group", choices=GROUPS, action="append", help="Process selected group(s); default: all four")
    args = parser.parse_args()

    total = 0
    for name in args.group or GROUPS:
        group = args.data_root / name
        wavs = sorted((group / "full_wavs").glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No WAV files in {group / 'full_wavs'}")
        count = sum(process_wav(group, wav) for wav in wavs)
        total += count
        print(f"{name}: {len(wavs)} recordings, {count} segments")
    print(f"Total: {total} segments")


if __name__ == "__main__":
    main()
