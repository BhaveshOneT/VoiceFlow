#!/usr/bin/env python3
"""Summarize VoiceFlow latency from log files."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


DEFAULT_LOG = (
    Path.home()
    / "Library"
    / "Application Support"
    / "VoiceFlow"
    / "logs"
    / "voiceflow.log"
)

_FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
_CAPTURE_RE = re.compile(rf"capture_stop_ms={_FLOAT}")
_PIPELINE_RE = re.compile(
    rf"Pipeline timings \(ms\): total={_FLOAT} stt={_FLOAT} clean={_FLOAT} "
    rf"refine={_FLOAT} finalize={_FLOAT}"
)
_E2E_RE = re.compile(
    rf"End-to-end post-record timings \(ms\): pipeline={_FLOAT} "
    rf"paste={_FLOAT} total={_FLOAT}"
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    arr = sorted(values)
    idx = (len(arr) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return arr[lo]
    frac = idx - lo
    return arr[lo] + (arr[hi] - arr[lo]) * frac


def summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    mean = sum(values) / len(values)
    p50 = percentile(values, 0.50)
    p90 = percentile(values, 0.90)
    p99 = percentile(values, 0.99)
    return (
        f"n={len(values)} mean={mean:.1f}ms "
        f"p50={p50:.1f}ms p90={p90:.1f}ms p99={p99:.1f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Path to log file")
    args = parser.parse_args()

    log_path: Path = args.log
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    capture_stop_ms: list[float] = []
    pipeline_total_ms: list[float] = []
    pipeline_stt_ms: list[float] = []
    pipeline_refine_ms: list[float] = []
    e2e_total_ms: list[float] = []
    e2e_paste_ms: list[float] = []

    for line in log_path.read_text(errors="ignore").splitlines():
        capture_match = _CAPTURE_RE.search(line)
        if capture_match:
            capture_stop_ms.append(float(capture_match.group(1)))

        pipeline_match = _PIPELINE_RE.search(line)
        if pipeline_match:
            pipeline_total_ms.append(float(pipeline_match.group(1)))
            pipeline_stt_ms.append(float(pipeline_match.group(2)))
            pipeline_refine_ms.append(float(pipeline_match.group(4)))

        e2e_match = _E2E_RE.search(line)
        if e2e_match:
            e2e_paste_ms.append(float(e2e_match.group(2)))
            e2e_total_ms.append(float(e2e_match.group(3)))

    print(f"log: {log_path}")
    print(f"capture_stop_ms: {summary(capture_stop_ms)}")
    print(f"pipeline_total_ms: {summary(pipeline_total_ms)}")
    print(f"pipeline_stt_ms: {summary(pipeline_stt_ms)}")
    print(f"pipeline_refine_ms: {summary(pipeline_refine_ms)}")
    print(f"e2e_paste_ms: {summary(e2e_paste_ms)}")
    print(f"e2e_total_ms: {summary(e2e_total_ms)}")


if __name__ == "__main__":
    main()
