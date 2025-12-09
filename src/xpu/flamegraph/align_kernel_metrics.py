#!/usr/bin/env python3
"""Project pm_sampling metrics onto individual GPU kernel intervals."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
import pathlib
import re
from typing import Dict, Iterable, List, Sequence

_HEADER_RE = re.compile(
    r"^\[CPU ts=(?P<ts>\d+) ns \(~[^,]+, (?P<human>[^)]+)\)\]\s+PID (?P<pid>\d+) "
    r"\(comm=(?P<comm>[^)]+)\)\s+TID (?P<tid>\d+)\s+CPU (?P<cpu>\d+)"
)
_KERNEL_RE = re.compile(r"^\s+GPU kernel: (?P<name>.+) \(corr=(?P<corr>\d+)\)")
_WINDOW_RE = re.compile(
    r"^\s+Launch start: (?P<launch>[\d\.]+) us, Kernel window: (?P<start>[\d\.]+)"
    r"-(?P<end>[\d\.]+) us \(duration (?P<duration>[\d\.]+) us\)"
)


def _us_str_to_ns(value: str) -> int:
    return int((Decimal(value) * 1000).to_integral_value())


def _collapse_chunks(lines: Iterable[str]) -> Iterable[List[str]]:
    chunk: List[str] = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped:
            chunk.append(stripped)
            continue
        if chunk:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


@dataclass
class KernelEvent:
    source: pathlib.Path
    idx: int
    pid: int
    tid: int
    comm: str
    cpu: int
    correlation_id: int
    kernel: str
    launch_us: int
    start_ns: int
    end_ns: int
    duration_ns: int


@dataclass
class PmSample:
    start_ns: int
    end_ns: int
    duration_ns: int
    metrics: Dict[str, float]


@dataclass
class KernelMetricResult:
    event: KernelEvent
    metrics: Dict[str, float]

    def as_row(self, metric_names: Sequence[str]) -> List[str]:
        row = [
            str(self.event.source),
            str(self.event.idx),
            str(self.event.pid),
            self.event.comm,
            str(self.event.tid),
            str(self.event.cpu),
            str(self.event.correlation_id),
            self.event.kernel,
            str(self.event.start_ns),
            str(self.event.end_ns),
            str(self.event.duration_ns),
        ]
        row.extend(f"{self.metrics[name]:.6f}" for name in metric_names)
        return row


def parse_timeline(path: pathlib.Path) -> List[KernelEvent]:
    events: List[KernelEvent] = []
    with path.open("r", encoding="utf-8") as fh:
        for chunk_idx, chunk in enumerate(_collapse_chunks(fh)):
            if len(chunk) < 3:
                continue
            header = _HEADER_RE.match(chunk[0])
            kernel = _KERNEL_RE.match(chunk[1])
            window = _WINDOW_RE.match(chunk[2])
            if not (header and kernel and window):
                continue
            start_ns = _us_str_to_ns(window.group("start"))
            end_ns = _us_str_to_ns(window.group("end"))
            duration_ns = max(end_ns - start_ns, 0)
            events.append(
                KernelEvent(
                    source=path,
                    idx=chunk_idx,
                    pid=int(header.group("pid")),
                    tid=int(header.group("tid")),
                    comm=header.group("comm"),
                    cpu=int(header.group("cpu")),
                    correlation_id=int(kernel.group("corr")),
                    kernel=kernel.group("name"),
                    launch_us=int(Decimal(window.group("launch"))),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    duration_ns=duration_ns,
                )
            )
    return events


def parse_pm_csv(path: pathlib.Path) -> tuple[List[PmSample], List[str]]:
    samples: List[PmSample] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        metric_names = [name for name in fieldnames if name not in {
            "range_index",
            "start_timestamp",
            "end_timestamp",
            "duration_ns",
        }]
        for row in reader:
            start_ns = int(row["start_timestamp"])
            end_ns = int(row["end_timestamp"])
            duration_ns = int(row["duration_ns"])
            metrics = {name: float(row[name]) for name in metric_names}
            samples.append(
                PmSample(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    duration_ns=max(duration_ns, 1),
                    metrics=metrics,
                )
            )
    samples.sort(key=lambda s: s.start_ns)
    return samples, metric_names


def project_metrics(
    kernels: Sequence[KernelEvent],
    samples: Sequence[PmSample],
    metric_names: Sequence[str],
) -> List[KernelMetricResult]:
    results: List[KernelMetricResult] = []
    sample_idx = 0
    num_samples = len(samples)
    for event in kernels:
        while sample_idx < num_samples and samples[sample_idx].end_ns <= event.start_ns:
            sample_idx += 1
        acc = {name: 0.0 for name in metric_names}
        idx = sample_idx
        while idx < num_samples and samples[idx].start_ns < event.end_ns:
            sample = samples[idx]
            overlap_start = max(event.start_ns, sample.start_ns)
            overlap_end = min(event.end_ns, sample.end_ns)
            if overlap_end > overlap_start:
                span = overlap_end - overlap_start
                weight = span / sample.duration_ns
                for name in metric_names:
                    acc[name] += sample.metrics[name] * weight
            if sample.end_ns <= event.end_ns:
                idx += 1
            else:
                break
        results.append(KernelMetricResult(event=event, metrics=acc))
    return results


def write_output(
    path: pathlib.Path,
    rows: Sequence[KernelMetricResult],
    metric_names: Sequence[str],
) -> None:
    header = [
        "timeline_source",
        "entry_index",
        "pid",
        "comm",
        "tid",
        "cpu",
        "correlation_id",
        "kernel_name",
        "kernel_start_ns",
        "kernel_end_ns",
        "kernel_duration_ns",
    ] + list(metric_names)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row.as_row(metric_names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", nargs="+", type=pathlib.Path, required=True, help="One or more cpu_gpu_timeline.txt files")
    parser.add_argument("--pm-csv", type=pathlib.Path, required=True, help="pm_sampling_metrics.csv file path")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("kernel_metric_alignment.csv"), help="Destination CSV path")
    args = parser.parse_args()

    timelines: List[KernelEvent] = []
    for path in args.timeline:
        if path.exists():
            timelines.extend(parse_timeline(path))
        else:
            print(f"Warning: timeline {path} not found; skipping")
    timelines.sort(key=lambda evt: evt.start_ns)
    samples, metric_names = parse_pm_csv(args.pm_csv)

    aligned = project_metrics(timelines, samples, metric_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_output(args.output, aligned, metric_names)
    print(f"Wrote {len(aligned)} kernel rows to {args.output}")


if __name__ == "__main__":
    main()
#   python align_kernel_metrics.py \
#   --timeline demores_dual/test1/cpu_gpu_timeline.txt demores_dual/test2/cpu_gpu_timeline.txt \
#   --pm-csv demores_dual/pm_sampling_metrics.csv \
#   --output demores_dual/kernel_metrics.csv