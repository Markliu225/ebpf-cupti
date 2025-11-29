#!/usr/bin/env python3
"""Merge multiple cpu_gpu_timeline.txt files into a single ordered view."""
from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, List

_HEADER_RE = re.compile(
    r"^\[CPU ts=(?P<ts>\d+) ns \(~[^,]+, (?P<human>[^)]+)\)\]\s+"
    r"PID (?P<pid>\d+) \(comm=(?P<comm>[^)]+)\)\s+TID (?P<tid>\d+)\s+CPU (?P<cpu>\d+)"
)
_KERNEL_RE = re.compile(r"^\s+GPU kernel: (?P<name>.+) \(corr=(?P<corr>\d+)\)")
_WINDOW_RE = re.compile(
    r"^\s+Launch start: (?P<launch>[\d\.]+) us, "
    r"Kernel window: (?P<start>[\d\.]+)-(?P<end>[\d\.]+) us \(duration (?P<duration>[\d\.]+) us\)"
)


@dataclass(order=True)
class TimelineEvent:
    sort_key: int
    source: pathlib.Path
    human_ts: str
    pid: int
    tid: int
    cpu: int
    comm: str
    kernel: str
    correlation_id: int
    launch_us: float
    kernel_start_us: float
    kernel_end_us: float
    duration_us: float

    def format(self) -> str:
        prefix = (
            f"[CPU ts={self.sort_key} ns (~{self.duration_us:.3f} us, {self.human_ts})] "
            f"PID {self.pid} (comm={self.comm}) TID {self.tid} CPU {self.cpu}"
        )
        kernel_line = f"  GPU kernel: {self.kernel} (corr={self.correlation_id}, source={self.source})"
        window_line = (
            "  Launch start: "
            f"{self.launch_us:.3f} us, Kernel window: {self.kernel_start_us:.3f}-"
            f"{self.kernel_end_us:.3f} us (duration {self.duration_us:.3f} us)"
        )
        return "\n".join((prefix, kernel_line, window_line))


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


def parse_timeline(path: pathlib.Path) -> List[TimelineEvent]:
    events: List[TimelineEvent] = []
    with path.open("r", encoding="utf-8") as fh:
        for chunk in _collapse_chunks(fh):
            if len(chunk) < 3:
                continue
            header = _HEADER_RE.match(chunk[0])
            kernel = _KERNEL_RE.match(chunk[1])
            window = _WINDOW_RE.match(chunk[2])
            if not (header and kernel and window):
                continue
            events.append(
                TimelineEvent(
                    sort_key=int(header.group("ts")),
                    source=path,
                    human_ts=header.group("human"),
                    pid=int(header.group("pid")),
                    tid=int(header.group("tid")),
                    cpu=int(header.group("cpu")),
                    comm=header.group("comm"),
                    kernel=kernel.group("name"),
                    correlation_id=int(kernel.group("corr")),
                    launch_us=float(window.group("launch")),
                    kernel_start_us=float(window.group("start")),
                    kernel_end_us=float(window.group("end")),
                    duration_us=float(window.group("duration")),
                )
            )
    return events


def combine_timelines(paths: Iterable[pathlib.Path]) -> List[TimelineEvent]:
    events: List[TimelineEvent] = []
    for path in paths:
        events.extend(parse_timeline(path))
    events.sort()
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="Paths to cpu_gpu_timeline.txt files to merge",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("combined_timeline.txt"),
        help="Destination file for the merged timeline",
    )
    args = parser.parse_args()

    events = combine_timelines(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(event.format())
            fh.write("\n\n")

    print(f"Wrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
