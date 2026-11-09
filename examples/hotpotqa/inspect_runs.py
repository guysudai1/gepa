"""Inspect live HotpotQA GEPA processes without reading secrets or payloads."""

from __future__ import annotations

import os
import socket
import struct
import time
from collections import Counter
from pathlib import Path


PROC = Path("/proc")
TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def command_line(pid: int) -> str:
    try:
        return (PROC / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def process_stat(pid: int) -> dict[str, int | str]:
    raw = read_text(PROC / str(pid) / "stat")
    close = raw.rfind(")")
    fields = raw[close + 2 :].split()
    if close < 0 or len(fields) < 22:
        return {}
    return {
        "state": fields[0],
        "ppid": int(fields[1]),
        "utime": int(fields[11]),
        "stime": int(fields[12]),
        "threads": int(fields[17]),
        "starttime": int(fields[19]),
        "rss_pages": int(fields[21]),
    }


def process_io(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in read_text(PROC / str(pid) / "io").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            result[key] = int(value.strip())
        except ValueError:
            pass
    return result


def socket_inodes(pid: int) -> set[str]:
    inodes: set[str] = set()
    fd_dir = PROC / str(pid) / "fd"
    try:
        entries = list(fd_dir.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return inodes
    for entry in entries:
        try:
            target = os.readlink(entry)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    return inodes


def decode_ipv4(value: str) -> str:
    return socket.inet_ntoa(struct.pack("<I", int(value, 16)))


def tcp_connections(pid: int, inodes: set[str]) -> list[str]:
    connections: list[str] = []
    for table in ("tcp", "tcp6"):
        lines = read_text(PROC / str(pid) / "net" / table).splitlines()[1:]
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[9] not in inodes:
                continue
            local_hex, remote_hex = fields[1], fields[2]
            local_addr, local_port = local_hex.split(":")
            remote_addr, remote_port = remote_hex.split(":")
            if table == "tcp":
                local_ip = decode_ipv4(local_addr)
                remote_ip = decode_ipv4(remote_addr)
            else:
                local_ip = "ipv6"
                remote_ip = "ipv6"
            connections.append(
                f"{local_ip}:{int(local_port, 16)} -> {remote_ip}:{int(remote_port, 16)} state={fields[3]}"
            )
    return connections


def open_data_files(pid: int) -> list[str]:
    paths: list[str] = []
    fd_dir = PROC / str(pid) / "fd"
    try:
        entries = list(fd_dir.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return paths
    for entry in entries:
        try:
            target = os.readlink(entry)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in target for marker in ("/experiments/", ".jsonl", ".parquet", "run_log")):
            paths.append(target)
    return sorted(set(paths))


def thread_waits(pid: int) -> Counter[str]:
    waits: Counter[str] = Counter()
    task_dir = PROC / str(pid) / "task"
    try:
        tasks = list(task_dir.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return waits
    for task in tasks:
        wait = read_text(task / "wchan").strip() or "running/unknown"
        waits[wait] += 1
    return waits


def live_pids() -> list[int]:
    found: list[int] = []
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = command_line(pid)
        if "examples/hotpotqa/main.py" in cmd and "inspect_runs.py" not in cmd:
            found.append(pid)
    return sorted(found)


def main() -> None:
    pids = live_pids()
    if not pids:
        print("No live HotpotQA GEPA processes found in this PID namespace.")
        return

    before = {pid: (process_stat(pid), process_io(pid)) for pid in pids}
    time.sleep(3)
    uptime = float(read_text(PROC / "uptime").split()[0])

    for pid in pids:
        stat = process_stat(pid)
        io = process_io(pid)
        old_stat, old_io = before.get(pid, ({}, {}))
        if not stat:
            print(f"PID {pid}: exited during sampling")
            continue
        cpu_delta = (
            int(stat["utime"])
            + int(stat["stime"])
            - int(old_stat.get("utime", 0))
            - int(old_stat.get("stime", 0))
        ) / TICKS
        elapsed = uptime - int(stat["starttime"]) / TICKS
        rss_mib = int(stat["rss_pages"]) * PAGE_SIZE / 1024 / 1024
        sockets = socket_inodes(pid)

        print(f"\nPID {pid}")
        print(f"  command: {command_line(pid)}")
        print(f"  state={stat['state']} elapsed={elapsed:.0f}s rss={rss_mib:.1f} MiB threads={stat['threads']}")
        print(f"  CPU used in sample: {cpu_delta:.3f}s / 3s")
        print(
            "  IO delta: "
            f"read={io.get('read_bytes', 0) - old_io.get('read_bytes', 0):,} bytes, "
            f"write={io.get('write_bytes', 0) - old_io.get('write_bytes', 0):,} bytes, "
            f"read syscalls={io.get('syscr', 0) - old_io.get('syscr', 0):,}, "
            f"write syscalls={io.get('syscw', 0) - old_io.get('syscw', 0):,}"
        )
        print(f"  thread waits: {dict(thread_waits(pid))}")
        connections = tcp_connections(pid, sockets)
        print(f"  TCP connections ({len(connections)}):")
        for connection in connections:
            print(f"    {connection}")
        files = open_data_files(pid)
        print(f"  open experiment files ({len(files)}):")
        for path in files:
            print(f"    {path}")


if __name__ == "__main__":
    main()
