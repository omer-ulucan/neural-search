"""Latency benchmark helpers: percentiles, hardware facts, memory sampling, HTTP workloads."""

import asyncio
import ntpath
import os
import platform
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import psutil

PERCENTILES = (50, 95, 99)


def summarize(values_ms: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99 (linear interpolation), mean and max of a latency sample, in ms."""
    if not values_ms:
        return {"n": 0}
    arr = np.asarray(values_ms, dtype=float)
    out = {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in PERCENTILES}
    out.update(n=len(arr), mean=round(float(arr.mean()), 2), max=round(float(arr.max()), 2))
    return out


def windows_cpu_name() -> str | None:
    """Marketing name of the CPU from the registry; platform.processor() is only a family id."""
    try:
        import winreg

        path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip() or None
    except (ImportError, OSError):
        return None


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Windows" and (name := windows_cpu_name()):
        return name
    return platform.processor() or platform.machine()


def hardware() -> dict:
    """The machine facts a latency number is meaningless without."""
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "cpu": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "torch_threads": torch.get_num_threads(),
        "gpu": gpu or "none",
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def _nvidia_smi(query: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    return [[c.strip() for c in line.split(",", 2)] for line in out.splitlines() if line.strip()]


def gpu_load() -> dict | None:
    """Utilization, memory and processes of GPU 0 per nvidia-smi; None without nvidia-smi."""
    try:
        query = "--query-gpu=utilization.gpu,memory.used,memory.total"
        util, used, total = (float(x) for x in _nvidia_smi(query)[0])
        apps = _nvidia_smi("--query-compute-apps=pid,process_name")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    names = Counter(ntpath.basename(row[-1]) for row in apps)  # handles \ and / on any OS
    return {
        "utilization_pct": util,
        "memory_used_mb": used,
        "memory_total_mb": total,
        "processes": dict(sorted(names.items())),
    }


def background_load() -> dict:
    """CPU and GPU load while the benchmark itself is idle, i.e. other processes' load."""
    return {"cpu_busy_pct": psutil.cpu_percent(interval=1.0), "gpu": gpu_load()}


class MemorySampler:
    """Samples RSS of a process tree in a background thread."""

    def __init__(self, pid: int | None = None, interval_s: float = 0.1) -> None:
        self.proc = psutil.Process(pid or os.getpid())
        self.interval_s = interval_s
        self.samples_mb: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _rss_mb(self) -> float:
        procs = [self.proc, *self.proc.children(recursive=True)]
        total = 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total / 2**20

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples_mb.append(self._rss_mb())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        self.samples_mb.append(self._rss_mb())
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.samples_mb.append(self._rss_mb())

    def summary(self) -> dict[str, float]:
        s = self.samples_mb
        return {
            "start_mb": round(s[0], 1),
            "peak_mb": round(max(s), 1),
            "mean_mb": round(sum(s) / len(s), 1),
        }


@dataclass
class RequestResult:
    """Outcome of one request: client-side latency, success, and server-reported time."""

    latency_ms: float
    ok: bool
    server_ms: float | None = None
    error: str | None = None


@dataclass
class WorkloadResult:
    concurrency: int
    wall_s: float
    results: list[RequestResult] = field(default_factory=list)
    peak_in_flight: int = 0

    def report(self) -> dict:
        ok = [r for r in self.results if r.ok]
        errors = [r for r in self.results if not r.ok]
        overhead = [r.latency_ms - r.server_ms for r in ok if r.server_ms is not None]
        return {
            "concurrency": self.concurrency,
            "requests": len(self.results),
            "errors": len(errors),
            "error_rate": round(len(errors) / len(self.results), 4) if self.results else 0.0,
            "error_examples": sorted({e.error or "unknown" for e in errors})[:5],
            "wall_s": round(self.wall_s, 2),
            "throughput_rps": round(len(ok) / self.wall_s, 2) if self.wall_s else 0.0,
            "peak_in_flight": self.peak_in_flight,
            "latency_ms": summarize([r.latency_ms for r in ok]),
            "server_ms": summarize([r.server_ms for r in ok if r.server_ms is not None]),
            "client_minus_server_ms": summarize(overhead),
        }


async def run_workload(
    send: Callable[[str], Awaitable[RequestResult]],
    queries: Sequence[str],
    concurrency: int,
) -> WorkloadResult:
    """Closed loop: `concurrency` workers each send their next query as soon as one returns."""
    pending = list(queries)
    result = WorkloadResult(concurrency=concurrency, wall_s=0.0)
    in_flight = 0

    async def worker() -> None:
        nonlocal in_flight
        while pending:
            query = pending.pop(0)
            in_flight += 1
            result.peak_in_flight = max(result.peak_in_flight, in_flight)
            t = time.perf_counter()
            try:
                r = await send(query)
            except Exception as exc:  # timeouts, connection resets: count, do not crash
                r = RequestResult(0.0, False, error=type(exc).__name__)
            finally:
                in_flight -= 1
            r.latency_ms = (time.perf_counter() - t) * 1000.0
            result.results.append(r)

    t = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    result.wall_s = time.perf_counter() - t
    return result
