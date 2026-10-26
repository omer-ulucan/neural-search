import asyncio
import os

import pytest

from src.benchmark import MemorySampler, RequestResult, hardware, run_workload, summarize


def test_summarize_percentiles():
    s = summarize([float(i) for i in range(1, 101)])
    assert s["n"] == 100 and s["max"] == 100.0 and s["mean"] == 50.5
    assert s["p50"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)
    assert s["p99"] == pytest.approx(99.01)
    assert summarize([]) == {"n": 0}


@pytest.mark.parametrize("concurrency", [1, 5, 20])
def test_closed_loop_workload_respects_concurrency(concurrency):
    async def send(q):
        await asyncio.sleep(0.01)
        return RequestResult(0.0, True, server_ms=5.0)

    result = asyncio.run(run_workload(send, [f"q{i}" for i in range(40)], concurrency))
    report = result.report()
    assert report["requests"] == 40 and report["errors"] == 0
    assert result.peak_in_flight == concurrency
    assert report["latency_ms"]["p50"] >= 10.0
    assert report["client_minus_server_ms"]["p50"] >= 5.0
    assert report["throughput_rps"] > 0


def test_workload_counts_errors_and_exceptions():
    async def send(q):
        if q == "boom":
            raise TimeoutError
        if q == "bad":
            return RequestResult(0.0, False, error="HTTP 500")
        return RequestResult(0.0, True)

    report = asyncio.run(run_workload(send, ["ok", "boom", "bad", "ok"], 2)).report()
    assert report["requests"] == 4 and report["errors"] == 2
    assert report["error_rate"] == 0.5
    assert report["error_examples"] == ["HTTP 500", "TimeoutError"]
    assert report["latency_ms"]["n"] == 2  # only successes feed latency percentiles


def test_memory_sampler_reports_this_process():
    with MemorySampler(os.getpid(), interval_s=0.01) as mem:
        blob = bytearray(20 * 2**20)
        asyncio.run(asyncio.sleep(0.05))
    s = mem.summary()
    assert s["peak_mb"] >= s["start_mb"] > 0 and len(mem.samples_mb) >= 3
    del blob


def test_hardware_reports_cpu_threads_and_gpu():
    hw = hardware()
    assert hw["cpu"] and hw["logical_cores"] >= 1 and hw["torch_threads"] >= 1
    assert hw["gpu"]


def test_windows_cpu_uses_registry_name(monkeypatch):
    from src import benchmark

    def no_cpuinfo(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(benchmark, "open", no_cpuinfo, raising=False)
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Windows")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "Intel64 Family 6")
    monkeypatch.setattr(benchmark, "windows_cpu_name", lambda: "Intel(R) Core(TM) i7-13700HX")
    assert benchmark.cpu_model() == "Intel(R) Core(TM) i7-13700HX"
    monkeypatch.setattr(benchmark, "windows_cpu_name", lambda: None)
    assert benchmark.cpu_model() == "Intel64 Family 6"
