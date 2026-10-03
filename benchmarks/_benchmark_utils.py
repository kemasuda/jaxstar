"""Shared reporting and synchronized timing for repository-only benchmarks."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jaxlib
import numpy as np


def environment(device):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "device_count": len(jax.devices()),
        "x64_enabled": bool(jax.config.x64_enabled),
    }


def measure(function, args, *, rounds=20, warmup=3, time_first=False):
    """Trace/lower, compile, then time device-resident, synchronized calls.

    The caller places and synchronizes args before entering this function.
    block_until_ready traverses the complete result PyTree, including all AD
    leaves. Setup, transfers, compilation and warmup are outside warm timings.
    time_first adds one synchronized first execution before the warmup calls,
    matching the historical legacy benchmark without changing existing callers.
    """
    start = perf_counter()
    lowered = jax.jit(function).lower(*args)
    lowering_seconds = perf_counter() - start
    start = perf_counter()
    compiled = lowered.compile()
    xla_compile_seconds = perf_counter() - start
    first_execution_seconds = None
    if time_first:
        start = perf_counter()
        jax.block_until_ready(compiled(*args))
        first_execution_seconds = perf_counter() - start
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    durations = []
    for _ in range(rounds):
        start = perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        durations.append(perf_counter() - start)
    timing = {
        "compile_seconds": lowering_seconds + xla_compile_seconds,
        "lowering_seconds": lowering_seconds,
        "xla_compile_seconds": xla_compile_seconds,
        "warm_median_seconds": float(np.median(durations)),
        "warm_min_seconds": min(durations),
        "warm_mean_seconds": float(np.mean(durations)),
    }
    if time_first:
        timing["first_execution_seconds"] = first_execution_seconds
    memory = None
    try:
        analysis = compiled.memory_analysis()
        if analysis is not None:
            memory = {}
            for label, attribute in (
                ("temporary_bytes", "temp_size_in_bytes"),
                ("argument_bytes", "argument_size_in_bytes"),
                ("output_bytes", "output_size_in_bytes"),
            ):
                value = getattr(analysis, attribute, None)
                memory[label] = None if value is None else int(value)
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        # Some backends do not implement the supported memory-analysis API.
        pass
    return timing, memory, result


def emit_result(report, json_out=None):
    """Print exactly one final JSON block; optionally save the identical JSON."""
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")
    print("BENCHMARK_RESULT_BEGIN")
    print(encoded)
    print("BENCHMARK_RESULT_END")
