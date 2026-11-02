import math
from statistics import mean


def percentile(
    values: list[float],
    p: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate empty percentile")

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Latencies must be finite and non-negative")

    ordered = sorted(values)

    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latency(
    durations_ms: list[float],
) -> dict[str, float | int]:
    if not durations_ms:
        raise ValueError("No latency measurements")

    return {
        "count": len(durations_ms),
        "mean_ms": mean(durations_ms),
        "p50_ms": percentile(durations_ms, 50),
        "p95_ms": percentile(durations_ms, 95),
        "p99_ms": percentile(durations_ms, 99),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
    }
