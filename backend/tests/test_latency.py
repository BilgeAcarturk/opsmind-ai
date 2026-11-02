import pytest

from app.evaluation.latency import percentile, summarize_latency


def test_percentile_median():
    values = [10, 20, 30, 40, 50]

    assert percentile(values, 50) == 30


def test_percentile_boundaries():
    values = [10, 20, 30]

    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 30


def test_percentile_interpolation():
    values = [10, 20, 30, 40]

    assert percentile(values, 50) == pytest.approx(25)


def test_invalid_percentile():
    with pytest.raises(ValueError):
        percentile([10, 20], 110)


def test_empty_percentile():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_negative_latency_rejected():
    with pytest.raises(ValueError):
        summarize_latency([10, -2])


def test_nan_latency_rejected():
    with pytest.raises(ValueError):
        summarize_latency([float("nan")])


def test_latency_summary():
    summary = summarize_latency([10, 20, 30, 40, 50])

    assert summary["count"] == 5
    assert summary["mean_ms"] == 30
    assert summary["p50_ms"] == 30
    assert summary["min_ms"] == 10
    assert summary["max_ms"] == 50
    assert summary["p95_ms"] >= summary["p50_ms"]
