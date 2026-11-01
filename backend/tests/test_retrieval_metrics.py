import pytest

from app.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert recall_at_k(retrieved, relevant, 2) == 0.5


def test_reciprocal_rank():
    retrieved = ["B", "C", "A"]
    relevant = {"A", "C"}

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_no_relevant_result():
    retrieved = ["A", "B"]
    relevant = {"C"}

    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0


def test_mrr():
    values = [1.0, 0.5, 0.25]

    assert mean_reciprocal_rank(values) == pytest.approx(0.5833333333)


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, 0)

    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)


def test_empty_retrieval():
    assert precision_at_k([], {"A"}, 2) == 0.0
    assert recall_at_k([], {"A"}, 2) == 0.0
    assert reciprocal_rank([], {"A"}) == 0.0
