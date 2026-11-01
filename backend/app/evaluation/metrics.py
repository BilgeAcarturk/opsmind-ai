def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    hits = sum(document_id in relevant for document_id in retrieved[:k])

    return hits / k


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant:
        return 0.0

    hits = len(set(retrieved[:k]) & relevant)

    return hits / len(relevant)


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:
    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
