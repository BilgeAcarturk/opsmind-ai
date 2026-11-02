import numpy as np
import pytest

from app.retrieval.embedding_cache import CachedEmbeddingModel


class FakeEmbeddingModel:
    def __init__(self):
        self.call_count = 0

    def encode(
        self,
        sentences,
        *,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        self.call_count += 1

        return np.array(
            [[float(len(text)), 1.0] for text in sentences],
            dtype=np.float32,
        )


def test_cache_hit_avoids_recomputation(tmp_path):
    model = FakeEmbeddingModel()

    cached = CachedEmbeddingModel(
        model,
        model_id="fake-v1",
        cache_dir=tmp_path,
    )

    first = cached.encode(["payment timeout"])
    second = cached.encode(["payment timeout"])

    np.testing.assert_array_equal(first, second)

    assert model.call_count == 1
    assert cached.cache_hits == 1
    assert cached.cache_misses == 1


def test_cache_persists_between_instances(tmp_path):
    first_model = FakeEmbeddingModel()

    first_cache = CachedEmbeddingModel(
        first_model,
        "fake-v1",
        tmp_path,
    )

    first_cache.encode(["database failure"])

    second_model = FakeEmbeddingModel()

    second_cache = CachedEmbeddingModel(
        second_model,
        "fake-v1",
        tmp_path,
    )

    second_cache.encode(["database failure"])

    assert second_model.call_count == 0
    assert second_cache.cache_hits == 1


def test_text_change_invalidates_cache(tmp_path):
    model = FakeEmbeddingModel()
    cached = CachedEmbeddingModel(
        model,
        "fake-v1",
        tmp_path,
    )

    cached.encode(["payment timeout"])
    cached.encode(["payment authentication"])

    assert model.call_count == 2


def test_model_change_invalidates_cache(tmp_path):
    model_a = FakeEmbeddingModel()
    model_b = FakeEmbeddingModel()

    CachedEmbeddingModel(model_a, "model-v1", tmp_path).encode(["payment"])

    CachedEmbeddingModel(model_b, "model-v2", tmp_path).encode(["payment"])

    assert model_a.call_count == 1
    assert model_b.call_count == 1


def test_corrupt_cache_is_recomputed(tmp_path):
    model = FakeEmbeddingModel()
    cached = CachedEmbeddingModel(
        model,
        "fake-v1",
        tmp_path,
    )

    cached.encode(["payment"])

    cache_file = next(tmp_path.glob("*.npy"))
    cache_file.write_bytes(b"corrupted file")

    cached.encode(["payment"])

    assert model.call_count == 2
    assert cached.cache_misses == 2


def test_invalid_model_id(tmp_path):
    with pytest.raises(ValueError):
        CachedEmbeddingModel(
            FakeEmbeddingModel(),
            "",
            tmp_path,
        )
