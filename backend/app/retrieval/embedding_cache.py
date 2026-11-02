import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]: ...


class CachedEmbeddingModel:
    def __init__(
        self,
        model: EmbeddingEncoder,
        model_id: str,
        cache_dir: str | Path,
    ):
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")

        self._model = model
        self._model_id = model_id
        self._cache_dir = Path(cache_dir)

        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(
        self,
        sentences: list[str],
        normalize_embeddings: bool,
    ) -> str:
        payload = {
            "cache_schema": 1,
            "model_id": self._model_id,
            "sentences": sentences,
            "normalize_embeddings": normalize_embeddings,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> NDArray[np.float32]:
        if not convert_to_numpy:
            raise ValueError("Only NumPy output is supported")

        if not sentences:
            return np.empty((0, 0), dtype=np.float32)

        key = self._cache_key(
            sentences,
            normalize_embeddings,
        )

        path = self._cache_dir / f"{key}.npy"

        if path.exists():
            try:
                cached = np.load(
                    path,
                    allow_pickle=False,
                )

                if (
                    cached.ndim != 2
                    or cached.shape[0] != len(sentences)
                    or cached.shape[1] == 0
                    or not np.isfinite(cached).all()
                ):
                    raise ValueError("Invalid cached embeddings")

                self.cache_hits += 1
                return cached.astype(
                    np.float32,
                    copy=False,
                )

            except (OSError, ValueError):
                # Invalid cache: recompute from the model.
                pass

        self.cache_misses += 1

        embeddings = np.asarray(
            self._model.encode(
                sentences,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(sentences)
            or embeddings.shape[1] == 0
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("Invalid model embeddings")

        self._cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=self._cache_dir,
                suffix=".npy",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.save(temporary, embeddings)

            os.replace(temporary_path, path)

        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        return embeddings
