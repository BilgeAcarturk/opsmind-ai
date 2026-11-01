import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    sentences = [
        "Payment provider timeout",
        "External payment gateway is not responding",
        "Database connection pool exhausted",
    ]

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Embedding shape:", embeddings.shape)

    similarities = embeddings @ embeddings.T

    print("\nCosine similarity matrix:")
    print(np.round(similarities, 3))


if __name__ == "__main__":
    main()
