"""Run the predefined 5x5 hybrid-feature parameter grid used by main.py."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "fine_tuning.csv"
SEED = 42
CHAR_WEIGHTS = (0.7, 0.85, 1.0, 1.15, 1.3)
C_VALUES = (0.3, 0.4, 0.5, 0.6, 0.8)


def main() -> None:
    data = pd.read_csv(ROOT / "train_data.csv")
    split_path = ROOT / "results" / "split_indices.npz"
    if split_path.exists():
        saved_split = np.load(split_path)
        train_indices = saved_split["train_indices"]
        validation_indices = saved_split["validation_indices"]
    else:
        all_indices = np.arange(len(data))
        train_indices, validation_indices = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=SEED,
            stratify=data["target"],
        )
        split_path.parent.mkdir(exist_ok=True)
        np.savez(
            split_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
    train_text = data.iloc[train_indices]["text"].astype(str)
    validation_text = data.iloc[validation_indices]["text"].astype(str)
    y_train = data.iloc[train_indices]["target"].to_numpy()
    y_validation = data.iloc[validation_indices]["target"].to_numpy()

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
        max_features=80_000,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=80_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_train = word_vectorizer.fit_transform(train_text)
    word_validation = word_vectorizer.transform(validation_text)
    char_train = char_vectorizer.fit_transform(train_text)
    char_validation = char_vectorizer.transform(validation_text)

    OUTPUT.parent.mkdir(exist_ok=True)
    rows = []
    for char_weight, c_value in itertools.product(CHAR_WEIGHTS, C_VALUES):
        x_train = sparse.hstack(
            [word_train, char_train * char_weight], format="csr"
        )
        x_validation = sparse.hstack(
            [word_validation, char_validation * char_weight], format="csr"
        )
        started = time.perf_counter()
        model = LinearSVC(C=c_value, random_state=SEED, dual="auto")
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        row = {
            "char_weight": char_weight,
            "C": c_value,
            "accuracy": accuracy_score(y_validation, prediction),
            "macro_f1": f1_score(y_validation, prediction, average="macro"),
            "fit_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values(
            ["accuracy", "macro_f1"], ascending=False
        ).to_csv(OUTPUT, index=False)
        print(
            f"weight={char_weight:.2f} C={c_value:.1f} "
            f"accuracy={row['accuracy']:.4f} macro-F1={row['macro_f1']:.4f}"
        )

    ranked = pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1"], ascending=False
    )
    ranked.to_csv(OUTPUT, index=False)
    print("\nGrid complete. Best configurations:")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
