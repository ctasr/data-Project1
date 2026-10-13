"""Run all 62 predefined settings on the saved seed-42 validation split.

The default starts from an empty checkpoint. Use --resume after an interruption.
Previously published results are never used as measurements for this run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FORMAL_RESULTS = RESULTS / "formal_tuning_checkpoint.csv"
FORMAL_HISTORY = RESULTS / "formal_mlp_checkpoint.csv"
CHECKPOINT_META = RESULTS / "formal_tuning_checkpoint_meta.json"
PUBLISHED_RESULTS = RESULTS / "formal_tuning_results.csv"
PUBLISHED_HISTORY = RESULTS / "formal_mlp_history.csv"
FORMAL_INNOVATION = RESULTS / "formal_innovation_results.csv"
SEED = 42
LABELS = np.arange(10)


def canonical_id(algorithm: str, representation: str, parameters: dict) -> str:
    return json.dumps(
        [algorithm, representation, parameters],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def save_rows(rows: list[dict]) -> None:
    pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1"], ascending=False
    ).to_csv(FORMAL_RESULTS, index=False)


def score_row(
    algorithm: str,
    representation: str,
    parameters: dict,
    y_true: np.ndarray,
    prediction: np.ndarray,
    fit_seconds: float,
    provenance: str,
) -> dict:
    return {
        "config_id": canonical_id(algorithm, representation, parameters),
        "algorithm": algorithm,
        "representation": representation,
        "parameters": json.dumps(parameters, sort_keys=True),
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "fit_seconds": fit_seconds,
        "converged": True,
        "provenance": provenance,
    }


def add_or_fit(
    rows: list[dict],
    algorithm: str,
    representation: str,
    parameters: dict,
    estimator,
    x_train,
    y_train,
    x_validation,
    y_validation,
) -> None:
    config_id = canonical_id(algorithm, representation, parameters)
    if any(row["config_id"] == config_id for row in rows):
        print(f"SKIP formal checkpoint {algorithm} {parameters}", flush=True)
        return
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x_train, y_train)
    prediction = estimator.predict(x_validation)
    result = score_row(
        algorithm,
        representation,
        parameters,
        y_validation,
        prediction,
        time.perf_counter() - started,
        "formal_protocol_run",
    )
    result["converged"] = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    rows.append(result)
    save_rows(rows)
    print(
        f"DONE  {algorithm:20s} {str(parameters):55s} "
        f"accuracy={result['accuracy']:.4f} macro-F1={result['macro_f1']:.4f} "
        f"converged={result['converged']}",
        flush=True,
    )


def train_mlp(
    parameters: dict,
    x_train,
    y_train,
    x_validation,
    y_validation,
) -> tuple[dict, list[dict]]:
    model = MLPClassifier(
        hidden_layer_sizes=(parameters["hidden_units"],),
        activation=parameters["activation"],
        solver="adam",
        alpha=parameters["alpha"],
        # partial_fit already receives explicit batches of at most 128 rows;
        # "auto" avoids a harmless clipping warning on the final short batch.
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=1,
        random_state=SEED,
    )
    generator = np.random.default_rng(SEED)
    history = []
    best_model = None
    best_loss = np.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, 31):
        order = generator.permutation(len(y_train))
        first_batch = epoch == 1
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            kwargs = {"classes": LABELS} if first_batch else {}
            model.partial_fit(x_train[batch], y_train[batch], **kwargs)
            first_batch = False
        train_probability = model.predict_proba(x_train)
        validation_probability = model.predict_proba(x_validation)
        prediction = LABELS[np.argmax(validation_probability, axis=1)]
        epoch_row = {
            "config_id": canonical_id("MLP", "word_tfidf_12k_1-2", parameters),
            **parameters,
            "epoch": epoch,
            "train_loss": log_loss(y_train, train_probability, labels=LABELS),
            "validation_loss": log_loss(
                y_validation, validation_probability, labels=LABELS
            ),
            "validation_accuracy": accuracy_score(y_validation, prediction),
            "validation_macro_f1": f1_score(
                y_validation, prediction, average="macro"
            ),
        }
        history.append(epoch_row)
        if epoch_row["validation_loss"] < best_loss - 1e-4:
            best_loss = epoch_row["validation_loss"]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break
    prediction = best_model.predict(x_validation)
    result_parameters = {**parameters, "best_epoch": best_epoch}
    result = score_row(
        "MLP",
        "word_tfidf_12k_1-2",
        result_parameters,
        y_validation,
        prediction,
        time.perf_counter() - started,
        "formal_protocol_run",
    )
    result["config_id"] = canonical_id("MLP", "word_tfidf_12k_1-2", parameters)
    return result, history


def save_innovation_evidence(
    train_text,
    validation_text,
    y_train,
    y_validation,
    word_train,
    word_validation,
) -> None:
    fine = pd.read_csv(RESULTS / "fine_tuning.csv")
    expected_weights = {0.7, 0.85, 1.0, 1.15, 1.3}
    expected_cs = {0.3, 0.4, 0.5, 0.6, 0.8}
    actual_pairs = {
        (float(row.char_weight), float(row.C)) for row in fine.itertuples()
    }
    expected_pairs = set(itertools.product(expected_weights, expected_cs))
    if actual_pairs != expected_pairs or len(fine) != 25:
        raise RuntimeError("The required 5x5 hybrid grid is incomplete.")
    fine = fine.copy()
    fine["experiment"] = "hybrid_weight_x_C_complete_grid"
    fine["provenance"] = "existing_complete_seed42_grid"

    experiments = pd.read_csv(RESULTS / "experiment_results.csv")
    svm = experiments[experiments["algorithm"] == "LinearSVM"].copy()
    ablation_rows = []
    for representation in ("raw", "body", "dual"):
        candidate = svm[
            (svm["representation"] == representation)
            & (svm["parameters"].map(lambda value: json.loads(value)["C"] == 1.0))
        ]
        if len(candidate) != 1:
            raise RuntimeError(
                f"Controlled C=1 ablation row missing for {representation}."
            )
        row = candidate.iloc[0].to_dict()
        row["experiment"] = "controlled_representation_ablation_C1"
        row["provenance"] = "existing_complete_seed42_ablation"
        ablation_rows.append(row)

    # The hybrid ablation fixes both C=1 and equal word/character weighting.
    # This isolates the representation change from the later 5x5 tuning step.
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=80_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_train = char_vectorizer.fit_transform(train_text)
    char_validation = char_vectorizer.transform(validation_text)
    hybrid_train = sparse.hstack([word_train, char_train], format="csr")
    hybrid_validation = sparse.hstack(
        [word_validation, char_validation], format="csr"
    )
    started = time.perf_counter()
    hybrid_model = LinearSVC(C=1.0, random_state=SEED, dual="auto")
    hybrid_model.fit(hybrid_train, y_train)
    hybrid_prediction = hybrid_model.predict(hybrid_validation)
    hybrid_row = score_row(
        "LinearSVM",
        "hybrid",
        {"C": 1.0, "char_weight": 1.0},
        y_validation,
        hybrid_prediction,
        time.perf_counter() - started,
        "formal_protocol_run",
    )
    hybrid_row["experiment"] = "controlled_representation_ablation_C1"
    ablation_rows.append(hybrid_row)
    innovation = pd.concat([fine, pd.DataFrame(ablation_rows)], ignore_index=True)
    innovation.to_csv(FORMAL_INNOVATION, index=False)
    print(
        "INNOVATION verified: four-way controlled ablation + complete 5x5 grid",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Continue this run's saved checkpoint")
    arguments = parser.parse_args()
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    labelled = pd.read_csv(ROOT / "train_data.csv")
    digest = hashlib.sha256((ROOT / "train_data.csv").read_bytes()).hexdigest()
    environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    if environment["train_sha256"] != digest:
        raise RuntimeError("Training CSV changed after the development run")
    split = np.load(RESULTS / "split_indices.npz")
    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]
    expected_train, expected_validation = train_test_split(
        np.arange(len(labelled)), test_size=0.2, random_state=SEED,
        stratify=labelled["target"],
    )
    if not (np.array_equal(train_indices, expected_train)
            and np.array_equal(validation_indices, expected_validation)):
        raise RuntimeError("Saved split is not the fixed seed-42 split")
    train_text = labelled.iloc[train_indices]["text"].astype(str)
    validation_text = labelled.iloc[validation_indices]["text"].astype(str)
    y_train = labelled.iloc[train_indices]["target"].to_numpy()
    y_validation = labelled.iloc[validation_indices]["target"].to_numpy()
    print(
        f"FORMAL PROTOCOL seed={SEED}; train={len(y_train)}; "
        f"validation={len(y_validation)}; official test unopened",
        flush=True,
    )

    if arguments.resume and not FORMAL_RESULTS.exists():
        raise RuntimeError("No fresh-run checkpoint exists to resume")
    if arguments.resume:
        checkpoint_meta = json.loads(CHECKPOINT_META.read_text(encoding="utf-8"))
        if checkpoint_meta != {"seed": SEED, "train_sha256": digest}:
            raise RuntimeError("Checkpoint belongs to different data or seed")
    else:
        CHECKPOINT_META.write_text(
            json.dumps({"seed": SEED, "train_sha256": digest}, indent=2),
            encoding="utf-8",
        )
    rows = (pd.read_csv(FORMAL_RESULTS).to_dict(orient="records")
            if arguments.resume else [])
    histories = (pd.read_csv(FORMAL_HISTORY).to_dict(orient="records")
                 if arguments.resume and FORMAL_HISTORY.exists() else [])

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
    x_train = word_vectorizer.fit_transform(train_text)
    x_validation = word_vectorizer.transform(validation_text)
    representation = "word_tfidf_80k_1-2"

    print("\nBLOCK 1/4 MultinomialNB: 6 x 2 = 12", flush=True)
    for alpha, fit_prior in itertools.product(
        (0.001, 0.01, 0.03, 0.1, 1.0, 10.0), (True, False)
    ):
        parameters = {"alpha": alpha, "fit_prior": fit_prior}
        add_or_fit(
            rows,
            "MultinomialNB",
            representation,
            parameters,
            MultinomialNB(**parameters),
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

    print("\nBLOCK 2/4 LogisticRegression: 8 + 6 = 14", flush=True)
    for c_value, class_weight in itertools.product(
        (0.1, 1.0, 10.0, 100.0), (None, "balanced")
    ):
        parameters = {
            "C": c_value,
            "class_weight": class_weight,
            "solver": "lbfgs",
            "l1_ratio": 0.0,
        }
        add_or_fit(
            rows,
            "LogisticRegression",
            representation,
            parameters,
            LogisticRegression(
                **parameters, max_iter=3_000, random_state=SEED
            ),
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
    for l1_ratio, class_weight in itertools.product(
        (0.0, 0.5, 1.0), (None, "balanced")
    ):
        parameters = {
            "C": 10.0,
            "class_weight": class_weight,
            "solver": "saga",
            "l1_ratio": l1_ratio,
        }
        add_or_fit(
            rows,
            "LogisticRegression",
            representation,
            parameters,
            LogisticRegression(
                **parameters, max_iter=3_000, random_state=SEED
            ),
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

    print("\nBLOCK 3/4 SVM: linear 6 + RBF 18 = 24", flush=True)
    for c_value, class_weight in itertools.product(
        (0.1, 1.0, 10.0), (None, "balanced")
    ):
        parameters = {
            "kernel": "linear",
            "C": c_value,
            "gamma": "scale",
            "class_weight": class_weight,
        }
        add_or_fit(
            rows,
            "SVC",
            representation,
            parameters,
            SVC(**parameters, cache_size=2_000, random_state=SEED),
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
    for c_value, gamma, class_weight in itertools.product(
        (0.1, 1.0, 10.0), ("scale", 0.01, 0.1), (None, "balanced")
    ):
        parameters = {
            "kernel": "rbf",
            "C": c_value,
            "gamma": gamma,
            "class_weight": class_weight,
        }
        add_or_fit(
            rows,
            "SVC",
            representation,
            parameters,
            SVC(**parameters, cache_size=2_000, random_state=SEED),
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

    print("\nBLOCK 4/4 MLP: 3 x 2 x 2 = 12", flush=True)
    mlp_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
        max_features=12_000,
        dtype=np.float32,
    )
    x_mlp_train = mlp_vectorizer.fit_transform(train_text)
    x_mlp_validation = mlp_vectorizer.transform(validation_text)
    for hidden_units, alpha, activation in itertools.product(
        (32, 64, 128), (1e-4, 1e-2), ("relu", "tanh")
    ):
        parameters = {
            "hidden_units": hidden_units,
            "alpha": alpha,
            "activation": activation,
        }
        config_id = canonical_id("MLP", "word_tfidf_12k_1-2", parameters)
        if any(row["config_id"] == config_id for row in rows):
            print(f"SKIP formal checkpoint MLP {parameters}", flush=True)
            continue
        result, history = train_mlp(
            parameters,
            x_mlp_train,
            y_train,
            x_mlp_validation,
            y_validation,
        )
        rows.append(result)
        histories.extend(history)
        save_rows(rows)
        pd.DataFrame(histories).to_csv(FORMAL_HISTORY, index=False)
        print(
            f"DONE  MLP {parameters} accuracy={result['accuracy']:.4f} "
            f"macro-F1={result['macro_f1']:.4f}",
            flush=True,
        )

    save_innovation_evidence(
        train_text,
        validation_text,
        y_train,
        y_validation,
        x_train,
        x_validation,
    )
    frame = pd.DataFrame(rows)
    expected_counts = {
        "MultinomialNB": 12,
        "LogisticRegression": 14,
        "SVC": 24,
        "MLP": 12,
    }
    actual_counts = frame.groupby("algorithm").size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Formal grid incomplete: expected {expected_counts}, got {actual_counts}"
        )
    best = frame.sort_values(
        ["accuracy", "macro_f1"], ascending=False
    ).iloc[0]
    summary = {
        "seed": SEED,
        "train_sha256": digest,
        "train_size": len(y_train),
        "validation_size": len(y_validation),
        "official_test_used_for_tuning": False,
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "total_configurations": int(len(frame)),
        "best_algorithm": best["algorithm"],
        "best_representation": best["representation"],
        "best_parameters": json.loads(best["parameters"]),
        "best_accuracy": float(best["accuracy"]),
        "best_macro_f1": float(best["macro_f1"]),
    }
    shutil.copyfile(FORMAL_RESULTS, PUBLISHED_RESULTS)
    shutil.copyfile(FORMAL_HISTORY, PUBLISHED_HISTORY)
    (RESULTS / "formal_tuning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nFORMAL GRID COMPLETE: {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
