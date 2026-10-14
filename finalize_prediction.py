"""Select from completed validation searches, then open the official test set."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from main import (
    ExperimentConfig,
    LABELS,
    PATHS,
    feature_input,
    load_labelled_data,
    load_official_test_data,
    make_feature_extractor,
    make_word_tfidf,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates() -> pd.DataFrame:
    main = pd.read_csv(PATHS.results / "experiment_results.csv")
    formal = pd.read_csv(PATHS.results / "formal_tuning_results.csv")
    fine = pd.read_csv(PATHS.results / "fine_tuning.csv")
    fine_summary = json.loads(
        (PATHS.results / "fine_tuning_summary.json").read_text(encoding="utf-8")
    )
    development = json.loads(
        (PATHS.results / "development_summary.json").read_text(encoding="utf-8")
    )
    formal_summary = json.loads(
        (PATHS.results / "formal_tuning_summary.json").read_text(encoding="utf-8")
    )
    if development["quick_mode"] or development["seed"] != 42:
        raise RuntimeError("Finalization requires the full seed-42 development run")
    if (formal_summary["seed"] != 42
            or formal_summary["train_sha256"] != sha256(PATHS.train_csv)
            or formal_summary["total_configurations"] != 62):
        raise RuntimeError("Formal results do not belong to this complete seed-42 run")
    if (fine_summary.get("seed") != 42
            or fine_summary.get("train_sha256") != sha256(PATHS.train_csv)
            or fine_summary.get("training_rows") != development["training_rows"]
            or fine_summary.get("validation_rows") != development["validation_rows"]
            or fine_summary.get("char_weights") != [0.7, 0.85, 1.0, 1.15, 1.3]
            or fine_summary.get("c_values") != [0.3, 0.4, 0.5, 0.6, 0.8]
            or fine_summary.get("expected_configurations") != 25
            or fine_summary.get("complete") is not True):
        raise RuntimeError("Hybrid grid does not belong to this complete seed-42 run")
    expected = {"MultinomialNB": 12, "LogisticRegression": 14, "SVC": 24, "MLP": 12}
    if formal.groupby("algorithm").size().to_dict() != expected:
        raise RuntimeError("The 62-configuration formal search is incomplete")
    expected_pairs = {(w, c) for w in (0.7, 0.85, 1.0, 1.15, 1.3)
                      for c in (0.3, 0.4, 0.5, 0.6, 0.8)}
    actual_pairs = set(zip(fine["char_weight"], fine["C"]))
    if len(fine) != 25 or actual_pairs != expected_pairs:
        raise RuntimeError("The 25-configuration hybrid search is incomplete")
    rows = []
    for source, frame in (("main", main), ("formal", formal)):
        for _, row in frame.iterrows():
            rows.append({
                "source": source,
                "algorithm": row["algorithm"],
                "representation": row["representation"],
                "parameters": json.loads(row["parameters"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            })
    for _, row in fine.iterrows():
        rows.append({
            "source": "hybrid_grid",
            "algorithm": "LinearSVM",
            "representation": "hybrid",
            "parameters": {"C": float(row["C"]), "char_weight": float(row["char_weight"])},
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "source", "algorithm"],
        ascending=[False, False, True, True], kind="stable"
    ).reset_index(drop=True)


def build_pipeline(candidate: pd.Series, config: ExperimentConfig) -> Pipeline:
    algorithm = candidate["algorithm"]
    parameters = candidate["parameters"]
    representation = candidate["representation"]
    if representation == "word_tfidf_80k_1-2":
        features = make_word_tfidf(80_000)
    elif representation == "word_tfidf_12k_1-2":
        features = make_word_tfidf(12_000)
    else:
        feature_config = replace(
            config,
            hybrid_char_weight=float(parameters.get("char_weight", config.hybrid_char_weight)),
        )
        features = make_feature_extractor(representation, feature_config)
    if algorithm == "MultinomialNB":
        estimator = MultinomialNB(
            alpha=parameters["alpha"], fit_prior=parameters.get("fit_prior", True)
        )
    elif algorithm == "LogisticRegression":
        estimator = LogisticRegression(
            C=parameters["C"],
            class_weight=parameters.get("class_weight"),
            solver=parameters.get("solver", "lbfgs"),
            l1_ratio=parameters.get("l1_ratio"),
            max_iter=3_000 if candidate["source"] == "formal" else 2_000,
            random_state=config.seed,
        )
    elif algorithm == "SVC":
        estimator = SVC(
            C=parameters["C"], kernel=parameters["kernel"],
            gamma=parameters["gamma"], class_weight=parameters["class_weight"],
            cache_size=2_000, random_state=config.seed,
        )
    elif algorithm == "LinearSVM":
        estimator = LinearSVC(C=parameters["C"], dual="auto", random_state=config.seed)
    else:
        raise ValueError(f"Unexpected pipeline algorithm: {algorithm}")
    return Pipeline([("features", features), ("classifier", estimator)])


def train_mlp(candidate: pd.Series, training: pd.DataFrame, config: ExperimentConfig):
    parameters = candidate["parameters"]
    feature_count = 12_000 if candidate["source"] == "formal" else config.mlp_features
    vectorizer = make_word_tfidf(feature_count)
    matrix = vectorizer.fit_transform(training["text"])
    labels = training["target"].to_numpy()
    classifier = MLPClassifier(
        hidden_layer_sizes=(int(parameters["hidden_units"]),),
        activation=parameters.get("activation", config.mlp_activation),
        solver="adam", alpha=parameters.get("alpha", config.mlp_alpha),
        batch_size="auto", learning_rate_init=0.001, max_iter=1,
        random_state=config.seed,
    )
    generator = np.random.default_rng(config.seed)
    for epoch in range(int(parameters["best_epoch"])):
        order = generator.permutation(len(labels))
        first_batch = epoch == 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            keyword = {"classes": LABELS} if first_batch else {}
            classifier.partial_fit(matrix[batch], labels[batch], **keyword)
            first_batch = False
    return {"vectorizer": vectorizer, "classifier": classifier}


def predict(candidate: pd.Series, training: pd.DataFrame, evaluation: pd.DataFrame,
            config: ExperimentConfig):
    if candidate["algorithm"] == "MLP":
        artifact = train_mlp(candidate, training, config)
        predictions = artifact["classifier"].predict(
            artifact["vectorizer"].transform(evaluation["text"])
        )
    else:
        pipeline = build_pipeline(candidate, config)
        kind = candidate["representation"]
        source_kind = kind if kind in {"raw", "body", "dual", "hybrid"} else "raw"
        pipeline.fit(feature_input(training, source_kind), training["target"])
        predictions = pipeline.predict(feature_input(evaluation, source_kind))
        artifact = {"pipeline": pipeline}
    return np.asarray(predictions, dtype=int), artifact


def main() -> None:
    config = ExperimentConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    labelled = load_labelled_data()
    candidates = load_candidates()
    selected = candidates.iloc[0]
    all_indices = np.arange(len(labelled))
    expected_train, expected_validation = train_test_split(
        all_indices, test_size=config.validation_size, random_state=config.seed,
        stratify=labelled["target"],
    )
    saved = np.load(PATHS.results / "split_indices.npz")
    if not (np.array_equal(saved["train_indices"], expected_train)
            and np.array_equal(saved["validation_indices"], expected_validation)):
        raise RuntimeError("Saved split differs from the fixed seed-42 split")
    training = labelled.iloc[expected_train]
    validation = labelled.iloc[expected_validation]
    checked_prediction, _ = predict(selected, training, validation, config)
    checked_accuracy = accuracy_score(validation["target"], checked_prediction)
    checked_macro_f1 = f1_score(validation["target"], checked_prediction, average="macro")
    if (abs(checked_accuracy - selected["accuracy"]) > 1e-8
            or abs(checked_macro_f1 - selected["macro_f1"]) > 1e-8):
        raise RuntimeError("Selected validation result cannot be reproduced")
    pd.DataFrame({
        "row_index": expected_validation,
        "true_label": validation["target"].to_numpy(),
        "predicted_label": checked_prediction,
    }).to_csv(PATHS.results / "validation_predictions.csv", index=False)
    print("Selection verified before opening the official test set:",
          selected["algorithm"], selected["representation"], selected["parameters"],
          f"accuracy={checked_accuracy:.4f}", f"macro-F1={checked_macro_f1:.4f}", flush=True)

    official_test = load_official_test_data()
    train_test_overlap = len(set(labelled["text"]) & set(official_test["text"]))
    if train_test_overlap:
        raise RuntimeError("Official test contains exact text also present in training data")
    final_prediction, artifact = predict(selected, labelled, official_test, config)
    if len(final_prediction) != len(official_test):
        raise RuntimeError("Prediction count differs from official test rows")
    if not set(np.unique(final_prediction)).issubset(set(LABELS)):
        raise RuntimeError("Predicted label outside 0-9")
    PATHS.models.mkdir(exist_ok=True)
    pd.DataFrame(final_prediction).to_csv(PATHS.prediction_csv, header=False, index=False)
    artifact.update({"config": asdict(config), "selected_candidate": selected.to_dict()})
    joblib.dump(artifact, PATHS.models / "best_model.joblib", compress=3)
    classification = pd.DataFrame(classification_report(
        validation["target"], checked_prediction, labels=LABELS,
        output_dict=True, zero_division=0,
    )).T
    classification.to_csv(PATHS.results / "classification_report.csv", index_label="label")
    counts = pd.Series(final_prediction).value_counts().sort_index()
    summary = {
        "seed": config.seed,
        "train_sha256": sha256(PATHS.train_csv),
        "test_sha256": sha256(PATHS.test_csv),
        "labelled_rows": len(labelled),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "official_test_rows": len(official_test),
        "train_test_exact_text_overlap": train_test_overlap,
        "selected_source": selected["source"],
        "best_algorithm": selected["algorithm"],
        "best_representation": selected["representation"],
        "best_parameters": selected["parameters"],
        "best_accuracy": checked_accuracy,
        "best_macro_f1": checked_macro_f1,
        "prediction_class_counts": {str(k): int(v) for k, v in counts.items()},
    }
    (PATHS.results / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
