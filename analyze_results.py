"""Inspect fixed-split mistakes and repeat the locked final model across splits."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from main import (
    ExperimentConfig,
    PATHS,
    load_labelled_data,
    make_feature_extractor,
    make_word_tfidf,
)


def inspect_mistakes() -> None:
    validation = pd.read_csv(PATHS.results / "validation_predictions.csv")
    mistakes = validation[validation["true_label"] != validation["predicted_label"]].copy()
    mistakes.to_csv(PATHS.results / "misclassified_rows.csv", index=False)
    pairs = (mistakes.groupby(["true_label", "predicted_label"])
             .size().reset_index(name="count")
             .sort_values(["count", "true_label", "predicted_label"],
                          ascending=[False, True, True]))
    pairs.to_csv(PATHS.results / "confusion_pairs.csv", index=False)
    print("Misclassified validation rows:", len(mistakes), flush=True)
    print("Largest confusion pairs:\n", pairs.head(10).to_string(index=False), flush=True)


def check_split_stability() -> None:
    summary = json.loads((PATHS.results / "run_summary.json").read_text(encoding="utf-8"))
    if (summary["best_algorithm"] != "LinearSVM"
            or summary["best_representation"] != "hybrid"):
        raise RuntimeError("Stability check needs the selected hybrid LinearSVM")
    parameters = summary["best_parameters"]
    config = replace(ExperimentConfig(), hybrid_char_weight=parameters["char_weight"])
    labelled = load_labelled_data()
    rows = []
    for seed in (17, 42, 73):
        train_index, validation_index = train_test_split(
            np.arange(len(labelled)), test_size=config.validation_size,
            random_state=seed, stratify=labelled["target"],
        )
        training = labelled.iloc[train_index]
        validation = labelled.iloc[validation_index]
        for name, representation, c_value in (
            ("word_baseline", "raw", 1.0),
            ("locked_hybrid", "hybrid", parameters["C"]),
        ):
            model = Pipeline([
                ("features", make_feature_extractor(representation, config)),
                ("classifier", LinearSVC(C=c_value, dual="auto", random_state=seed)),
            ])
            model.fit(training["text"], training["target"])
            prediction = model.predict(validation["text"])
            row = {
                "model": name,
                "split_seed": seed,
                "accuracy": accuracy_score(validation["target"], prediction),
                "macro_f1": f1_score(validation["target"], prediction, average="macro"),
            }
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(PATHS.results / "stability_runs.csv", index=False)
    statistics = result.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    statistics.to_csv(PATHS.results / "stability_summary.csv")
    print(statistics.to_string(), flush=True)


def check_feature_budget() -> None:
    """Hold classifier/C/split fixed while testing matched feature budgets."""

    labelled = load_labelled_data()
    config = ExperimentConfig()
    train_index, validation_index = train_test_split(
        np.arange(len(labelled)), test_size=config.validation_size,
        random_state=config.seed, stratify=labelled["target"],
    )
    training = labelled.iloc[train_index]
    validation = labelled.iloc[validation_index]
    designs = (
        ("full_word_80k", make_word_tfidf(80_000), "text"),
        ("full_word_160k", make_word_tfidf(160_000), "text"),
        ("clean_body_80k", make_word_tfidf(80_000), "body"),
        ("word_char_80k_each", make_feature_extractor("hybrid", replace(
            config, hybrid_char_weight=1.0)), "text"),
    )
    rows = []
    for name, features, text_column in designs:
        model = Pipeline([
            ("features", features),
            ("classifier", LinearSVC(C=1.0, dual="auto", random_state=config.seed)),
        ])
        start = time.perf_counter()
        model.fit(training[text_column], training["target"])
        prediction = model.predict(validation[text_column])
        row = {
            "design": name,
            "seed": config.seed,
            "C": 1.0,
            "feature_count": len(model.named_steps["features"].get_feature_names_out()),
            "fit_predict_seconds": time.perf_counter() - start,
            "accuracy": accuracy_score(validation["target"], prediction),
            "macro_f1": f1_score(validation["target"], prediction, average="macro"),
        }
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(PATHS.results / "matched_budget_ablation.csv", index=False)


def check_bow_vs_tfidf() -> None:
    """Compare raw word counts with TF-IDF under the same split and SVM."""

    labelled = load_labelled_data()
    config = ExperimentConfig()
    train_index, validation_index = train_test_split(
        np.arange(len(labelled)), test_size=config.validation_size,
        random_state=config.seed, stratify=labelled["target"],
    )
    training = labelled.iloc[train_index]
    validation = labelled.iloc[validation_index]
    word_counts = CountVectorizer(
        lowercase=True, strip_accents="unicode", min_df=2, max_df=0.98,
        ngram_range=(1, 2), max_features=80_000, dtype=np.float32,
    )
    rows = []
    for name, extractor in (
        ("bag_of_words_counts", word_counts),
        ("word_tfidf", make_word_tfidf(80_000)),
    ):
        model = Pipeline([
            ("features", extractor),
            ("classifier", LinearSVC(C=1.0, dual="auto", random_state=config.seed)),
        ])
        model.fit(training["text"], training["target"])
        prediction = model.predict(validation["text"])
        rows.append({
            "representation": name,
            "split_seed": config.seed,
            "C": 1.0,
            "feature_count": len(model.named_steps["features"].get_feature_names_out()),
            "accuracy": accuracy_score(validation["target"], prediction),
            "macro_f1": f1_score(validation["target"], prediction, average="macro"),
        })
    result = pd.DataFrame(rows)
    result.to_csv(PATHS.results / "bow_tfidf_comparison.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    inspect_mistakes()
    check_split_stability()
    check_feature_budget()
    check_bow_vs_tfidf()
