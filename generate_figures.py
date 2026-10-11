"""Generate compact figures from the completed formal experiment evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def best_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]


def model_comparison(formal: pd.DataFrame, main: pd.DataFrame) -> None:
    rows = []
    labels = []
    for algorithm, label in (
        ("MultinomialNB", "MultinomialNB"),
        ("LogisticRegression", "Logistic regression"),
        ("SVC", "RBF SVC"),
        ("MLP", "MLP"),
    ):
        rows.append(best_row(formal[formal["algorithm"] == algorithm]))
        labels.append(label)
    word = main[(main["algorithm"] == "LinearSVM") & (main["representation"] == "raw")]
    hybrid = main[(main["algorithm"] == "LinearSVM") & (main["representation"] == "hybrid")]
    rows.extend([best_row(word), best_row(hybrid)])
    labels.extend(["LinearSVC word", "LinearSVC word+char"])

    accuracy = np.array([float(row["accuracy"]) for row in rows])
    macro_f1 = np.array([float(row["macro_f1"]) for row in rows])
    y = np.arange(len(labels))
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.barh(y + 0.17, accuracy, height=0.32, label="Accuracy", color="#4C78A8")
    axis.barh(y - 0.17, macro_f1, height=0.32, label="Macro-F1", color="#F58518")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.88, 0.96)
    axis.set_xlabel("Validation score")
    axis.set_title("Best validation result for each model and representation")
    axis.legend(loc="upper right", frameon=True)
    for row_index, (acc, f1) in enumerate(zip(accuracy, macro_f1)):
        axis.text(acc + 0.0006, row_index + 0.17, f"{acc:.4f}", va="center", fontsize=8)
        axis.text(f1 + 0.0006, row_index - 0.17, f"{f1:.4f}", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "model_comparison.png", dpi=220)
    plt.close(figure)


def ablation_comparison(innovation: pd.DataFrame) -> None:
    names = ["raw", "body", "dual", "hybrid"]
    display = ["Raw full text", "Cleaned body", "Subject + body", "Word + character"]
    scores = []
    controlled = innovation[
        innovation["experiment"] == "controlled_representation_ablation_C1"
    ]
    for name in names:
        subset = controlled[controlled["representation"] == name]
        if len(subset) != 1:
            raise RuntimeError(
                f"Expected exactly one controlled C=1 ablation row for {name}"
            )
        scores.append(float(subset.iloc[0]["macro_f1"]))
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    bars = axis.bar(display, scores, color=["#4C78A8", "#F58518", "#E45756", "#54A24B"])
    axis.set_ylim(0.89, 0.96)
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Controlled feature representation ablation")
    axis.tick_params(axis="x", labelrotation=10)
    for bar, value in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(FIGURES / "ablation_comparison.png", dpi=220)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    formal = pd.read_csv(RESULTS / "formal_tuning_results.csv")
    main_results = pd.read_csv(RESULTS / "experiment_results.csv")
    innovation = pd.read_csv(RESULTS / "formal_innovation_results.csv")
    model_comparison(formal, main_results)
    ablation_comparison(innovation)
    print("report figures generated")


if __name__ == "__main__":
    main()
