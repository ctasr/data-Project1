"""Optional controlled MLP depth check; never changes the selected model.

The assignment asks whether adding a hidden layer helps. Both runs use the
same seed-42 split, training-only TF-IDF, width, activation, regularization,
mini-batches, and loss-based stopping rule. The official test is not read.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from main import ExperimentConfig, LABELS, PATHS, load_labelled_data, make_word_tfidf


def fit_depth(name, layers, x_train, y_train, x_validation, y_validation, config):
    model = MLPClassifier(
        hidden_layer_sizes=layers, activation="tanh", solver="adam",
        alpha=0.01, batch_size="auto", learning_rate_init=0.001,
        max_iter=1, random_state=config.seed,
    )
    generator = np.random.default_rng(config.seed)
    history = []
    best_model = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.mlp_max_epochs + 1):
        order = generator.permutation(len(y_train))
        first_batch = True
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            kwargs = {"classes": LABELS} if first_batch and epoch == 1 else {}
            model.partial_fit(x_train[batch], y_train[batch], **kwargs)
            first_batch = False
        train_probability = model.predict_proba(x_train)
        validation_probability = model.predict_proba(x_validation)
        prediction = LABELS[np.argmax(validation_probability, axis=1)]
        row = {
            "architecture": name,
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
        history.append(row)
        print(f"{name} epoch {epoch}: validation loss={row['validation_loss']:.4f}", flush=True)
        if row["validation_loss"] < best_loss - 1e-4:
            best_loss = row["validation_loss"]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.mlp_patience:
                break
    prediction = best_model.predict(x_validation)
    return {
        "architecture": name,
        "hidden_layer_sizes": json.dumps(layers),
        "best_epoch": best_epoch,
        "accuracy": accuracy_score(y_validation, prediction),
        "macro_f1": f1_score(y_validation, prediction, average="macro"),
        "validation_loss": best_loss,
    }, history


def main():
    config = ExperimentConfig()
    labelled = load_labelled_data()
    train_index, validation_index = train_test_split(
        np.arange(len(labelled)), test_size=config.validation_size,
        random_state=config.seed, stratify=labelled["target"],
    )
    saved = np.load(PATHS.results / "split_indices.npz")
    if not (np.array_equal(saved["train_indices"], train_index)
            and np.array_equal(saved["validation_indices"], validation_index)):
        raise RuntimeError("Depth check requires the saved fixed seed-42 split")
    train = labelled.iloc[train_index]
    validation = labelled.iloc[validation_index]
    vectorizer = make_word_tfidf(12_000)
    x_train = vectorizer.fit_transform(train["text"])
    x_validation = vectorizer.transform(validation["text"])
    y_train = train["target"].to_numpy()
    y_validation = validation["target"].to_numpy()
    results = []
    histories = []
    for name, layers in (("one_layer_64", (64,)), ("two_layers_64_32", (64, 32))):
        result, history = fit_depth(
            name, layers, x_train, y_train, x_validation, y_validation, config
        )
        results.append(result)
        histories.extend(history)
    PATHS.results.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(PATHS.results / "mlp_depth_comparison.csv", index=False)
    pd.DataFrame(histories).to_csv(PATHS.results / "mlp_depth_history.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    print("Explanatory check only: final model and prediction.csv unchanged.", flush=True)


if __name__ == "__main__":
    main()
