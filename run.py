"""
Produce submission.csv, the predictions we upload to AIcrowd.

Final model (the primary configuration of report/report.tex):
  - preprocessing: the full pipeline of ex.ipynb (cell 15)
  - model: regularised logistic regression, lambda = 1e-4, gamma = 0.3,
    1000 gradient descent steps from w = 0
  - decision threshold: the one maximising F1 on the training predictions

Preprocessing, model and threshold are all fitted on the complete labelled
set (x_train.csv) and then applied to x_test.csv. Every step is
deterministic, so running the script again writes the same file.

The pipeline, the training call and the threshold search are imported from
report/experiments.py, so this script uses exactly the code behind the
numbers in the report.

Usage (from the repository root, with x_train.csv, y_train.csv and
x_test.csv in dataset/):
    python run.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "report"))

from experiments import (  # noqa: E402
    REF,
    add_bias,
    f1_optimal_threshold,
    log,
    metrics_at,
    run_full_pipeline,
    train_logistic,
)
from helpers import create_csv_submission, load_csv_data  # noqa: E402
from implementations import sigmoid  # noqa: E402

OUTPUT_PATH = os.path.join(REPO_ROOT, "submission.csv")


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset")
    y = (y_train == 1).astype(int)  # logistic regression expects {0, 1}
    log(f"data: train {x_train.shape}, test {x_test.shape}")

    # Fit the preprocessing on all labelled data and apply it to the
    # competition set. The pipeline only prints statistics of its
    # validation labels, so placeholders are enough for the test rows.
    F_train, F_test, names = run_full_pipeline(
        x_train, y, x_test, np.zeros(len(x_test), dtype=int)
    )
    del x_train, x_test
    tx_train, tx_test = add_bias(F_train), add_bias(F_test)
    del F_train, F_test
    log(f"{len(names)} features after preprocessing")

    w, loss = train_logistic(y, tx_train, **REF)
    s_train = sigmoid(tx_train @ w)
    t = f1_optimal_threshold(y, s_train)
    train = metrics_at(y, s_train, t)
    log(f"training loss {loss:.4f}, threshold {t:.4f}, training F1 {train['f1']:.4f}")

    # A score at or above the threshold is a positive prediction, as in
    # report/experiments.py; the submission format uses labels {-1, 1}.
    pred = np.where(sigmoid(tx_test @ w) >= t, 1, -1)
    create_csv_submission(test_ids, pred, OUTPUT_PATH)
    log(f"wrote {OUTPUT_PATH}: {len(pred)} rows, {np.mean(pred == 1):.1%} positive")


if __name__ == "__main__":
    main()
