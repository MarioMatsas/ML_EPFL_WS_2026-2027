"""
Supplementary experiments behind the numbers in report/report.tex.

The preprocessing is NOT re-implemented here: we execute the preprocessing
cell of ex.ipynb (cell 15) and its split_data function (cell 5) directly, so
every number uses exactly the features of the notebook. Ablations patch that
cell at two hook points (see run_full_pipeline).

Parts:
  holdout  80/20 split with seed 2026 (same split as ex.ipynb)
             - the six methods of implementations.py on the full features
             - ablation: remove one preprocessing step at a time
             - paired bootstrap confidence intervals for the ablation
  cv       5-fold cross-validation on all labelled data; preprocessing and
           threshold are refit on every training fold

The decision threshold is always chosen to maximise F1 on the TRAINING
predictions. ex.ipynb chose it on the validation set; we report that number
too ("oracle") to show how optimistic it is.

Usage (from the repository root):
    python report/experiments.py holdout
    python report/experiments.py cv
Add --cache DIR to read x_train.npy / x_test.npy / y_train.npy from DIR
instead of parsing the CSV files (saves about a minute per run).

Results are merged into report/results.json. Runtime on a 6-core laptop
CPU: about 35 minutes for holdout and 90 minutes for cv.
"""

import contextlib
import io
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # the notebook cell reads dataset/x_train.csv relatively

from helpers import load_csv_data  # noqa: E402
from implementations import (  # noqa: E402
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)

RESULTS_PATH = os.path.join(REPO_ROOT, "report", "results.json")

SEED = 2026
N_FOLDS = 5

# Reference optimiser budget for all comparisons (rank 7 of the ex.ipynb
# grid, within 0.002 F1 of the best) and the configuration ex.ipynb selected.
REF = dict(lambda_=1e-4, gamma=0.3, max_iters=1000)
SELECTED = dict(lambda_=1e-4, gamma=0.6, max_iters=3000)
RIDGE_LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SGD_GAMMA = 1e-4  # one pass over the data, batch size 1

# Commit before "use decoded columns ..." = previous preprocessing (v1).
V1_COMMIT = "31cc81c"

# Survey metadata and sampling weights dropped by hand in ex.ipynb.
METADATA_FEATURES = {
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "QSTVER", "QSTLANG", "_STSTR", "_STRWT", "_RAWRAKE",
    "_WT2RAKE", "_CLLCPWT", "_DUALCOR", "_LLCPWT",
}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ----------------------------------------------------------------------
# Notebook code
# ----------------------------------------------------------------------

def notebook_cell(index, commit=None):
    """Source of one ex.ipynb cell, optionally at an older commit."""
    if commit is None:
        with open("ex.ipynb") as f:
            nb = json.load(f)
    else:
        raw = subprocess.run(
            ["git", "show", f"{commit}:ex.ipynb"],
            check=True, capture_output=True, text=True,
        ).stdout
        nb = json.loads(raw)
    return "".join(nb["cells"][index]["source"])


def patch_once(src, anchor, new):
    assert src.count(anchor) == 1, f"anchor not unique: {anchor!r}"
    return src.replace(anchor, new)


def run_full_pipeline(X_tr, y_tr, X_val, y_val, variant="full"):
    """
    Run the ex.ipynb preprocessing cell on (X_tr, X_val).

    Everything is fitted on X_tr only, as in the notebook. The variant
    removes one step:
      full          the notebook pipeline as committed
      v1            the cell from commit V1_COMMIT
      no_codes      no 'don't know / refused' decoding, no 88 -> 0, no remap
      no_onehot     nominal features kept as numbers (like ordinals)
      no_indicator  no missingness indicator for numerical/ordinal columns
      keep_meta     survey metadata and sampling weights are kept
      keep_sparse   features with >= 99% missing values are kept
    """
    src = notebook_cell(15, V1_COMMIT if variant == "v1" else None)

    # Hook 1: right after the hand-written drop list.
    anchor1 = "DROP_FEATURES &= set(feature_names)\n"
    src = patch_once(src, anchor1, anchor1 + "__hook1__(globals())\n")
    # Hook 2: everything is defined and classified, nothing is fitted yet.
    anchor2 = "preprocessing_params = fit_preprocessor("
    src = patch_once(src, anchor2, "__hook2__(globals())\n" + anchor2)

    if variant == "keep_sparse":
        src = patch_once(src, "SPARSE_THRESHOLD = 0.99", "SPARSE_THRESHOLD = 1.01")

    def hook1(ns):
        if variant == "keep_meta":
            ns["DROP_FEATURES"] -= METADATA_FEATURES

    def hook2(ns):
        if variant == "no_codes":
            ns["find_missing_codes"] = lambda col, name: set()
            ns["remap_ordinal"] = lambda col, name: col.copy()
            ns["clean_numerical_feature"] = lambda col, name: col.astype(float).copy()
        elif variant == "no_onehot":
            ns["ORDINAL_FEATURES"] |= ns["CATEGORICAL_FEATURES"]
            ns["CATEGORICAL_FEATURES"].clear()
        elif variant == "no_indicator":
            fit = ns["fit_preprocessor"]

            def fit_without_indicators(X, names):
                params = fit(X, names)
                for info in params.values():
                    if info["type"] in ("ordinal", "numerical"):
                        info["has_missing"] = False
                return params

            ns["fit_preprocessor"] = fit_without_indicators

    ns = {
        "__hook1__": hook1,
        "__hook2__": hook2,
        "x_train": X_tr,
        "train_X": X_tr,
        "test_X": X_val,
        "x_test": X_val[:100],  # competition set is not needed here
        "train_y": y_tr,
        "test_y": y_val,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "ex.ipynb[15]", "exec"), ns)
    result = (
        ns["train_X_processed"],
        ns["test_X_processed"],
        list(ns["processed_feature_names"]),
    )
    # The cell's functions reference ns, so free it explicitly.
    ns.clear()
    return result


def run_minimal_pipeline(X_tr, X_val):
    """
    Minimal baseline of ex-simple.ipynb: training-median imputation,
    removal of constant columns, standardisation. No feature engineering.
    """
    medians = np.nanmedian(X_tr, axis=0)
    X_tr = np.where(np.isnan(X_tr), medians, X_tr)
    X_val = np.where(np.isnan(X_val), medians, X_val)
    std = X_tr.std(axis=0)
    keep = std > 1e-12
    mean = X_tr[:, keep].mean(axis=0)
    return (X_tr[:, keep] - mean) / std[keep], (X_val[:, keep] - mean) / std[keep]


def add_bias(X):
    return np.column_stack([np.ones(X.shape[0]), X])


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def metrics_at(y, scores, threshold):
    """F1, precision, recall, accuracy of the rule 'positive if score >= t'."""
    pred = scores >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(
        f1=float(f1), precision=float(precision), recall=float(recall),
        accuracy=float(np.mean(pred == (y == 1))),
        positive_rate=float(np.mean(pred)), threshold=float(threshold),
    )


def f1_optimal_threshold(y, scores):
    """
    Threshold maximising F1 on (y, scores), searched exactly over all
    distinct score values: with the k highest scores predicted positive,
    F1 = 2 TP_k / (k + P).
    """
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    tp = np.cumsum(y[order] == 1)
    k = np.arange(1, len(y) + 1)
    f1 = 2 * tp / (k + np.sum(y == 1))
    # Only cut between different scores, so ties stay on the same side.
    valid = np.append(s[1:] < s[:-1], True)
    best = np.argmax(np.where(valid, f1, -1))
    return float(s[best])


def evaluate(y_tr, s_tr, y_val, s_val):
    """
    honest: threshold chosen on the training predictions
    oracle: threshold chosen on the validation predictions (ex.ipynb)
    """
    t = f1_optimal_threshold(y_tr, s_tr)
    t_oracle = f1_optimal_threshold(y_val, s_val)
    return dict(
        train=metrics_at(y_tr, s_tr, t),
        val=metrics_at(y_val, s_val, t),
        val_oracle=metrics_at(y_val, s_val, t_oracle),
    )


def paired_bootstrap(y, pred_a, pred_b, n_boot=1000, seed=SEED):
    """95% interval of F1(b) - F1(a) over bootstrap resamples of the rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pos = y == 1
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        p, a, b = pos[idx], pred_a[idx], pred_b[idx]
        f1_a = 2 * np.sum(a & p) / (np.sum(a) + np.sum(p))
        f1_b = 2 * np.sum(b & p) / (np.sum(b) + np.sum(p))
        diffs[i] = f1_b - f1_a
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def train_logistic(y, tx, lambda_, gamma, max_iters):
    w, loss = reg_logistic_regression(
        y, tx, lambda_, np.zeros(tx.shape[1]), max_iters, gamma
    )
    return w, float(loss)


def save(key, value):
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results[key] = value
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)


def load_data(cache):
    if cache:
        return (np.load(os.path.join(cache, "x_train.npy")),
                np.load(os.path.join(cache, "y_train.npy")))
    x_train, _, y_train, _, _ = load_csv_data("dataset")
    return x_train, y_train


def holdout_split(x, y):
    ns = {"np": np}
    exec(notebook_cell(5), ns)  # defines split_data exactly as in ex.ipynb
    X_tr, X_val, y_tr, y_val = ns["split_data"](x, y, 0.8, seed=SEED)
    return X_tr, X_val, (y_tr == 1).astype(int), (y_val == 1).astype(int)


# ----------------------------------------------------------------------
# Part 1: hold-out
# ----------------------------------------------------------------------

def run_holdout(x, y):
    X_tr, X_val, y_tr, y_val = holdout_split(x, y)
    log(f"hold-out split: train {X_tr.shape}, val {X_val.shape}, "
        f"positives {y_tr.mean():.4f} / {y_val.mean():.4f}")
    out = dict(n_train=len(y_tr), n_val=len(y_val),
               pos_train=float(y_tr.mean()), pos_val=float(y_val.mean()))

    # ---------------- full features, all six methods ----------------
    F_tr, F_val, names = run_full_pipeline(X_tr, y_tr, X_val, y_val)
    tx_tr, tx_val = add_bias(F_tr), add_bias(F_val)
    del F_tr, F_val
    log(f"full features: {tx_tr.shape[1] - 1} columns")
    out["n_features_full"] = tx_tr.shape[1] - 1

    gram = tx_tr.T @ tx_tr / len(y_tr)
    eig = np.linalg.eigvalsh(gram)
    out["gram_eig_max"] = float(eig[-1])
    out["gram_eig_min"] = float(eig[0])
    out["gram_cond"] = float(eig[-1] / max(eig[0], 1e-300))
    log(f"eig(X^T X / N): max {eig[-1]:.3f}, min {eig[0]:.3e}")

    models = {}
    models["majority"] = dict(
        val=metrics_at(y_val, np.zeros(len(y_val)), 1.0),
    )

    def record(name, w, loss, extra=None, scores=lambda tx, w: tx @ w):
        s_tr, s_val = scores(tx_tr, w), scores(tx_val, w)
        res = evaluate(y_tr, s_tr, y_val, s_val)
        res["loss"] = loss
        res.update(extra or {})
        models[name] = res
        log(f"{name:28s} val F1 {res['val']['f1']:.4f} "
            f"(oracle {res['val_oracle']['f1']:.4f}), train F1 {res['train']['f1']:.4f}")
        return s_tr, s_val

    try:
        w, loss = least_squares(y_tr.astype(float), tx_tr)
        if np.all(np.isfinite(w)):
            record("least_squares", w, float(loss), dict(w_norm=float(np.linalg.norm(w))))
        else:
            models["least_squares"] = dict(failed="non-finite weights")
    except np.linalg.LinAlgError as e:
        models["least_squares"] = dict(failed=str(e))
        log(f"least squares failed: {e}")

    ridge_grid = {}
    for lam in RIDGE_LAMBDAS:
        w, loss = ridge_regression(y_tr.astype(float), tx_tr, lam)
        s_tr, s_val = tx_tr @ w, tx_val @ w
        ridge_grid[str(lam)] = evaluate(y_tr, s_tr, y_val, s_val)
    best_lam = max(RIDGE_LAMBDAS, key=lambda l: ridge_grid[str(l)]["train"]["f1"])
    w, loss = ridge_regression(y_tr.astype(float), tx_tr, best_lam)
    record("ridge_regression", w, float(loss), dict(lambda_=best_lam, grid=ridge_grid))

    gamma_gd = 1.0 / eig[-1]
    t0 = time.time()
    w, loss = mean_squared_error_gd(y_tr.astype(float), tx_tr, np.zeros(tx_tr.shape[1]),
                                    REF["max_iters"], gamma_gd)
    record("mean_squared_error_gd", w, float(loss),
           dict(gamma=float(gamma_gd), max_iters=REF["max_iters"], seconds=time.time() - t0))

    random.seed(SEED)
    t0 = time.time()
    w, loss = mean_squared_error_sgd(y_tr.astype(float), tx_tr, np.zeros(tx_tr.shape[1]),
                                     len(y_tr), SGD_GAMMA)
    record("mean_squared_error_sgd", w, float(loss),
           dict(gamma=SGD_GAMMA, max_iters=len(y_tr), seconds=time.time() - t0))

    t0 = time.time()
    w, loss = logistic_regression(y_tr, tx_tr, np.zeros(tx_tr.shape[1]),
                                  REF["max_iters"], REF["gamma"])
    record("logistic_regression", w, float(loss),
           dict(gamma=REF["gamma"], max_iters=REF["max_iters"], seconds=time.time() - t0),
           scores=lambda tx, w: sigmoid(tx @ w))

    t0 = time.time()
    w_ref, loss = train_logistic(y_tr, tx_tr, **REF)
    s_tr_ref, s_val_ref = record(
        "reg_logistic_regression", w_ref, loss, dict(**REF, seconds=time.time() - t0),
        scores=lambda tx, w: sigmoid(tx @ w))
    ref = models["reg_logistic_regression"]

    # Same model with the default threshold 0.5 (no threshold tuning).
    models["reg_logistic_regression_t05"] = dict(
        train=metrics_at(y_tr, s_tr_ref, 0.5), val=metrics_at(y_val, s_val_ref, 0.5))

    # Threshold curve on the validation set for the figure.
    out["threshold_curve"] = [
        dict(t=round(float(t), 3), **{k: v for k, v in metrics_at(y_val, s_val_ref, t).items()
                                      if k != "threshold"})
        for t in np.arange(0.02, 0.801, 0.02)
    ]

    # Largest coefficients (features are standardised or 0/1 dummies).
    order = np.argsort(-np.abs(w_ref[1:]))[:25]
    out["top_coefficients"] = [[names[i], float(w_ref[1 + i])] for i in order]

    t0 = time.time()
    w_sel, loss = train_logistic(y_tr, tx_tr, **SELECTED)
    record("reg_logistic_regression_selected", w_sel, loss,
           dict(**SELECTED, seconds=time.time() - t0),
           scores=lambda tx, w: sigmoid(tx @ w))

    out["models"] = models
    save("holdout", out)
    del tx_tr, tx_val, gram

    # ---------------- ablation ----------------
    pred_ref = s_val_ref >= ref["train"]["threshold"]
    ablation = dict(full=dict(n_features=out["n_features_full"], val=ref["val"],
                              val_oracle=ref["val_oracle"], train=ref["train"]))

    def ablate(name, F_tr, F_val):
        tx_a, tx_b = add_bias(F_tr), add_bias(F_val)
        t0 = time.time()
        w, loss = train_logistic(y_tr, tx_a, **REF)
        s_tr, s_val = sigmoid(tx_a @ w), sigmoid(tx_b @ w)
        res = evaluate(y_tr, s_tr, y_val, s_val)
        res.update(n_features=tx_a.shape[1] - 1, loss=loss, seconds=time.time() - t0)
        pred = s_val >= res["train"]["threshold"]
        res["delta_f1_ci"] = paired_bootstrap(y_val, pred_ref, pred)
        ablation[name] = res
        log(f"ablation {name:14s} {res['n_features']:4d} cols  val F1 {res['val']['f1']:.4f} "
            f"delta CI {res['delta_f1_ci'][0]:+.4f}..{res['delta_f1_ci'][1]:+.4f}")
        out["ablation"] = ablation
        save("holdout", out)

    for variant in ["v1", "no_codes", "no_onehot", "no_indicator", "keep_meta", "keep_sparse"]:
        F_tr, F_val, _ = run_full_pipeline(X_tr, y_tr, X_val, y_val, variant)
        ablate(variant, F_tr, F_val)
        del F_tr, F_val

    F_tr, F_val = run_minimal_pipeline(X_tr, X_val)
    ablate("minimal", F_tr, F_val)

    ablation["threshold_05"] = dict(
        val=models["reg_logistic_regression_t05"]["val"],
        delta_f1_ci=paired_bootstrap(y_val, pred_ref, s_val_ref >= 0.5))
    out["ablation"] = ablation
    save("holdout", out)


# ----------------------------------------------------------------------
# Part 2: cross-validation
# ----------------------------------------------------------------------

def run_cv(x, y, include_selected=True):
    y01 = (y == 1).astype(int)
    np.random.seed(SEED)
    folds = np.array_split(np.random.permutation(len(y01)), N_FOLDS)
    configs = ["minimal_reglog", "full_ridge", "full_reglog"]
    if include_selected:
        configs.append("full_reglog_selected")
    cv = {c: [] for c in configs}

    for k in range(N_FOLDS):
        val_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        X_tr, X_val, y_tr, y_val = x[tr_idx], x[val_idx], y01[tr_idx], y01[val_idx]

        M_tr, M_val = run_minimal_pipeline(X_tr, X_val)
        tx_a, tx_b = add_bias(M_tr), add_bias(M_val)
        del M_tr, M_val
        w, _ = train_logistic(y_tr, tx_a, **REF)
        cv["minimal_reglog"].append(evaluate(y_tr, sigmoid(tx_a @ w), y_val, sigmoid(tx_b @ w)))
        del tx_a, tx_b

        F_tr, F_val, _ = run_full_pipeline(X_tr, y_tr, X_val, y_val)
        tx_a, tx_b = add_bias(F_tr), add_bias(F_val)
        del F_tr, F_val

        w, _ = ridge_regression(y_tr.astype(float), tx_a, 1e-4)
        cv["full_ridge"].append(evaluate(y_tr, tx_a @ w, y_val, tx_b @ w))

        w, _ = train_logistic(y_tr, tx_a, **REF)
        cv["full_reglog"].append(evaluate(y_tr, sigmoid(tx_a @ w), y_val, sigmoid(tx_b @ w)))

        if include_selected:
            w, _ = train_logistic(y_tr, tx_a, **SELECTED)
            cv["full_reglog_selected"].append(
                evaluate(y_tr, sigmoid(tx_a @ w), y_val, sigmoid(tx_b @ w)))
        del tx_a, tx_b

        log(f"fold {k + 1}/{N_FOLDS}: " + ", ".join(
            f"{c} {cv[c][-1]['val']['f1']:.4f}" for c in configs))

        summary = {}
        for c in configs:
            for metric in ["f1", "accuracy", "precision", "recall"]:
                vals = [r["val"][metric] for r in cv[c]]
                summary[f"{c}_{metric}_mean"] = float(np.mean(vals))
                summary[f"{c}_{metric}_std"] = float(np.std(vals))
        save("cv", dict(n_folds=N_FOLDS, folds_done=k + 1, folds=cv, summary=summary))


def main():
    args = sys.argv[1:]
    cache = None
    if "--cache" in args:
        cache = args[args.index("--cache") + 1]
    x, y = load_data(cache)
    log(f"data {x.shape}")
    if "holdout" in args:
        run_holdout(x, y)
    if "cv" in args:
        run_cv(x, y, include_selected="--no-selected" not in args)


if __name__ == "__main__":
    main()
