# CS-433 Project 1: Coronary Heart Disease Prediction

Mario Matsas, Ioanna Poulou, Abdullah Rehman

We predict coronary heart disease (MICHD) from the BRFSS 2015 survey with
linear models implemented from scratch in NumPy. The report
([report/report.tex](report/report.tex)) describes the method and results.

## Reproducing the submission

Requirements: Python 3 and NumPy (tested with Python 3.12 and NumPy 2.5).
No other libraries are used.

1. Put the three competition files in `dataset/` with their original names:
   `dataset/x_train.csv`, `dataset/y_train.csv`, `dataset/x_test.csv`.
2. From the repository root, run

   ```
   python run.py
   ```

This writes `submission.csv` (columns `Id,Prediction`, labels -1/1), the
file we uploaded to AIcrowd. 

`run.py` fits everything on the full labelled set:

- **Preprocessing:** the pipeline in cell 15 of `ex.ipynb` (summarised below).
- **Model:** regularised logistic regression (`reg_logistic_regression`),
  with lambda = 1e-4, gamma = 0.3 and 1000 gradient descent steps from w = 0.
- **Threshold:** a person is predicted positive when the predicted probability
  is at least t, where t maximises F1 on the training predictions.

## Files

| File | Contents |
| --- | --- |
| `implementations.py` | The six required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`, plus their helpers. |
| `helpers.py` | Course-provided CSV loading and submission writing. |
| `run.py` | Trains the final model and writes `submission.csv`. |
| `ex.ipynb` | Main notebook: exploratory analysis (cells 1–14), the full preprocessing pipeline (cell 15) and the hyperparameter and threshold search on an 80/20 split with seed 2026 (cell 17). |
| `part_2.ipynb` | Early data exploration. |
| `report/experiments.py` | All experiments in the report: comparison of the six methods, ablation of each preprocessing step with bootstrap confidence intervals, and 5-fold cross-validation. Results go to `report/results.json`. |
| `report/report.tex` | The report (IEEEtran template). |

The experiments are run from the repository root with
`python report/experiments.py holdout` (about 35 minutes) and
`python report/experiments.py cv` (about 90 minutes). They execute the
preprocessing cell of `ex.ipynb` directly instead of copying it, so the
notebook, the experiments and `run.py` share one implementation.

## Preprocessing

All statistics (codes, medians, means, standard deviations, categories) are
learned from the training data only.

1. **Dropped variables:** 43 fields chosen by hand (interview metadata and
   survey weights, redundant or almost empty variables, and raw answers that
   have decoded equivalents), plus every field with at least 99 % missing
   values.
2. **Non-response codes:** "don't know / refused" codes (77/99, 777/999 or
   7/9) are found per field from the training data and treated as missing,
   except where the codebook says the code is a real answer (for example
   income category 7).
3. **Encoding:** categorical fields (nominal, or at most ten levels) get k−1
   dummy variables with "missing" as its own category. Ordinal and continuous
   fields are median-imputed, standardised and given a missingness indicator.
   Fields that are constant whenever observed keep only their missingness
   indicator.

This gives 836 features.
