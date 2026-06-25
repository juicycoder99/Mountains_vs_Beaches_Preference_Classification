# Mountains vs. Beaches Preference Classification

Predicting whether a person prefers a **beach** or a **mountain** vacation from demographic and
lifestyle features, comparing three classifiers with cross-validation, handling overfitting, and
analysing the features that drive the preference.

- Notebook: [`mountains_vs_beaches_classification.ipynb`](mountains_vs_beaches_classification.ipynb)
- Report: [`report.md`](report.md)

## Approach

1. **Choice of classifiers** — Decision Tree (interpretable, gives feature importances), K-Nearest
   Neighbours (non-parametric baseline), and a Neural Network / MLP (flexible non-linear model).
2. **Assessment** — 10-fold stratified cross-validation plus held-out test metrics (accuracy,
   precision, recall, F1), confusion matrices, and an overfitting study of tree depth.
3. **Improvement** — grid search to tune KNN, depth-limiting the tree, and a feature-importance
   analysis.
4. **Conclusion** — what was achieved, the dataset's drawbacks (class imbalance, strong
   separability), and possible improvements.

## Results

| Model | CV accuracy | Test accuracy | Test F1 (mountain) |
|-------|------------:|--------------:|-------------------:|
| Decision Tree | 0.995 | 0.996 | 0.992 |
| KNN (default) | 0.923 | 0.920 | 0.831 |
| KNN (tuned) | — | 0.939 | 0.865 |
| Neural Net (MLP) | 0.998 | 0.998 | 0.996 |

The most influential features are proximity to mountains/beaches and the preferred activity.

## Dataset

`mountains_vs_beaches_preferences.csv` (52,444 rows, 13 features + target), originally from the
Kaggle "Mountains vs. Beaches Preferences" dataset. The only transformation is one-hot encoding of
the categorical columns and standardisation of the numeric columns, fit on the training split.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook mountains_vs_beaches_classification.ipynb
```

## Files

| File | Description |
|------|-------------|
| `mountains_vs_beaches_classification.ipynb` | Full implementation and analysis (Steps 1–4) |
| `report.md` | Written report |
| `mountains_vs_beaches_preferences.csv` | Dataset |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
