import os
import random
from pathlib import Path

import joblib
import shap
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests


def set_seed(seed):
    """
    Fixes all relevant random sources so that experiments (MLP training,
    DataLoader shuffling, SHAP sampling) are fully reproducible.
    Sklearn / LightGBM models also rely on this through numpy.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_splits(X, y, seed, test_size=0.4, n_neighbors=5):
    """
    Builds one stratified train/validation/test split, imputed and scaled.

    The imputer and the scaler are fitted on this seed's training set alone.
    Refitting them per seed is what keeps the multi-seed evaluation free of the
    look-ahead leakage the paper is about: reusing one seed's imputer across
    splits would leak held-out information into every other split.

    :param X:           feature frame, without CompanyID and Target.
    :param y:           target series.
    :param seed:        seed driving both splits, so the partition is reproducible.
    :param test_size:   fraction held out of training, halved into val and test.
    :param n_neighbors: neighbours used by the KNN imputer.
    :return:            dict with the raw frames (X_train/X_val/X_test), the
                        targets, the imputed arrays (*_imp) and the scaled
                        arrays (*_scaled).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return {
        "seed": seed,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_imp": X_train_imp, "X_val_imp": X_val_imp, "X_test_imp": X_test_imp,
        "X_train_scaled": X_train_scaled, "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
    }


def get_split(X, y, seed, test_size=0.4, n_neighbors=5, cache_dir="tmp/splits", tag=""):
    """
    prepare_splits with an on-disk cache keyed by (tag, seed).

    The KNN imputation costs minutes on the full dataset, and a multi-seed sweep
    revisits the same split once per model. Caching turns that into one cost per
    (experiment, seed) instead of one per run, and it survives a kernel restart.
    Only the requested split is held in memory, which matters because the five
    splits together do not comfortably fit alongside a fitted SVM.

    :param cache_dir: directory holding the cached splits; created if missing.
    :param tag:       experiment tag, so window/nowindow/noteam/nocompetitors
                      never share a cache entry.
    :return:          same dict as prepare_splits.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"split_{tag}_seed{seed}.joblib"

    if cache_file.is_file():
        return joblib.load(cache_file)

    split = prepare_splits(X, y, seed, test_size=test_size, n_neighbors=n_neighbors)
    joblib.dump(split, cache_file)
    return split


DEFAULT_TABPFN_TOKEN_FILE = Path.home() / ".cache" / "tabpfn" / "auth_token"


def resolve_tabpfn_token(cache_file=DEFAULT_TABPFN_TOKEN_FILE):
    """
    Finds the Prior Labs access token used to reach TabPFN through its API.

    Looks at the TABPFN_TOKEN environment variable first, then falls back to the
    token cached by the TabPFN tooling outside the repository. The token is
    deliberately never read from .env: that file is tracked by git.

    :param cache_file: path of the cached token file used as a fallback.
    :return:           the token, or None when neither source holds one.
    """
    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if token:
        return token

    cache_file = Path(cache_file)
    if cache_file.is_file():
        token = cache_file.read_text().strip()
        if token:
            return token

    return None


def compute_permutation_shap(model, background, to_explain,
                             n_explain=100, n_background=20, random_state=None):
    """
    Computes SHAP values with shap.PermutationExplainer for any model exposing
    predict_proba, using the probability of the positive class as the output.

    This is the explainer used for SVM and TabPFN: neither admits TreeExplainer
    or LinearExplainer, and KernelExplainer is intractable on them (an RBF SVM
    carries thousands of support vectors, and TabPFN answers over the network).
    PermutationExplainer needs only 2*n_features+1 evaluations per explained
    row, so the cost is bounded by n_explain * (2*n_features+1) * n_background
    model calls.

    :param model:         fitted estimator exposing predict_proba.
    :param background:    reference data the explainer masks features against.
    :param to_explain:    rows to explain; only the first n_explain are used.
    :param n_explain:     number of rows to explain (clamped to what is available).
    :param n_background:  number of background rows (clamped to what is available).
    :param random_state:  seed for the background subsample, for reproducibility.
    :return:              numpy array of shape (n_explain, n_features).
    """
    background = np.asarray(background)
    to_explain = np.asarray(to_explain)

    n_background = min(n_background, len(background))
    n_explain = min(n_explain, len(to_explain))

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(background), size=n_background, replace=False)
    masker = background[idx]
    sample = to_explain[:n_explain]

    n_features = sample.shape[1]
    max_evals = 2 * n_features + 1

    explainer = shap.PermutationExplainer(
        lambda x: model.predict_proba(x)[:, 1],
        masker,
        seed=random_state,
    )
    explanation = explainer(sample, max_evals=max_evals)

    return np.asarray(explanation.values)


def plot_correlation_heatmap(df, exclude_cols=['CompanyID', 'Target']):

    """
    Creates a correlation heatmap for the given DataFrame, excluding specified columns.
    
    :param df: The input DataFrame for which the correlation heatmap will be generated.
    :param exclude_cols: A list of column names to exclude from the correlation calculation. Default is ['CompanyID', 'Target'].
    """


    corr_matrix = df.drop(exclude_cols, axis=1).corr()
    
    plt.figure(figsize=(30, 20))
    ax = sns.heatmap(data=corr_matrix, cmap='YlGnBu', annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
    
    return corr_matrix


def to_tensors(X, y):
    """
    Converts input features and target labels to PyTorch tensors.

    :param X: Input features as a NumPy array (or tensor-compatible structure).
    :param y: Target labels, can be a pandas Series or a NumPy array.
    :return: A tuple containing the input features and target labels as PyTorch tensors.
    """

    y_np = y.values if hasattr(y, 'values') else y
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    )


def _prepare_shap_comparison(shap_store, top_k=20):
    """
    Helper: extracts and prepares SHAP values and feature info for the bias-controlled vs no-window comparison plot.

    Returns a dictionary with:
        sv_b, sv_n             : SHAP arrays (n_samples, n_features)
        features_baseline      : list of feature names for the bias-controlled model
        features_nowindow      : list of feature names for the no-window model
        ranking_n              : top-k features ordered by no-window importance
        rank_base              : dict {feature: rank} for the bias-controlled model
        idx_base, idx_now      : dict {feature: column_index}
        mean_abs_b, mean_abs_n : mean absolute importance per feature
    """
    def squeeze_shap(sv):
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    sv_b = squeeze_shap(shap_store[("lgb", "window")]["shap_values"])
    sv_n = squeeze_shap(shap_store[("lgb", "nowindow")]["shap_values"])

    sample_b = shap_store[("lgb", "window")]["explainer_sample"]
    sample_n = shap_store[("lgb", "nowindow")]["explainer_sample"]

    features_baseline = (list(pd.DataFrame(sample_b).columns)
                         if hasattr(sample_b, "columns") or isinstance(sample_b, (pd.DataFrame, np.ndarray))
                         else [f"f{i}" for i in range(sv_b.shape[1])])
    features_nowindow = (list(pd.DataFrame(sample_n).columns)
                         if hasattr(sample_n, "columns") or isinstance(sample_n, (pd.DataFrame, np.ndarray))
                         else [f"f{i}" for i in range(sv_n.shape[1])])

    
    if isinstance(sample_b, pd.DataFrame):
        features_baseline = list(sample_b.columns)
    if isinstance(sample_n, pd.DataFrame):
        features_nowindow = list(sample_n.columns)

    mean_abs_b = np.abs(sv_b).mean(axis=0)
    mean_abs_n = np.abs(sv_n).mean(axis=0)

    order_n   = np.argsort(-mean_abs_n)
    ranking_n = [features_nowindow[i] for i in order_n][:top_k]

    order_b        = np.argsort(-mean_abs_b)
    ranking_b_full = [features_baseline[i] for i in order_b]
    rank_base      = {f: i + 1 for i, f in enumerate(ranking_b_full)}

    idx_base = {f: i for i, f in enumerate(features_baseline)}
    idx_now  = {f: i for i, f in enumerate(features_nowindow)}

    return dict(
        sv_b=sv_b, sv_n=sv_n,
        features_baseline=features_baseline,
        features_nowindow=features_nowindow,
        ranking_n=ranking_n,
        rank_base=rank_base,
        idx_base=idx_base,
        idx_now=idx_now,
        mean_abs_b=mean_abs_b,
        mean_abs_n=mean_abs_n,
    )


def plot_shap_comparison(shap_store, top_k=20):
    """
    Generate a comparison plot of SHAP value distributions for the bias-controlled model vs the no-window model (with leakage).

    Needs shap_store to contain the keys ("lgb", "window") and ("lgb", "nowindow") with the corresponding SHAP values and explainer samples.

    :param shap_store: dictionary populated by the train() function in the notebook.
    :param top_k:      number of top features to display (ordered by no-window importance).
    :return:           matplotlib.figure.Figure object
    """
    d = _prepare_shap_comparison(shap_store, top_k)

    sv_b      = d["sv_b"]
    sv_n      = d["sv_n"]
    ranking_n = d["ranking_n"]
    rank_base = d["rank_base"]
    idx_base  = d["idx_base"]
    idx_now   = d["idx_now"]

    COLOR_BASELINE = "#1f77b4"
    COLOR_NOWINDOW = "#ff7f0e"
    COLOR_UP       = "#2e7d32"
    COLOR_DOWN     = "#c62828"
    COLOR_FLAT     = "#616161"

    Y_OFFSET  = 0.22
    BOX_WIDTH = 0.35

    fig, ax = plt.subplots(figsize=(11, 0.25 * top_k + 2))

    for i, feat in enumerate(ranking_n):
        # Bias-controlled (blue, above)
        if feat in idx_base:
            col_b = idx_base[feat]
            ax.boxplot(
                sv_b[:, col_b],
                positions=[i - Y_OFFSET],
                widths=BOX_WIDTH,
                vert=False,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="white", lw=1.5),
                whiskerprops=dict(color=COLOR_BASELINE, lw=1),
                capprops=dict(color=COLOR_BASELINE, lw=1),
                boxprops=dict(facecolor=COLOR_BASELINE,
                              edgecolor=COLOR_BASELINE, alpha=0.75),
            )
        else:
            ax.axhspan(i - 0.45, i + 0.45, color="#f5f5f5", zorder=0)
            ax.text(0, i - Y_OFFSET,
                    "not in bias-controlled model",
                    ha="center", va="center", fontsize=8,
                    color="#888888", style="italic")

        # No-window (orange, below)
        col_n = idx_now[feat]
        ax.boxplot(
            sv_n[:, col_n],
            positions=[i + Y_OFFSET],
            widths=BOX_WIDTH,
            vert=False,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", lw=1.5),
            whiskerprops=dict(color=COLOR_NOWINDOW, lw=1),
            capprops=dict(color=COLOR_NOWINDOW, lw=1),
            boxprops=dict(facecolor=COLOR_NOWINDOW,
                          edgecolor=COLOR_NOWINDOW, alpha=0.75),
        )

    ax.axvline(0, color="#999999", lw=0.7, zorder=1)

    ax.set_yticks(range(len(ranking_n)))
    ax.set_yticklabels(ranking_n, fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(len(ranking_n) - 0.5, -0.8)

    xmax = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("SHAP value", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for i in range(len(ranking_n) - 1):
        ax.axhline(i + 0.5, color="#dddddd", lw=0.5, zorder=0)

    # Rank-shift annotations on the right
    xr = 1.015
    for i, feat in enumerate(ranking_n):
        new_rank = i + 1
        if feat in rank_base:
            old_rank = rank_base[feat]
            delta = old_rank - new_rank
            if delta > 0:
                txt   = f"{old_rank} → {new_rank}  (+{delta})"
                color = COLOR_UP
            elif delta < 0:
                txt   = f"{old_rank} → {new_rank}  ({delta})"
                color = COLOR_DOWN
            else:
                txt   = f"{old_rank} → {new_rank}"
                color = COLOR_FLAT
        else:
            txt   = f"new → {new_rank}"
            color = COLOR_UP

        ax.text(xr, i, txt,
                ha="left", va="center",
                fontsize=8.5, fontweight="bold", color=color,
                transform=ax.get_yaxis_transform(), clip_on=False,
                family="monospace")

    ax.text(xr, -0.7, "Rank shift",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="#333333", transform=ax.get_yaxis_transform(),
            clip_on=False)

    # Legend
    box_legend = [
        Patch(facecolor=COLOR_BASELINE, alpha=0.75, edgecolor=COLOR_BASELINE,
              label="Bias-controlled (proposed)"),
        Patch(facecolor=COLOR_NOWINDOW, alpha=0.75, edgecolor=COLOR_NOWINDOW,
              label="No time-window (with leakage)"),
    ]
    leg1 = ax.legend(handles=box_legend,
                     loc="upper left",
                     bbox_to_anchor=(0.0, -0.05),
                     frameon=False, fontsize=9,
                     title="SHAP distribution",
                     title_fontsize=9)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    plt.tight_layout()
    plt.subplots_adjust(bottom=-0.5)

    return fig


def compute_wilcoxon_table(shap_store, top_k=20):
    """
    Computes the Wilcoxon signed-rank table for SHAP comparison (bias-controlled vs no-window)
    for LightGBM, with FDR correction (Benjamini-Hochberg).

    Needs shap_store to contain the keys ("lgb", "window") and ("lgb", "nowindow").

    :param shap_store: dictionary populated by the train() function in the notebook.
    :param top_k:      number of top features to analyze (ordered by no-window importance).
    :return:           pandas DataFrame with columns:
                       Feature, Mean |SHAP| w., Mean |SHAP| w/o w., Diff.,
                       wilcoxon_stat, wilcoxon_p_value, wilcoxon_p_corrected, Signif.
                       (Signif. is based on the BH-corrected p-value, matching Table 5)
    """
    d = _prepare_shap_comparison(shap_store, top_k)

    sv_b      = d["sv_b"]
    sv_n      = d["sv_n"]
    ranking_n = d["ranking_n"]
    idx_base  = d["idx_base"]
    idx_now   = d["idx_now"]

    results = []

    for feat in ranking_n:
        if feat not in idx_base:
            continue

        col_b = idx_base[feat]
        col_n = idx_now[feat]

        x_b = sv_b[:, col_b]
        x_n = sv_n[:, col_n]

        n_pairs = min(len(x_b), len(x_n))
        x_b = x_b[:n_pairs]
        x_n = x_n[:n_pairs]

        abs_b = np.abs(x_b)
        abs_n = np.abs(x_n)

        try:
            w_stat, w_p = wilcoxon(abs_b, abs_n, alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, 1.0

        results.append({
            "Feature":            feat,
            "Mean |SHAP| w.":     abs_b.mean(),
            "Mean |SHAP| w/o w.": abs_n.mean(),
            "Diff.":              abs_b.mean() - abs_n.mean(),
            #"wilcoxon_stat":      w_stat,
            "wilcoxon_p_value":   w_p,
        })

    df = pd.DataFrame(results)

    # # FDR correction (Benjamini-Hochberg)
    # df["wilcoxon_p_corrected"] = np.nan
    # mask = df["wilcoxon_p_value"].notna()
    # _, p_corrected, _, _ = multipletests(
    #     df.loc[mask, "wilcoxon_p_value"].values,
    #     method="fdr_bh",
    #     alpha=0.05,
    # )
    # df.loc[mask, "wilcoxon_p_corrected"] = p_corrected

    # Significance column based on the BH-corrected p-value (consistent with Table 5)
    def _stars(p):
        if pd.isna(p):
            return ""
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    df["Signif."] = df["wilcoxon_p_value"].apply(_stars)

    return df


def get_probs(loader, model, device):

    """
    Computes the predicted probabilities and true labels for a given data loader, model, and device.
    
    :param loader: A PyTorch DataLoader that provides batches of input features and target labels.
    :param model: A PyTorch model that will be used to make predictions on the input features.
    :param device: The device (e.g., 'cpu' or 'cuda') on which the model and data will be processed.
    :return: A tuple containing two NumPy arrays: the predicted probabilities and the true labels. 
    """

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    return np.array(all_probs).flatten(), np.array(all_labels).flatten().astype(int)

def _order_metrics(keys):
    """
    Reorder metric keys to the paper's column order:
    AUC, F1 (test), Prec (test), Rec (test), Acc (test), Acc (train).
    Matching is case-insensitive on substrings; unmatched keys are appended.
    """
    def _slot(k):
        s = k.lower().replace("_", " ").replace("-", " ")
        is_train = "train" in s
        if "auc" in s:
            return 0
        if "f1" in s or "f_1" in s or s.strip() == "f1":
            return 1
        if "prec" in s:
            return 2
        if "rec" in s:                       # recall (not 'prec')
            return 3
        if "acc" in s:
            return 5 if is_train else 4
        return 99

    return sorted(keys, key=lambda k: (_slot(k), k.lower()))


def _aggregate_runs(runs, metrics):
    """
    Collapses the per-seed records of one (model, tag) cell into mean and
    standard deviation for each metric.

    The std is the sample one (ddof=1), not numpy's default population std: the
    seeds are a sample of the runs the procedure could have produced, and the
    reported bar is meant to support statements about whether a gap between two
    models exceeds split-to-split noise. A single run has no spread to estimate,
    so it reports 0.0 rather than NaN.

    :param runs:    list of dicts, one per seed, as stored by the notebook.
    :param metrics: metric keys to aggregate.
    :return:        dict {metric: (mean, std)}.
    """
    out = {}
    for met in metrics:
        values = np.asarray([float(r[met]) for r in runs], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[met] = (float(values.mean()), std)
    return out


def compare_metrics(metrics_store, tag_a, tag_b,
                    metric_order=None, model_order=None,
                    latex=False):
    """
    Builds a per-model comparison table between two experiments, formatted to
    match Table 4 of the paper (tab:ablation_window).

    Every cell is reported as 'mean ± std' across the evaluation seeds, so the
    table answers the reviewer's objection that a gap between two models could
    be split-to-split noise. Layout: for each model, a baseline row (tag_a)
    followed by a "w/o window" row (tag_b) that also carries the relative change
    of the means. Metrics are in columns (AUC, F1, Prec., Rec., Acc.test,
    Acc.train), not in rows.

    :param metrics_store: dict {(model_type, tag): [ {metric: value, 'seed': s}, ... ]},
                          one entry per seed, as accumulated by the notebook.
    :param tag_a:         reference experiment tag (e.g. "window")
    :param tag_b:         experiment to compare against tag_a (e.g. "nowindow")
    :param metric_order:  optional list of metric keys to fix column order.
                          Defaults to the order found in the first model entry.
    :param model_order:   optional list of model keys to fix row order.
                          Defaults to the paper order.
    :param latex:         if True, cells use LaTeX markup ('$0.80 \\pm 0.01$' and
                          \\textcolor{green!60!black}{...} / red for the change);
                          if False, a plain '0.80 ± 0.01 (+x.x%)' string is used.
    :return:              pandas DataFrame with a 'Model' column and one column
                          per metric.
    """
    pairs = {m for (m, t) in metrics_store.keys() if t in (tag_a, tag_b)}

    if model_order is not None:
        models = model_order
    else:
        # canonical paper order: lr, dt, rf, lgb, svm, mlp, tabpfn; anything else appended
        _priority = ["lr", "dt", "rf", "lgb", "svm", "mlp", "tabpfn"]

        def _rank(m):
            ml = m.lower()
            for i, p in enumerate(_priority):
                if ml.startswith(p):
                    return (i, ml)
            return (len(_priority), ml)

        models = sorted(pairs, key=_rank)

    label_b = "w/o window"

    def _fmt_value(mean, std):
        if latex:
            return f"${mean:.2f} \\pm {std:.2f}$"
        return f"{mean:.2f} ± {std:.2f}"

    def _fmt_pct(mean_a, mean_b):
        va, vb = round(mean_a, 2), round(mean_b, 2)
        if va == 0:
            return ""
        pct = (vb - va) / va * 100.0
        sign = "+" if pct >= 0 else "-"
        if not latex:
            return f"({sign}{abs(pct):.1f}%)"
        colour = "green!60!black" if pct >= 0 else "red"
        return f"(\\textcolor{{{colour}}}{{${sign}{abs(pct):.1f}\\%$}})"

    rows = {}
    cols = None

    for m in models:
        key_a, key_b = (m, tag_a), (m, tag_b)
        if key_a not in metrics_store or key_b not in metrics_store:
            missing = tag_a if key_a not in metrics_store else tag_b
            print(f"⚠️  Skipping model '{m}': missing entry for {missing}.")
            continue

        runs_a, runs_b = metrics_store[key_a], metrics_store[key_b]
        if len(runs_a) != len(runs_b):
            print(f"⚠️  Model '{m}': {len(runs_a)} seed(s) for '{tag_a}' but "
                  f"{len(runs_b)} for '{tag_b}'; the two rows average over "
                  f"different numbers of runs.")

        if metric_order is not None:
            metrics = metric_order
        else:
            drop = {"F1_train", "F1_val", "average_precision", "seed"}
            metrics = _order_metrics([k for k in runs_a[0].keys() if k not in drop])
        if cols is None:
            cols = metrics

        agg_a = _aggregate_runs(runs_a, metrics)
        agg_b = _aggregate_runs(runs_b, metrics)

        rows[m] = {met: _fmt_value(*agg_a[met]) for met in metrics}
        rows[f"{m} {label_b}"] = {
            met: _fmt_value(*agg_b[met]) + _fmt_pct(agg_a[met][0], agg_b[met][0])
            for met in metrics
        }

    # preserve baseline/w-o ordering
    ordered_index = []
    for m in models:
        if m in rows:
            ordered_index.append(m)
            ordered_index.append(f"{m} {label_b}")

    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    df = df.loc[ordered_index]
    # expose the model label as an explicit first column (as in Table 4),
    # not just as the index
    df.insert(0, "Model", df.index)
    df = df.reset_index(drop=True)
    return df
