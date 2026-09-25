"""One sweep run: build a model, fit it, log its metrics, explain it.

Everything that differs between the seven families lives in its class under
src/models/, so what is here is what they have in common and it is also what
the paper's tables and the W&B dashboards read.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import wandb
import wandb.sklearn  # wandb does not import this submodule on its own
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from src.experiments import grid_runs
from src.models import build
from src.utils import get_split, save_figure, set_seed


def _roc_figure(labels, probs, model_type):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - {model_type.upper()}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig, roc_auc


def _pr_figure(labels, probs, model_type):
    prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_curve, prec_curve, color="green", lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall - {model_type.upper()}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig, ap


def run_once(
    params,
    tag,
    X,
    y,
    config,
    split_kwargs,
    metrics_store,
    shap_store,
    *,
    use_wandb=True,
):
    """Train, score and explain one model on one seed.

    It is the body of a sweep run, and it is the same whether a sweep drives it or
    the command line does. With ``use_wandb`` off nothing is sent anywhere: the
    metrics are printed and kept in the stores, which is what the notebook and the
    command line read to build their tables.

    :param params: The parameters of this run, which have to carry ``model_type``
        and ``seed`` and the hyperparameters of that family.
    :param tag: The experiment the run belongs to; keys both stores.
    :param X: Features of the selected experiment.
    :param y: Its target.
    :param config: The parsed ``config.yaml``.
    :param split_kwargs: Passed through to ``get_split``.
    :param metrics_store: Dict the run appends its metrics to, keyed by
        ``(model_type, tag)``.
    :param shap_store: The same, for the SHAP values of the explained seed.
    :param use_wandb: Whether to log to Weights & Biases.
    :return: The metrics of the run.
    """
    model_type = params.model_type if hasattr(params, "model_type") else params["model_type"]
    wandb_config = params if hasattr(params, "model_type") else SimpleNamespace(**params)

    # Each run draws its own seed. It drives BOTH the split and the model's
    # randomness, so the five runs of a model are five independent replications,
    # not five reruns of one partition, which is what makes the mean +- std in
    # the result tables able to separate a real gap between models from
    # split-to-split noise.
    seed = wandb_config.seed
    split = get_split(X, y, seed, tag=tag, **split_kwargs)

    # Reproducibility: seed Python, NumPy, PyTorch (CPU+CUDA), cuDNN. The
    # estimators also get a random_state of their own.
    set_seed(seed)

    model = build(model_type, wandb_config, seed)
    X_train, X_val, X_test = model.matrices(split)
    y_train, y_val, y_test = split["y_train"], split["y_val"], split["y_test"]

    model.fit(X_train, y_train, X_val, y_val)

    _, preds_train = model.score(X_train)
    _, preds_val = model.score(X_val)
    probs_test, preds_test = model.score(X_test)

    metrics = {
        "accuracy_train": accuracy_score(y_train, preds_train),
        "F1_train": f1_score(y_train, preds_train, zero_division=0),
        "F1_val": f1_score(y_val, preds_val, zero_division=0),
        "accuracy": accuracy_score(y_test, preds_test),
        "F1": f1_score(y_test, preds_test, zero_division=0),
        "precision": precision_score(y_test, preds_test, zero_division=0),
        "recall": recall_score(y_test, preds_test, zero_division=0),
    }
    fig_roc, roc_auc = _roc_figure(y_test, probs_test, model_type)
    fig_pr, ap = _pr_figure(y_test, probs_test, model_type)
    metrics["AUC"] = roc_auc
    metrics["AP"] = ap
    # A random classifier scores an AP equal to the prevalence of the positive
    # class, which moves with the definition of the target: the lift puts AP on
    # that baseline. The prevalence itself is context, kept out of the tables.
    prevalence = float(np.mean(y_test))
    metrics["AP_lift"] = ap / prevalence if prevalence > 0 else float("nan")
    metrics["prevalence"] = prevalence

    if use_wandb:
        wandb.log({**metrics, "roc_curve": wandb.Image(fig_roc), "pr_curve": wandb.Image(fig_pr)})
        wandb.sklearn.plot_confusion_matrix(y_test, preds_test, ["Neg", "Pos"])
    plt.close(fig_roc)
    plt.close(fig_pr)

    print(
        f"{model_type} seed {seed} | AUC: {roc_auc:.3f} | AP: {ap:.3f} | "
        f"AP lift: {metrics['AP_lift']:.3f} (prevalence {prevalence:.3f})"
    )
    print(classification_report(y_test, preds_test))

    # Persist the metrics for the later cross-experiment comparison, in parallel
    # with the SHAP store.
    if tag is not None:
        metrics_store.setdefault((model_type, tag), []).append({"seed": seed, **metrics})

    # SHAP is computed on the first evaluation seed only. Those tables answer a
    # different question from the mean +- std ones: which features move when the
    # leakage is removed, not how much the metrics vary across splits. The Wilcoxon
    # table also pairs rows within one explained sample, so pooling five seeds
    # would change what the test measures. It additionally keeps TabPFN to one
    # SHAP pass per experiment instead of five, which its inference cost notices.
    if seed == config["shap_seed"]:
        # Re-seed before SHAP so that the stochastic estimators give the same
        # values across runs.
        set_seed(seed)
        shap_cfg = config["shap_permutation"].get(model_type, {})
        shap_values, explainer_sample = model.explain(split, shap_cfg)

        fig_shap = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, explainer_sample, feature_names=X.columns.tolist(), show=False
        )
        # One file per (model, experiment): a re-run overwrites its own figure.
        path = save_figure(fig_shap, f"shap_summary_{model_type}_{tag or 'untagged'}", config)
        print(f"SHAP summary plot saved to {path}")
        if use_wandb:
            wandb.log({"shap_summary_plot": wandb.Image(plt)})
            plt.close()
        else:
            plt.show()

        if tag is not None:
            # Four families explain a bare array: the column names go back on
            # here, or the comparison would match features by position, and an
            # ablation shifts every position after the columns it removes.
            if not isinstance(explainer_sample, pd.DataFrame):
                explainer_sample = pd.DataFrame(explainer_sample, columns=X.columns)
            shap_store[(model_type, tag)] = {
                "shap_values": np.asarray(shap_values),
                "explainer_sample": explainer_sample,
            }
    return metrics


def run_grid(tag, X, y, config, split_kwargs, metrics_store, shap_store):
    """Run every combination the sweep declares, without a sweep.

    This is the route the command line takes with Weights & Biases off, and the
    one a notebook can take to see the numbers without an account: the runs are
    the same, only nothing is logged remotely.

    :param tag: The experiment the runs belong to.
    :param X: Features of the selected experiment.
    :param y: Its target.
    :param config: The parsed ``config.yaml``.
    :param split_kwargs: Passed through to ``get_split``.
    :param metrics_store: Dict the runs append their metrics to.
    :param shap_store: The same, for the SHAP values.
    :return: How many runs were executed.
    """
    runs = grid_runs(config)
    for i, params in enumerate(runs, start=1):
        print(f"\n── run {i}/{len(runs)} ─────────────────────────────────────────")
        run_once(
            params,
            tag,
            X,
            y,
            config,
            split_kwargs,
            metrics_store,
            shap_store,
            use_wandb=False,
        )
    return len(runs)


def make_train(tag, X, y, config, split_kwargs, metrics_store, shap_store):
    """Build the function ``wandb.agent`` calls once per sweep run.

    :param tag:           the experiment the run belongs to (``controlled``,
                          ``leakboth``, ``noteam``, ``nocompetitors``,
                          ``leaklabel``, ``leakfeat``). Keys both stores.
    :param X, y:          features and target of the selected experiment.
    :param config:        the parsed config.yaml.
    :param split_kwargs:  passed through to ``get_split``.
    :param metrics_store: dict the run appends its metrics to, keyed by
                          ``(model_type, tag)``. Owned by the caller so that it
                          survives a module reload and accumulates across the
                          experiments the comparison tables need.
    :param shap_store:    same, for the SHAP values of the explained seed.
    """

    def train():
        with wandb.init():
            run_once(
                wandb.config,
                tag,
                X,
                y,
                config,
                split_kwargs,
                metrics_store,
                shap_store,
                use_wandb=True,
            )

    return train
