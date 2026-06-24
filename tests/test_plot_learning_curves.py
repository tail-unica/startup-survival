import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from src.utils import plot_learning_curves

def _store():
    return {
        ("rf", "window", 1000): {"F1": 0.50, "AUC": 0.60},
        ("rf", "window", 2000): {"F1": 0.55, "AUC": 0.65},
        ("lgb", "window", 1000): {"F1": 0.52, "AUC": 0.62},
        ("lgb", "window", 2000): {"F1": 0.58, "AUC": 0.66},
        ("rf", "nowindow", 1000): {"F1": 0.40, "AUC": 0.50},  # tag diverso: escluso
    }

def test_returns_figure_for_single_metric():
    fig = plot_learning_curves(_store(), "window", metrics="F1")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # una linea per modello (rf, lgb)

def test_one_line_per_model_sorted_by_train_size():
    fig = plot_learning_curves(_store(), "window", metrics="F1")
    ax = fig.axes[0]
    xs = ax.lines[0].get_xdata()
    assert list(xs) == sorted(xs)  # ordinate per train_size

def test_excludes_other_tags():
    fig = plot_learning_curves(_store(), "window", metrics="F1")
    ax = fig.axes[0]
    # solo train_size 1000 e 2000 (nowindow escluso)
    for line in ax.lines:
        assert set(line.get_xdata()) == {1000, 2000}
