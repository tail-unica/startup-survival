# Learning Curves via Sweep Parameter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere `train_size` come parametro dello sweep wandb e sottocampionare il train per osservare le learning curve di tutte le metriche per ogni modello.

**Architecture:** Un helper puro in `src/utils.py` produce sottocampioni del train pool stratificati, annidati e deterministici. Il notebook aggiunge `train_size` allo sweep (`config.yaml`), sottocampiona il train dentro `make_train`, logga le metriche su wandb e le salva in `learning_curve_store`. Un secondo helper + una cella dedicata costruiscono i chart delle curve (una linea/colore per modello).

**Tech Stack:** Python, numpy, scikit-learn, wandb, matplotlib, Jupyter notebook, pytest.

## Global Constraints

- Seed di riproducibilità: `config['random_seed']` (= 12). Tutto il sottocampionamento deve essere deterministico con questo seed.
- Opzione A: **val e test restano fissi**; si sottocampiona solo il train.
- `metrics_store` e le celle di confronto cross-experiment (35–41) **non vanno modificate**.
- Imputer e scaler restano fittati una sola volta sul train pool completo (cella 25).
- File helper di riferimento: `src/utils.py` (già importa `numpy as np` e `matplotlib.pyplot as plt` in testa).
- Notebook: `notebook.ipynb`. Modifiche alle celle via NotebookEdit; non rinumerare le altre celle.

---

### Task 1: Helper `make_nested_subsampler`

**Files:**
- Modify: `src/utils.py` (aggiunta funzione in coda)
- Test: `tests/test_subsampler.py`
- Modify: `.venv` (installazione pytest, una tantum)

**Interfaces:**
- Produces: `make_nested_subsampler(y, seed) -> subsample`, dove `subsample(k: int) -> np.ndarray` restituisce indici posizionali interi ordinati (stratificati, annidati, deterministici).

- [ ] **Step 1: Installa pytest nel venv** (una tantum)

Run: `.venv/bin/pip install pytest`
Expected: installazione completata (`Successfully installed pytest-...`).

- [ ] **Step 2: Write the failing test**

Create `tests/test_subsampler.py`:

```python
import numpy as np
from src.utils import make_nested_subsampler

def _make_y(n_pos, n_neg):
    return np.array([1] * n_pos + [0] * n_neg)

def test_returns_requested_size_approximately():
    y = _make_y(300, 700)  # 1000, 30% positivi
    sub = make_nested_subsampler(y, seed=12)
    idx = sub(100)
    assert abs(len(idx) - 100) <= 1

def test_stratified_proportions_preserved():
    y = _make_y(300, 700)  # 30% positivi
    sub = make_nested_subsampler(y, seed=12)
    idx = sub(200)
    frac_pos = y[idx].mean()
    assert abs(frac_pos - 0.3) < 0.03

def test_nested_subsamples():
    y = _make_y(300, 700)
    sub = make_nested_subsampler(y, seed=12)
    small = set(sub(100).tolist())
    big = set(sub(400).tolist())
    assert small.issubset(big)

def test_deterministic_with_same_seed():
    y = _make_y(300, 700)
    a = make_nested_subsampler(y, seed=12)(250)
    b = make_nested_subsampler(y, seed=12)(250)
    assert np.array_equal(a, b)

def test_caps_at_pool_size():
    y = _make_y(300, 700)
    sub = make_nested_subsampler(y, seed=12)
    idx = sub(100000)  # oltre il pool
    assert len(idx) == 1000
    assert sorted(idx.tolist()) == list(range(1000))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_subsampler.py -v`
Expected: FAIL con `ImportError`/`AttributeError` (`make_nested_subsampler` non esiste).

- [ ] **Step 4: Write minimal implementation**

In coda a `src/utils.py`:

```python
def make_nested_subsampler(y, seed):
    """Restituisce subsample(k) -> indici posizionali del train pool.

    I sottocampioni sono stratificati (rispettano le proporzioni di classe di y),
    annidati (sub(k1) e' sottoinsieme di sub(k2) per k1 < k2) e deterministici
    dato lo stesso seed. Usato per costruire le learning curve.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    fractions = counts / counts.sum()
    order = {c: rng.permutation(np.where(y == c)[0]) for c in classes}  # shuffle una volta

    def subsample(k):
        idx = []
        for c, frac in zip(classes, fractions):
            n_c = min(int(round(k * frac)), len(order[c]))
            idx.extend(order[c][:n_c])  # prefisso -> annidato
        return np.sort(np.array(idx, dtype=int))

    return subsample
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_subsampler.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/utils.py tests/test_subsampler.py
git commit -m "feat: add make_nested_subsampler for learning curves"
```

---

### Task 2: Helper `plot_learning_curves`

**Files:**
- Modify: `src/utils.py` (aggiunta funzione in coda)
- Test: `tests/test_plot_learning_curves.py`

**Interfaces:**
- Consumes: nulla dai task precedenti.
- Produces: `plot_learning_curves(store, tag, metrics=("F1", "AUC")) -> matplotlib.figure.Figure`. `store` ha chiavi `(model_type, tag, train_size)` e valori dict di metriche. `metrics` può essere una stringa singola o un iterabile. Una linea (colore distinto) per `model_type`, x = `train_size`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_learning_curves.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plot_learning_curves.py -v`
Expected: FAIL con `ImportError` (`plot_learning_curves` non esiste).

- [ ] **Step 3: Write minimal implementation**

In coda a `src/utils.py`:

```python
def plot_learning_curves(store, tag, metrics=("F1", "AUC")):
    """Costruisce i chart delle learning curve da learning_curve_store.

    store: dict con chiavi (model_type, tag, train_size) e valori dict di metriche.
    tag:   esperimento da filtrare (es. "window").
    metrics: stringa singola o iterabile di nomi metrica.
    Restituisce una Figure con un asse per metrica e una linea (colore distinto)
    per model_type, x = train_size.
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    keys = [(m, t, k) for (m, t, k) in store if t == tag]
    models = sorted({m for (m, _t, _k) in keys})
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for m in models:
            pts = sorted(((k, store[(m, tag, k)][metric]) for (mm, _t, k) in keys if mm == m),
                         key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", color=colors[m], label=m)
        ax.set_xlabel("train_size")
        ax.set_ylabel(metric)
        ax.set_title(f"Learning curve — {metric} ({tag})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_plot_learning_curves.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_plot_learning_curves.py
git commit -m "feat: add plot_learning_curves helper"
```

---

### Task 3: Parametro `train_size` nello sweep config

**Files:**
- Modify: `config/config.yaml` (blocco `sweep_settings.parameters`, dopo `thinking_metric`)

**Interfaces:**
- Produces: chiave `train_size` in `config['sweep_settings']['parameters']` con `values` = lista di 10 interi.

- [ ] **Step 1: Aggiungi il parametro**

In `config/config.yaml`, dentro `sweep_settings.parameters`, dopo il blocco `thinking_metric`, aggiungi:

```yaml
    # Learning curve
    train_size:
      values: [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
```

- [ ] **Step 2: Verifica che il YAML sia valido e il parametro presente**

Run:
```bash
.venv/bin/python -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['sweep_settings']['parameters']['train_size']['values'])"
```
Expected: `[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]`

- [ ] **Step 3: Commit**

```bash
git add config/config.yaml
git commit -m "feat: add train_size sweep parameter for learning curves"
```

---

### Task 4: Notebook — store, subsampler, sottocampione e logging

**Files:**
- Modify: `notebook.ipynb` (cella 1, cella 25, cella 29)

**Interfaces:**
- Consumes: `make_nested_subsampler` (Task 1), `train_size` config (Task 3).
- Produces: variabile globale `learning_curve_store` e `subsample_train`; metriche loggate su wandb con `train_size`/`train_size_actual`; `learning_curve_store[(model_type, tag, train_size)]` popolato.

- [ ] **Step 1: Cella 1 — importa l'helper e dichiara lo store**

Nella cella 1, nella riga di import da `src.utils`, aggiungi `make_nested_subsampler` e `plot_learning_curves`:

```python
from src.utils import plot_correlation_heatmap, to_tensors, get_probs, plot_shap_comparison, compute_wilcoxon_table, set_seed, compare_metrics, make_nested_subsampler, plot_learning_curves
```

E accanto a `shap_store = {}` / `metrics_store = {}` aggiungi:

```python
learning_curve_store = {}
```

- [ ] **Step 2: Cella 25 — istanzia il subsampler dopo lo scaling**

In fondo alla cella 25 (dopo la riga `print(f"Total: ...")`), aggiungi:

```python
# Subsampler annidato/stratificato del train pool per le learning curve.
subsample_train = make_nested_subsampler(y_train, config['random_seed'])
```

- [ ] **Step 3: Cella 29 — sottocampiona il train all'inizio di `train()`**

Subito dopo `set_seed(config['random_seed'])` (e prima del primo `if wandb_config.model_type == "rf":`), inserisci:

```python
            # --- LEARNING CURVE: sottocampione del train (val/test fissi) ---
            idx = subsample_train(wandb_config.train_size)
            Xtr_imp    = X_train_imp[idx]
            Xtr_scaled = X_train_scaled[idx]
            ytr        = y_train.iloc[idx].reset_index(drop=True)
```

- [ ] **Step 4: Cella 29 — sostituisci gli usi del training nei rami modello**

Nei rami modello sostituisci (solo i riferimenti al **training**, non val/test):

- ramo `rf`: `model.fit(X_train_imp, y_train)` → `model.fit(Xtr_imp, ytr)`; `preds_train = model.predict(X_train_imp)` → `model.predict(Xtr_imp)`; `probs_train = model.predict_proba(X_train_imp)[:, 1]` → `model.predict_proba(Xtr_imp)[:, 1]`; `labels_train = y_train` → `labels_train = ytr`.
- ramo `lgb`: `ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])` → `ratio = float(ytr.value_counts()[0] / ytr.value_counts()[1])`; `model.fit(X_train_imp, y_train)` → `model.fit(Xtr_imp, ytr)`; `preds_train`/`probs_train` su `Xtr_imp`; `labels_train = ytr`.
- ramo `dt`: `model.fit(X_train_imp, y_train)` → `model.fit(Xtr_imp, ytr)`; `preds_train`/`probs_train` su `Xtr_imp`; `labels_train = ytr`.
- ramo `lr`: `model.fit(X_train_scaled, y_train)` → `model.fit(Xtr_scaled, ytr)`; `preds_train`/`probs_train` su `Xtr_scaled`; `labels_train = ytr`.
- ramo `TFPN`: `model.fit(X_train_imp, y_train)` → `model.fit(Xtr_imp, ytr)`; `preds_train`/`probs_train` su `Xtr_imp`; `labels_train = ytr`.
- ramo `mlp`:
  - `X_train_t, y_train_t = to_tensors(X_train_scaled, y_train)` → `to_tensors(Xtr_scaled, ytr)`
  - `input_size=X_train_scaled.shape[1]` → `input_size=Xtr_scaled.shape[1]`
  - `n_pos = y_train.sum()` → `n_pos = ytr.sum()`
  - `n_neg = len(y_train) - n_pos` → `n_neg = len(ytr) - n_pos`
  - le `probs_train, labels_train = get_probs(train_loader, ...)` restano invariate (il `train_loader` è già costruito sul sottocampione).

**Non toccare** i riferimenti a `X_val_*`, `X_test_*`, `y_val`, `y_test`.

- [ ] **Step 5: Cella 29 — logga `train_size` e salva nello store**

Nel dizionario passato a `wandb.log({...})`, aggiungi due chiavi (prima di `"roc_curve"`):

```python
                "train_size":        wandb_config.train_size,
                "train_size_actual": len(idx),
```

Dopo il blocco `if tag is not None: metrics_store[...] = {...}` (lasciandolo invariato), aggiungi:

```python
            # Learning curve: salva tutte le metriche indicizzate per train_size.
            if tag is not None:
                learning_curve_store[(wandb_config.model_type, tag, int(wandb_config.train_size))] = {
                    "train_size_actual": len(idx),
                    "accuracy_train":    acc_train,
                    "F1_train":          f1_train,
                    "F1_val":            f1_val,
                    "accuracy":          acc,
                    "F1":                f1,
                    "precision":         precision,
                    "recall":            recall,
                    "AUC":               roc_auc,
                    "average_precision": ap,
                }
```

- [ ] **Step 6: Verifica statica del notebook (sintassi celle modificate)**

Run:
```bash
.venv/bin/python - <<'PY'
import json, ast
nb = json.load(open('notebook.ipynb'))
for i in (1, 25, 29):
    src = ''.join(nb['cells'][i]['source'])
    ast.parse(src)  # solleva SyntaxError se la cella e' rotta
    print(f"cell {i}: OK")
assert 'learning_curve_store' in ''.join(nb['cells'][1]['source'])
assert 'subsample_train' in ''.join(nb['cells'][25]['source'])
src29 = ''.join(nb['cells'][29]['source'])
assert 'Xtr_imp' in src29 and 'ytr' in src29 and 'train_size_actual' in src29
assert 'X_train_imp, y_train' not in src29  # nessun fit sul pool intero residuo
print("checks passed")
PY
```
Expected: `cell 1: OK`, `cell 25: OK`, `cell 29: OK`, `checks passed`.

- [ ] **Step 7: Cella 32 — porta `number_of_runs` a 60**

Nella cella 32, modifica:

```python
number_of_runs = 60   # 6 modelli x 10 dimensioni (learning curve completa)
```

- [ ] **Step 8: Commit**

```bash
git add notebook.ipynb
git commit -m "feat: subsample train by train_size and log learning-curve metrics"
```

---

### Task 5: Notebook — cella chart delle learning curve

**Files:**
- Modify: `notebook.ipynb` (nuova cella markdown + nuova cella code, in coda)

**Interfaces:**
- Consumes: `plot_learning_curves` (Task 2), `learning_curve_store` e `tag` (Task 4).

- [ ] **Step 1: Aggiungi una cella markdown in coda**

Contenuto:

```markdown
## Learning curves

Esegui lo sweep (cella 32) con il parametro `train_size` attivo, poi scegli una
metrica: la cella sotto costruisce il chart con una curva di colore diverso per
ogni modello. Metriche disponibili: `F1`, `AUC`, `precision`, `recall`,
`accuracy`, `F1_val`, `F1_train`, `accuracy_train`, `average_precision`.
```

- [ ] **Step 2: Aggiungi la cella code in coda**

Contenuto:

```python
metric_to_plot = "F1"   # cambia con la metrica desiderata

fig = plot_learning_curves(learning_curve_store, tag, metrics=metric_to_plot)
plt.show()
```

- [ ] **Step 3: Verifica statica della nuova cella**

Run:
```bash
.venv/bin/python - <<'PY'
import json, ast
nb = json.load(open('notebook.ipynb'))
last_code = [c for c in nb['cells'] if c['cell_type']=='code'][-1]
src = ''.join(last_code['source'])
ast.parse(src)
assert 'plot_learning_curves' in src and 'metric_to_plot' in src
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add notebook.ipynb
git commit -m "feat: add learning-curve chart cell to notebook"
```

---

## Self-Review

- **Spec coverage:** config `train_size` (Task 3) ✓; helper subsampler (Task 1) ✓; `learning_curve_store` (Task 4) ✓; sottocampione annidato/stratificato/seeded (Task 1) ✓; val/test fissi (Task 4, sostituiti solo gli usi del train) ✓; imputer/scaler invariati (cella 25 non rifitta) ✓; `metrics_store` invariato (Task 4 lo lascia) ✓; logging wandb con `train_size` (Task 4) ✓; helper plot + cella chart per metrica con colore per modello (Task 2, Task 5) ✓; `number_of_runs = 60` (Task 4, Step 7) ✓.
- **Placeholder scan:** nessun TBD/TODO; ogni step ha codice/comando concreti.
- **Type consistency:** `make_nested_subsampler` → `subsample(k)` → `np.ndarray` usato in cella 29; `plot_learning_curves(store, tag, metrics)` → `Figure`, firma coerente tra Task 2 e Task 5; chiavi store `(model_type, tag, train_size)` coerenti tra Task 4 e Task 2.
