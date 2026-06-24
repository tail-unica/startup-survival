# Learning curves via sweep parameter — Design

**Data:** 2026-06-24
**Branch:** TabPFN

## Obiettivo

Osservare le *learning curve* dei modelli, cioè l'andamento di tutte le metriche
al crescere del numero di campioni nel training set. La dimensione del training
viene integrata come parametro dello sweep, quindi entra in `wandb.config` di ogni
run e diventa l'asse x dei chart. I valori delle metriche vengono inoltre salvati
in una struttura indicizzata per dimensione, così da poter costruire i chart anche
nel notebook.

## Decisioni metodologiche

- **Opzione A — test/val fissi.** Si tiene lo split attuale (train pool 18.178,
  val ≈ 6.060, test ≈ 6.060 su `dataset_window`). Per ogni `train_size` si
  sottocampiona **solo** il train; val e test restano invariati così le metriche
  sono confrontabili tra le diverse dimensioni. È il principio di
  `sklearn.learning_curve`.
- **Sottocampione annidato e stratificato.** I 1000 ⊂ i 2000 ⊂ … (curva più
  liscia, meno rumore) e le proporzioni di classe sono rispettate ad ogni
  dimensione. Deterministico (seed fisso).
- **Imputer e scaler fittati una sola volta** sul train pool completo (cella 25),
  non rifittati per ogni sottocampione. Modifica minima; imputazione/scaling sono
  stabili. (Confermato dall'utente.)
- **`metrics_store` non viene toccato** per non rompere le celle di confronto
  cross-experiment (35–41). Le learning curve usano una struttura separata.

## Dimensioni e costo

- `train_size ∈ {1000, 2000, …, 10000}` → 10 valori.
- `model_type ∈ {rf, lgb, mlp, dt, lr, TFPN}` → 6 modelli.
- Sweep `grid` → 6 × 10 = **60 run**.

## Modifiche

### 1. `config/config.yaml` — nuovo parametro di sweep

Dentro `sweep_settings.parameters`:

```yaml
    train_size:
      values: [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
```

Essendo `method: grid`, wandb genera tutte le combinazioni `model_type × train_size`
e popola `wandb.config.train_size`.

### 2. `src/utils.py` — helper per il sottocampione

```python
def make_nested_subsampler(y, seed):
    """Restituisce subsample(k) -> indici posizionali, stratificati e annidati."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    fractions = counts / counts.sum()
    order = {c: rng.permutation(np.where(y == c)[0]) for c in classes}  # shuffle una volta
    def subsample(k):
        idx = []
        for c, frac in zip(classes, fractions):
            n_c = min(int(round(k * frac)), len(order[c]))
            idx.extend(order[c][:n_c])      # prefisso -> annidato
        return np.sort(np.array(idx))
    return subsample
```

Proprietà: **stratificato** (rispetta le proporzioni di classe), **annidato**
(`round(k·frac)` monotòno + prefisso), **deterministico** (seed fisso).

Helper opzionale per i chart nel notebook:

```python
def plot_learning_curves(store, tag, metrics=("F1", "AUC")):
    """Da learning_curve_store costruisce un chart per ogni metrica:
    x = train_size, una linea per model_type, filtrando per tag."""
    # raggruppa le chiavi (model_type, tag, train_size) per model_type,
    # ordina per train_size, plotta una figura per metrica.
```

### 3. Notebook cella 1 — nuova struttura di salvataggio

Accanto a `shap_store = {}` e `metrics_store = {}`:

```python
learning_curve_store = {}
```

### 4. Notebook cella 25 — istanzia il subsampler

Dopo lo split / imputazione / scaling:

```python
subsample_train = make_nested_subsampler(y_train, config['random_seed'])
```

`X_train_imp`, `X_train_scaled`, `y_train` restano il pool completo, allineati
posizionalmente.

### 5. Notebook cella 29 `make_train` — sottocampiona il train

Subito dopo `set_seed(...)`:

```python
idx = subsample_train(wandb_config.train_size)
Xtr_imp    = X_train_imp[idx]
Xtr_scaled = X_train_scaled[idx]
ytr        = y_train.iloc[idx].reset_index(drop=True)
```

Nei rami dei modelli si sostituiscono gli usi del training:

- `X_train_imp` → `Xtr_imp` (rf, lgb, dt, TFPN)
- `X_train_scaled` → `Xtr_scaled` (lr, mlp)
- `y_train` → `ytr` (fit, `labels_train`, ratio `scale_pos_weight` lgb,
  `n_pos`/`pos_weight` mlp, `to_tensors`)

**Val e test restano invariati** (`X_val_*`, `X_test_*`, `y_val`, `y_test`).

### 6. Notebook cella 29 — logging e salvataggio metriche

Nel blocco `wandb.log(...)` si aggiunge la dimensione effettiva:

```python
"train_size":        wandb_config.train_size,
"train_size_actual": len(idx),
```

(`train_size` è anche già in `wandb.config`; loggarlo come metrica facilita i chart.
`train_size_actual` può differire di ±1 per arrotondamento.)

Dopo il calcolo delle metriche, si salva nella struttura dedicata:

```python
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

`metrics_store` resta invariato.

### 7. Notebook cella 32 — numero di run

```python
number_of_runs = 60   # 6 modelli × 10 dimensioni
```

## Come si osservano le learning curve

- **In wandb:** pannello con x = `train_size` (config o metrica), y = qualsiasi
  metrica loggata, *group by* `model_type`. Tutte le metriche loggate sono
  osservabili rispetto a `train_size`.
- **Nel notebook:** `plot_learning_curves(learning_curve_store, tag, metrics=(...))`
  dopo aver eseguito lo sweep.

## Fuori scope

- Rifit di imputer/scaler per sottocampione.
- Modifica di `metrics_store` e delle celle di confronto cross-experiment (35–41).
- Sottocampionamento del val/test.
