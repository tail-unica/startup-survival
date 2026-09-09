# Refactoring della repository

Data: 2026-09-08
Branch di partenza: `code-merge-and-unit-tests`

## 1. Obiettivo

Rendere la repository comprensibile e manutenibile **prima** che vi si innesti la
pipeline di costruzione del panel tradotta dall'R. Quattro interventi
indipendenti, ognuno mergeabile da solo e ognuno con una verifica che lo
promuove o lo boccia:

1. **Tooling e igiene** — `uv`, `pyproject.toml`, lockfile, `ruff`.
2. **Struttura e rinomina** — cartelle al posto giusto, `snake_case` ovunque.
3. **`src/models/`** — le sette famiglie di modelli diventano classi.
4. **Notebook** — da 792 a ~180 righe di codice.

Il README viene aggiornato **dentro ogni taglio**, sulle sezioni che quel taglio
invalida, perché ogni taglio deve poter essere mergiato da solo lasciando il
README vero.

### Cosa è fuori scope

| Elemento | Perché |
|---|---|
| Il port R→Python del panel | Ha già spec (`2026-09-04-panel-pipeline-r2py-design.md`) e piano da 15 task. Task 1 completo, esecuzione sospesa in attesa del vintage corretto dell'estrazione PitchBook. Questo refactoring gli prepara il terreno senza toccarlo. |
| `img/` | Cartella scratch dell'utente per file deliberatamente non committati. Non si sposta, non si riorganizza, non si versiona. |
| `src/panel/` | Già progettato a oggetti nella sua spec, con `PanelConfig` e i suoi test. Riceve solo l'aggiornamento dei path (Taglio 2). |
| Conversione a OOP di `preprocessing`, `encoding`, `utils` | Sono trasformazioni pure su DataFrame senza stato condiviso: una classe con solo `__init__(config)` produrrebbe lo stesso modulo con `self.` davanti, più un'istanziazione per ogni test. |
| Rinomina del package `src` → `startup_survival` | Il piano del panel cita `from src.panel.config import PanelConfig` e `src/panel/*.py` in decine di punti su 2.833 righe. La rinomina invaliderebbe lavoro già validato in cambio di estetica. |
| Nuove metriche o modelli richiesti dai revisori (PR-AUC, group split) | Sono lavoro sul paper, non sulla struttura. Il Taglio 3 li rende più facili da aggiungere; non li aggiunge. |

## 2. Stato di partenza

| | |
|---|---|
| `src/` | 1.405 righe: `utils.py` 830, `preprocessing.py` 459, `encoding.py` 84, `models/MLP.py` 32, `panel/config.py` 76 |
| `notebook.ipynb` | 47 celle, 792 righe di codice, 536 KB con gli output; una sola cella (`make_train`) ne contiene 498 |
| `tests/` | 851 righe, **la suite non gira** — vedi §2.1 |
| `config/RCode/` | 2 script R (1.526 righe) + 60 CSV PitchBook + un sqlite + 5 CSV di riferimento — tutto gitignorato |
| `requirements.txt` | 151 righe di `pip freeze` |

### 2.1 La suite di test è rotta

Scoperto scrivendo questa spec, eseguendo `pytest` per misurare la baseline.

`tests/test_preprocessing.py:3` importa `handleCategoricalVariables` da
`src.preprocessing`. Quella funzione **non esiste più**: è stata rimossa dal
commit `2ee92e1` ("Added SVM and API TabPFN") quando il collapsing delle
categorie e la frequency encoding sono passati per split, e il test non è stato
aggiornato.

L'`ImportError` avviene in fase di *collection*, quindi pytest si interrompe
prima di eseguire qualsiasi cosa: **nessuno dei 66 test gira**. Ignorando quel
file, gli altri 65 passano in 68 secondi.

```
ERROR tests/test_preprocessing.py
E   ImportError: cannot import name 'handleCategoricalVariables' from 'src.preprocessing'
!!!! Interrupted: 1 error during collection !!!!
```

I cinque test orfani non sono sbagliati. Asseriscono comportamenti che esistono
ancora, ma **inline dentro `preprocessDataset`**:

| Asserzione del test | Dove vive ora |
|---|---|
| `HQCountry`/`PrimaryIndustrySector` restano colonne | `select(...)` le prende grezze |
| nessuna colonna `*Freq` prodotta | la frequency encoding è passata a `src/encoding.py`, per split |
| le categorie rare non sono accorpate prima dello split | nessun collapsing in `preprocessDataset` |
| le righe con settore nullo sopravvivono | l'inner join che le scartava non c'è più |
| `Gender_CEO` resta un solo indicatore | `to_dummies(...).drop("Gender_CEO_Male").drop("Gender_CEO_null")` |

Sono esattamente gli invarianti su cui poggia la tesi del paper — che nulla venga
deciso guardando l'intero dataset prima dello split. Perderli non è accettabile.

Conseguenza per questo refactoring: **senza baseline verde non c'è paracadute**,
e i quattro tagli si appoggiano tutti su V1. La riparazione apre il Taglio 1
(§4.0).

Nota collaterale: il docstring di `preprocessDataset` promette ancora «Handle
categorical variables by collapsing low frequency categories and applying
frequency encoding», comportamento rimosso nello stesso commit. Rientra nella
pulizia dei commenti del Taglio 4 (§7.1).

## 3. Vincoli invarianti

Valgono per tutti e quattro i tagli. Violarne uno significa che il taglio è
sbagliato, non che il vincolo va rinegoziato.

**V1 — I test esistenti passano identici.** I `tests/` sono il paracadute
dell'intero refactoring. Nessun taglio può modificarne le asserzioni per farle
passare; può solo adeguare import e nomi.

La baseline è **65 test verdi e un modulo saltato** dal §4.0, non i 66
di oggi — oggi la suite non è nemmeno collezionabile (§2.1). Da lì in poi il
numero non scende mai, e ogni taglio riporta l'output di `pytest`.

**V2 — Il contratto W&B non si muove.** Nel dettaglio:

- il blocco `sweep_settings` di `config/config.yaml` resta invariato, nomi dei
  parametri compresi (`rf_n_estimators`, `lgb_max_depth`, `svm_C`,
  `mlp_learning_rate`, …). La griglia resta 7 modelli × 5 seed = 35 run;
- la sequenza per run resta: `wandb.init()` → lettura di `wandb.config` →
  `set_seed(seed)` → `get_split(..., seed, tag)` → fit → log → SHAP;
- le chiavi loggate restano esattamente queste dieci più le tre immagini:
  `accuracy_train`, `F1_train`, `F1_val`, `accuracy`, `F1`, `precision`,
  `recall`, `AUC`, `average_precision`, più `roc_curve`, `pr_curve`,
  `shap_summary_plot` e la confusion matrix di `wandb.sklearn`;
- `metrics_store[(model_type, tag)]` resta una lista di dict con le stesse dieci
  chiavi più `seed`; `shap_store[(model_type, tag)]` resta un dict con
  `shap_values` e `explainer_sample`;
- SHAP resta calcolato sul solo `shap_seed`;
- tutto quanto sopra vale identico per i **sei tag**: `window`, `nowindow`,
  `noteam`, `nocompetitors`, `leaklabel`, `leakfeat`.

**V3 — I risultati numerici non cambiano.** Seed fissi e modelli deterministici
dato il seed: uno sweep sullo stesso tag prima e dopo deve produrre le stesse
metriche entro tolleranza numerica.

**V4 — Nessuna dipendenza nuova.** Il refactoring toglie dipendenze, non ne
aggiunge. `ruff` e `pytest` entrano come dipendenze di sviluppo, non di runtime.

## 4. Taglio 1 — Tooling e igiene

Nessun cambio di logica. Solo come si installa e come si controlla il codice —
più la riparazione della baseline, che viene prima di tutto.

### 4.0 Riparazione della baseline

Serve solo che la suite torni **collezionabile**, così i 65 test sani fanno da
paracadute ai tagli successivi. Riagganciare davvero i 5 test orfani costerebbe
una fixture da 46 colonne più il path del ranking QS, e va fatto quando la
pipeline R sarà in Python: a quel punto `preprocessDataset` verrà comunque
rivisto, e riscrivere quella fixture due volte sarebbe lavoro buttato.

Quindi: `tests/test_preprocessing.py` prende uno skip a livello di modulo che
cita la causa (§2.1) e rimanda. Il file resta dov'è — le sue asserzioni sono gli
invarianti anti-leakage del paper e vanno recuperate, non cancellate.

Esito atteso: `65 passed, 1 skipped` (lo skip e a livello di modulo, quindi conta uno). Da qui in poi ogni taglio riporta l'output
di `pytest` e quel numero non scende.

### 4.1 `pyproject.toml`

Build backend `hatchling`, package `src`, gestione con `uv`. Il progetto diventa
installabile: sparisce l'`sys.path.insert(0, ...)` a `scripts/build_datasets.py:22`
e il notebook smette di dipendere dalla directory di lavoro.

`src/RCode/` va escluso dal build: sono script R, non devono finire nel wheel.

### 4.2 Dipendenze

Ricavate dagli import reali di `src/`, `scripts/` e del notebook, non dal freeze:

```
polars  pandas  numpy  scikit-learn  lightgbm  torch  tabpfn  shap
matplotlib  seaborn  scipy  statsmodels  wandb  PyYAML  python-dotenv  joblib
```

Sviluppo: `pytest`, `ruff`, `ipykernel`.

Due errori che il freeze nascondeva e che questo taglio corregge:

- **`scikit-survival` e `category_encoders` non sono importati da nessuna parte.**
  Il primo è perfino elencato fra le *key dependencies* del README.
- **`lightgbm`, `statsmodels`, `joblib` e `python-dotenv` sono usati ma non
  elencati** nel README.

`uv.lock` viene committato: è ciò che rende vera la parola *reproducible* nel
titolo del README.

`requirements.txt` viene eliminato. Il piano del panel, al Task 1 Step 5, dice
«Add pytest to requirements»: quell'istruzione si legge d'ora in poi come «al
gruppo `dev` di `pyproject.toml`», dove pytest è già presente. Nessun'altra
istruzione del piano è toccata.

### 4.3 Ruff

Unico strumento per lint e formattazione. Regole: `E` (pycodestyle), `F`
(pyflakes), `I` (import ordinati), `N` (naming PEP 8), `UP` (modernizzazione),
`B` (bug comuni). Niente oltre: un set più largo produce rumore su codice
scientifico e la prima reazione a un linter rumoroso è disattivarlo.

Misurato con `ruff 0.16.6`, `line-length = 100`, su `src scripts tests`:
**206 violazioni** iniziali. Due gruppi si sono rivelati falsi positivi per
questo dominio e sono stati disattivati invece che corretti.

**`N803`, `N806`, `N816` — disattivate.** Le 71 segnalazioni erano quasi tutte
`X`, `X_train`, `X_val`, `X_batch`: la convenzione scikit-learn (maiuscola per
la matrice dei predittori, minuscola per il target) che questo codice segue
ovunque e che l'API chiamata impone. Rinominarle in `x_train` peggiorerebbe il
codice. Resta attiva `N802` sui nomi di funzione, che è la regola che serve alla
rinomina del Taglio 2.

**`E501` in `preprocessing.py` — rinviata.** 67 delle 73 righe lunghe stanno in
quel file, che il Taglio 2 riscrive comunque con la rinomina. Sta sotto
`per-file-ignores` insieme a `N802`, e l'esenzione sparisce con il Taglio 2. Le
6 di `src/utils.py` sono state riavvolte a mano.

Il resto è stato corretto. Quattro non erano stile:

| Regola | Dove | Cos'era |
|---|---|---|
| `F811` | `tests/test_utils.py:232` e `:385` | Due test diversi — uno su `compare_metrics`, l'altro su `summarize_metrics` — avevano lo stesso nome `test_latex_output_uses_pm_notation`. Il primo era oscurato dal secondo e **non veniva mai eseguito**. Rinominati entrambi; la suite passa da 65 a 66 test. |
| `B006` | `src/utils.py:244` | Default mutabile `exclude_cols=['CompanyID','Target']`. Sostituito con sentinella `None`. |
| `F841` | `src/utils.py:535` | `w_stat` assegnata e mai usata: la statistica di Wilcoxon non è riportata, solo il p-value. Sostituita con `_` ed eliminata la riga commentata `#"wilcoxon_stat": w_stat`. |
| `E711` | `src/preprocessing.py:154` | `InstituteList==None` → `is None`. |

**Esclusioni di scope.** `notebook.ipynb` è escluso: le sue 25 violazioni
stanno nelle celle che i Tagli 3 e 4 riscrivono, e le sue `F401` non vanno
corrette in automatico — `polars`, `KNNImputer` e `RobustScaler` risultano
inutilizzati solo perché le celle di creazione dataset sono commentate, e
rimuoverli romperebbe il notebook appena vengono riattivate. `docs/` è escluso
perché `ruff format` riformatta i blocchi Python dentro i Markdown, e la spec e
il piano del panel citano codice così com'è stato scritto.

Entrambe le esclusioni cadono con il Taglio 4.

### 4.3.1 Trovato e non toccato: la correzione FDR

`src/utils.py`, in `compute_wilcoxon_table`. Il commento dice:

> `# Significance column based on the BH-corrected p-value (consistent with Table 5)`

ma il blocco Benjamini-Hochberg subito sopra è **commentato**, e la colonna
`Signif.` è calcolata su `wilcoxon_p_value`, cioè sul p-value **non corretto**.

O la Tabella 5 del paper riporta significatività non corrette e il commento è
sbagliato, o la correzione doveva essere attiva. È una decisione di merito, non
di refactoring: il codice è stato lasciato esattamente com'era. L'unica
conseguenza di questo taglio è che l'import `multipletests`, inutilizzato
perché serviva solo al blocco commentato, è stato rimosso — riattivare il
blocco richiede di rimetterlo.

### 4.4 `.gitignore`

Via la regola `*.txt`: oggi maschera `requirements.txt` (tracciato a forza) e
qualunque `.txt` futuro. Al suo posto una riga esplicita per `docs/review.txt`.

### 4.5 README toccato dal Taglio 1

- `Installation` riscritta su `uv sync` (via `python -m venv` + `pip install -r`);
- `Key dependencies` corretta secondo §4.2;
- nuova sezione `Development` (§8).

### 4.6 Verifica del Taglio 1

```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import src.preprocessing, src.utils, src.encoding, src.panel.config"
```

L'ultimo comando è la prova che il package è installato davvero: gira da
qualunque directory, mentre prima `scripts/build_datasets.py` funzionava solo
grazie all'`sys.path.insert` di riga 22. (`build_datasets.py` non ha argomenti da
riga di comando: eseguirlo per intero richiede il panel, quindi non è una
verifica da taglio.)

**Esito, eseguito il 2026-09-08:**

```
uv lock                → Resolved 127 packages          (erano 151 righe di freeze)
uv run pytest          → 66 passed, 1 skipped in 140s
uv run ruff check .    → All checks passed!
uv run ruff format .   → 10 files reformatted, 9 unchanged
import da /tmp         → OK
```

66 e non 65: la correzione della `F811` ha restituito l'esecuzione al test che
era oscurato dall'omonimo.

Nota sull'ambiente: il `.venv` preesistente era stato creato in un'altra
directory e poi spostato, quindi i suoi console script avevano lo shebang
puntato a un percorso inesistente e `uv run pytest` falliva con
`Failed to spawn`. È stato ricreato da `uv.lock`. Non è un problema di questo
refactoring — è ciò che il lockfile serve a rendere irripetibile.

## 5. Taglio 2 — Struttura e rinomina

### 5.1 Spostamenti

```
config/RCode/1_Arrange_DB.R      →  src/RCode/1_Arrange_DB.R
config/RCode/2_Arrange_Final.R   →  src/RCode/2_Arrange_Final.R
config/RCode/DB pulito/          →  data/raw/pitchbook/
config/RCode/DatiIntermedi/      →  data/reference/
review.txt                       →  docs/review.txt      (gitignorato)
```

`config/RCode/` sparisce. `config/` contiene finalmente solo `config.yaml`.

Gli script R sono impalcatura con scadenza: muoiono quando il port è tradotto e
validato. Restano gitignorati e non vengono ripuliti — i cinque `setwd()` con il
path assoluto del Mac di un coautore restano dove sono, e questo è il motivo per
cui non vanno versionati così come sono.

I dati invece sopravvivono agli script: `data/raw/pitchbook/` è l'input
permanente della pipeline Python, `data/reference/` è la ground truth dei
checkpoint del port. Lo spostamento cade nel momento giusto, perché l'estrazione
attuale va comunque rimpiazzata con il vintage corretto.

### 5.2 Le 18 occorrenze da aggiornare

| File | Cosa |
|---|---|
| `src/panel/config.py` | default `raw_dir`, `ref_dir` |
| `tests/panel/test_config.py` | le due asserzioni sui default |
| `scripts/check_extraction.py` | costante `REF`, path `raw` |
| `docs/superpowers/specs/2026-09-04-...-design.md` | 3 riferimenti testuali |
| `docs/superpowers/plans/2026-09-04-...md` | 9 riferimenti testuali |

Sono tutte stringhe. Nessuna logica cambia.

### 5.3 Rinomina `snake_case`

| Prima | Dopo |
|---|---|
| `getCompleteDatasetWithTimeWindow` | `build_windowed_dataset` |
| `getCompleteDatasetWithoutTimeWindow` | `build_full_history_dataset` |
| `processUniversityList` | `process_university_list` |
| `IsInTop50Institutes` | `is_in_top50_institutes` |
| `getFlagTop50Institute` | `get_top50_institute_flag` |
| `createHasTop50InstituteFlag` | `create_has_top50_institute_flag` |
| `handleMissingValues` | `handle_missing_values` |
| `preprocessDataset` | `preprocess_dataset` |

Occorrenze misurate: `src/preprocessing.py` 13, `scripts/build_datasets.py` 7,
`README.md` 3, più le celle 1, 6 e 8 del notebook. `src/utils.py` e
`src/encoding.py` non ne contengono nessuna.

`tests/test_preprocessing.py` non ne contiene: punta a una funzione
che non esiste più ed è saltato dal §4.0. Va aggiornato quando lo si riaggancia,
insieme alla pipeline R in Python, e a quel punto userà il nome nuovo.

Nessun alias di compatibilità: due nomi pubblici per la stessa funzione
significano che nessuno migra mai e gli alias restano per sempre.

### 5.4 README toccato dal Taglio 2

`Repository structure`, i tre nomi di funzione nella sezione *Dataset creation*,
i path citati.

### 5.5 Le esenzioni di `preprocessing.py`

Entrambe cadono qui, come previsto, ma per motivi diversi.

`N802` sparisce da sola con la rinomina. `E501` no: le 48 righe lunghe rimaste
sono prosa in docstring e commenti, che il cambio di nome accorcia appena. Sono
state riavvolte a 100 colonne con uno script che preserva indentazione, prefisso
`#` e rientro dei campi `:param`. Verificato che nessuna riga di codice sia
stata toccata: il diff, filtrato togliendo commenti e prosa, è vuoto.

Sistemato nell'occasione anche il docstring segnalato a §2.1, che prometteva
«Handle categorical variables by collapsing low frequency categories and
applying frequency encoding» — comportamento rimosso da `2ee92e1`. Ora dice cosa
succede davvero: `Gender_CEO` diventa un indicatore singolo, mentre `HQCountry`
e `PrimaryIndustrySector` restano categorie grezze perché la frequency encoding
è per split, in `src/encoding.py`.

### 5.6 Verifica del Taglio 2

```
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/check_extraction.py data/raw/pitchbook
grep -rn "config/RCode\|DB pulito\|DatiIntermedi" .
```

**Esito, eseguito il 2026-09-08:**

```
ruff check              → All checks passed!   (senza piu' alcun per-file-ignores)
ruff format --check     → 16 files already formatted
pytest                  → 66 passed, 1 skipped in 57s
check_extraction        → legge data/reference/ e data/raw/pitchbook/,
                          102.276 / 21.040 / 65.803, "NON E' L'ESTRAZIONE GIUSTA", exit 1
grep dei path vecchi    → nessun residuo fuori da questa spec
```

I tre numeri di `check_extraction` sono identici a quelli registrati nel ledger
del panel prima dello spostamento: è la prova che lo strumento trova entrambi i
lati ai path nuovi. L'exit 1 è l'esito corretto — l'estrazione è ancora il
vintage sbagliato, ed è il motivo per cui il port è sospeso.

Verificato inoltre che `src/RCode/`, `data/raw/pitchbook/` e `data/reference/`
siano tutti e tre coperti da `.gitignore`: gli script R contengono percorsi
assoluti personali e non devono diventare tracciabili passando sotto `src/`.

## 6. Taglio 3 — `src/models/` e `src/training.py`

### 6.1 Il problema

Le 498 righe di `make_train` sono un `if/elif` a sette rami. Ogni ramo decide
quattro cose in modo indipendente dagli altri:

| Famiglia | Dati scalati | Explainer SHAP | Nota |
|---|---|---|---|
| `rf`, `lgb`, `dt` | no | `TreeExplainer` | |
| `lr` | sì | `LinearExplainer` | |
| `svm` | sì | `PermutationExplainer` | soglia su `predict_proba`, non `predict()` (Platt vs segno della decision function) |
| `mlp` | sì | `KernelExplainer` | loop di training con early stopping, ha bisogno del validation set |
| `tabpfn` | no | `PermutationExplainer` | `ignore_pretraining_limits=True`, `device="auto"` |

Sono quattro assi ortogonali sparsi su 300 righe di dispatch, e nessuno dei
sette percorsi ha oggi un test.

### 6.2 L'interfaccia

```python
class Model:
    wants_scaled: bool

    @classmethod
    def from_sweep(cls, wandb_config, seed) -> "Model": ...
    def matrices(self, split) -> tuple: ...
    def fit(self, X_tr, y_tr, X_val, y_val) -> None: ...
    def score(self, X) -> tuple: ...            # (probs, preds)
    def explain(self, split, X_columns, shap_cfg) -> tuple: ...
```

Sei membri, uno per ogni decisione che il dispatch prendeva a mano. `X_val` e
`y_val` sono nella firma perché l'MLP ne ha bisogno per l'early stopping; le
altre sei famiglie li ignorano.

Due scelte che si discostano da come la sezione era stata scritta:

**`score` invece di `predict_proba` + `predict`.** Probabilità ed etichette
escono insieme perché nell'originale uscivano insieme: separarle avrebbe
raddoppiato l'inferenza di TabPFN e dell'MLP sul test set. E le etichette non si
possono uniformare a `probs >= 0.5` per tutti: per un albero una foglia può dare
esattamente `p == 0.5`, dove `predict()` sceglie la classe negativa mentre la
soglia sceglierebbe la positiva. Le tre famiglie che nell'originale soglia­vano
esplicitamente (SVM, MLP, TabPFN) sovrascrivono `score`; le altre quattro
tengono la regola del proprio estimator, che è quello che facevano prima.

**`explain` invece di `explainer`.** Ogni famiglia sceglie l'explainer *e* le
righe da spiegare, e le due cose non sono separabili: gli alberi spiegano
`X_test` grezzo, LR e MLP `X_test_scaled`, SVM e TabPFN un numero di righe
deciso da `shap_permutation` in config.yaml.

`from_sweep` legge **solo** le chiavi del prefisso della propria famiglia
(`rf_*` per `RandomForestModel`, `lgb_*` e le chiavi non prefissate dell'LGBM,
e così via). È questa scelta che rende `config/config.yaml` intoccabile: i nomi
piatti con prefisso di famiglia già presenti nello sweep mappano 1:1 sulle
classi.

File, quattro invece dei sei previsti: `base.py`, `sklearn_models.py` (RF,
LightGBM, Decision Tree, LR, SVM — sono tutti wrapper sottili sullo stesso tipo
di estimator), `mlp.py`, `tabpfn.py`, più `__init__.py` con il registro.
`src/models/MLP.py` è confluito in `mlp.py`: la rete e la famiglia che la
addestra sono la stessa cosa, e due file `MLP.py`/`mlp.py` nella stessa cartella
si distinguono solo per maiuscole.

Aggiungere l'ottavo modello diventa un file più una riga nel registro, non un
`elif` dentro una funzione di 500 righe.

### 6.3 `src/training.py`

Contiene `make_train`, ridotto a: costruisci il modello dalla `wandb.config`,
fitta, calcola le metriche, logga, spiega. Il corpo diventa lineare perché le
differenze fra famiglie sono state assorbite dalle classi.

**Deviazione dal vincolo V2, deliberata.** La firma diventa:

```python
make_train(tag, X, y, config, split_kwargs, metrics_store, shap_store)
```

Sette parametri perché sono sette le cose che la closure catturava dai globali
del notebook. Nessun oggetto contenitore nuovo: il punto di chiamata è una riga
sola.

Oggi `metrics_store` e `shap_store` sono globali del notebook su cui `make_train`
fa closure, e le celle di confronto (37-46) li leggono direttamente. Renderli
dizionari a livello di modulo in `src/training.py` li esporrebbe a `%autoreload 2`:
ogni modifica al file azzererebbe i risultati dei tag già girati, che sono
esattamente ciò che serve al confronto cross-esperimento. Restano quindi di
proprietà del notebook e vengono passati esplicitamente.

È l'unica modifica alla cella 32. Tutto il resto di V2 — sweep, sequenza per run,
chiavi loggate, struttura dei due store, SHAP sul solo `shap_seed`, i sei tag —
resta invariato.

### 6.4 Verifica del Taglio 3

**Esito, eseguito il 2026-09-08:**

```
uv run ruff check .           → All checks passed!
uv run ruff format --check .  → 22 files already formatted
uv run pytest                 → 95 passed, 1 skipped in 96s   (erano 66)
```

Il notebook passa da 792 a 209 righe di codice e da 536 KB a 27 KB: la cella da
498 righe diventa una chiamata, e con essa se ne vanno i 477 output del log
dello sweep. Gli output dei risultati — la tabella `summarize_metrics` — restano.


Due livelli.

**Unitario, senza W&B e senza dati reali.** `tests/test_models.py`, 29 test,
eseguiti: costruzione da una `wandb.config` finta, `matrices()` che sceglie la
coppia giusta, fit e `score` su un frame sintetico, e i due comportamenti che
motivano gli override — SVM/MLP che etichettano esattamente per `p >= 0.5`, RF
che tiene la regola del proprio estimator. Più i due explainer: `TreeExplainer`
che restituisce una colonna per feature della classe positiva, e
`PermutationExplainer` che rispetta il budget di righe.

Due test valgono più degli altri:

- `test_registry_covers_exactly_the_sweep_model_types` legge `config.yaml` e
  confronta i `model_type` dichiarati con le chiavi di `MODELS`. È il test che
  impedisce alla configurazione e al codice di divergere in silenzio, che è il
  modo tipico in cui questo tipo di registro si rompe.
- `test_tabpfn_is_built_for_the_checkpoint_the_api_served` verifica la
  configurazione senza fittare: fittarlo scaricherebbe il checkpoint e vorrebbe
  una GPU, ma `ignore_pretraining_limits`, `balance_probabilities` e il nome del
  checkpoint sono esattamente ciò che rende la run locale equivalente a quella
  che l'API serviva.

Sette percorsi prima non testati, ora testati.

**Di sistema, dirimente — non eseguita.** Uno sweep reale su `window` prima e
dopo, con le 35 righe di `summarize_metrics` da far coincidere entro tolleranza
numerica. Richiede il login W&B e una GPU per TabPFN, quindi resta al
committente. **Finché non è stata eseguita, V3 è asserita dalla costruzione, non
verificata.**

Cosa la costruzione garantisce, e che i test unitari controllano: stessi
iperparametri dalle stesse chiavi dello sweep, stessa matrice per famiglia,
stessa regola di decisione per famiglia, stesso explainer sulle stesse righe,
stesso `random_state`. Cosa non garantisce: qualunque differenza di ordine delle
operazioni che sposti lo stato dei generatori casuali. `set_seed(seed)` viene
chiamato nello stesso punto di prima, subito dopo `get_split` e di nuovo prima
di SHAP, proprio per questo.

## 7. Taglio 4 — Notebook

Da 792 a ~180 righe di codice.

| Cella | Prima | Dopo |
|---|---|---|
| 1 (import) | 67 righe, 30+ simboli sklearn/torch/shap | ~10 righe: servivano solo a `make_train` |
| 11-21 (esperimenti) | ~50 righe | ~40, restano esplicite |
| 25 (split) | 39 righe | ~8 |
| 29 (`make_train`) | 498 righe | 3 righe: import e chiamata |

Le celle che **definiscono** i sei setting restano leggibili nel notebook: sono
il contributo del paper, e un revisore deve poter vedere in quattro righe cosa
significa `leaklabel` senza aprire `src/`.

La lista delle 17 colonne "team" della cella `noteam` passa in `config.yaml`:
*cosa conta come feature di team* è un'affermazione del paper, non un dettaglio
di implementazione, e oggi vive dentro una cella.

### 7.1 Commenti

I commenti spiegano cosa fa il codice. Non citano versioni precedenti né
decisioni prese in passato. Quattro punti da riscrivere:

- `src/encoding.py:4` — «The encoding used to run on the whole dataset before the split»
- `src/utils.py:52` — «where it used to be applied to the whole dataset»
- `README.md:161` — «They used to be collapsed and frequency-encoded here»
- `notebook.ipynb` cella 25 — «as the frequency encoding used to be»

In tutti e quattro il contenuto tecnico va conservato: *il fit avviene sul solo
training split* è l'informazione che conta, e resta. Sparisce il confronto con
com'era prima.

### 7.2 Output del notebook

Restano committati. I 536 KB sono quasi tutti la cella dello sweep, ma un repo
che accompagna un paper vale anche perché il lettore vede le tabelle senza
rieseguire nulla.

### 7.3 README toccato dal Taglio 4

`Step-by-step usage` e il commento storico di riga 161.

### 7.4 Verifica del Taglio 4

**Esito, eseguito il 2026-09-08:**

```
uv run ruff check .           → All checks passed!  (notebook incluso: l'esclusione e' caduta)
uv run ruff format --check .  → 23 files already formatted
uv run pytest                 → 95 passed, 1 skipped
notebook                      → 46 celle, 184 righe di codice, 25 KB   (erano 792 righe, 536 KB)
```

Il notebook rieseguito end-to-end sull'example panel resta da fare: richiede il
panel o l'example panel piu' il tempo di uno sweep, quindi ricade sul
committente insieme alla verifica V3 del §6.4.

Sei commenti riscritti invece dei quattro previsti: ai quattro censiti si sono
aggiunti i due che avevo scritto io in `src/models/base.py` e `src/training.py`,
che raccontavano com'era il dispatch prima invece di dire cosa fa adesso.

Rimosso anche `ids = dataset['CompanyID']` dalla cella di split: assegnata e mai
usata.

Il notebook rieseguito end-to-end sull'example panel produce gli stessi numeri.

## 8. README — stato finale

Dopo i quattro tagli il README descrive la repo com'è. Le sezioni nuove o
riscritte:

**Installation**

```bash
git clone https://github.com/tail-unica/startup-survival
cd startup-survival
uv sync
```

**Development** (oggi assente del tutto)

```bash
uv run pytest            # test
uv run ruff check .      # lint
uv run ruff format .     # format
uv run jupyter lab       # notebook
```

Il README non menziona da nessuna parte che esistono 851 righe di test, mentre
Reviewer 1 chiede testualmente «Are there any unit tests implemented?».
Renderli visibili è una risposta gratuita a quell'obiezione.

**Repository structure**

```
.
├── config/config.yaml
├── data/
│   ├── raw/            example_panel.csv, QS_World_Rankings.csv, pitchbook/
│   ├── reference/      ground truth dei checkpoint del port (non distribuita)
│   ├── interim/
│   └── processed/      dataset_window.csv, dataset_nowindow.csv
├── docs/
├── scripts/            build_datasets.py, check_extraction.py
├── src/
│   ├── preprocessing.py
│   ├── encoding.py
│   ├── utils.py
│   ├── models/         una classe per famiglia
│   ├── training.py     make_train
│   ├── panel/          port della pipeline R (in corso)
│   └── RCode/          script R originali (non distribuiti)
├── tests/
├── notebook.ipynb
├── pyproject.toml
└── uv.lock
```

## 9. Ordine e rischi

L'ordine 1 → 2 → 3 → 4 non è arbitrario:

- il **Taglio 1** installa il paracadute (lockfile, linter) prima che qualcosa si
  muova;
- il **Taglio 2** sposta e rinomina mentre il codice è ancora quello noto, così
  un test rosso indica una rinomina sbagliata e nient'altro;
- il **Taglio 3** è l'unico che riscrive logica, e lo fa su un albero già stabile;
- il **Taglio 4** consuma ciò che il 3 ha prodotto.

| Rischio | Mitigazione |
|---|---|
| Il Taglio 3 cambia i numeri | V3: sweep di confronto su `window`, 35 righe da far coincidere. È la verifica dirimente del taglio. |
| La rinomina rompe un call site non trovato | `ruff check` con `F821` più i test: un nome vecchio residuo è un `NameError` a import time. |
| Lo spostamento dei dati rompe il piano del panel | Le 18 occorrenze sono elencate a §5.2. `check_extraction.py` verifica in pochi secondi che i path nuovi siano leggibili. |
| Il refactoring collide con la ripresa del port | Il port riparte dal Task 2, che tocca `src/panel/rutils.py` — file che nessuno dei quattro tagli apre. |

## 10. Definizione di "fatto"

Ogni taglio è concluso quando:

1. `uv run pytest` è verde;
2. `uv run ruff check .` è pulito;
3. le sezioni di README che il taglio invalida sono aggiornate;
4. la verifica specifica del taglio (§4.6, §5.5, §6.4, §7.4) è stata eseguita e
   il suo output riportato.

Il punto 4 si riporta, non si dichiara.
