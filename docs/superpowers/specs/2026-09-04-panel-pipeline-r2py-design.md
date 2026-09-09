# Traduzione in Python della pipeline R di costruzione del panel

Data: 2026-09-04
Stato: approvato, pronto per il piano di implementazione

## 1. Obiettivo

Riscrivere in Python (polars) la pipeline che, a partire dai CSV grezzi
PitchBook, costruisce il panel di startup usato in
`src/preprocessing.py`. Attualmente la pipeline esiste solo come due
script R scritti da terzi.

L'obiettivo di **questa fase** è la riproduzione fedele: l'output Python
deve coincidere con quello R, bug compresi. Le correzioni sono una fase
successiva, abilitate da flag già predisposti ma spenti di default.

Il criterio di successo è verificabile, non dichiarativo: i CSV
intermedi prodotti dagli script R sono disponibili e fanno da ground
truth in quattro checkpoint indipendenti, più verifiche parziali per
colonna negli stadi che non hanno un file di riferimento dedicato
(§8).

### Cosa è fuori scope

- Le lavorazioni successive che hanno prodotto `data/raw/panel.csv.gz`
  a partire dall'output R: temporizzazione anno per anno delle colonne
  competitor, `GrowthStageGroup` / `GrowthNextStageGroup` /
  `TimeNextStageGroup`, e il filtro che porta da 1.001.625 a 882.324
  righe. Verranno aggiunte dopo.
- La correzione dei bug (fase 2).
- La sostituzione definitiva dell'imputazione RandomForest (fase 2),
  di cui qui si predispone solo il punto di innesto.
- Qualunque modifica a `src/preprocessing.py` e al resto del codice
  di modellazione.

## 1-bis. Emendamento 2026-09-09

Tre decisioni successive alla stesura. **Dove questa sezione e il resto
del documento divergono, vale questa.**

**E1 — l'imputazione RandomForest non viene portata.**
`TotalInvestedCapital_Est` e le sei colonne `TotalRaised_Est*` non
vengono prodotte. Gli importi mancanti restano mancanti e li riempie
l'imputazione che già gira prima del training. Il §7 resta come analisi
del problema, ma le tre strategie descritte alla voce «Interfaccia»
decadono: niente `imputation.py`, niente `imputation_strategy` in
`PanelConfig`, niente `r_injected` (non c'è più nulla da iniettare) e
niente `seed` (la pipeline diventa deterministica).

Nel punto dove l'R esegue il modello, lo stadio 4 porta un commento che
elenca le sette colonne create dalla RF e perché è sospesa.

Impatto misurato sui dataset di modellazione, non sul panel:
`dataset_window` 5.914 righe su 30.300 (19,5%) hanno un valore imputato,
pari al 29,7% della massa della feature; `dataset_nowindow` circa 9.739
aziende su 30.300 (32,1%), +7,6% di massa. Ripetere i run è quindi
obbligatorio, SHAP compreso.

Resta aperta, da decidere prima dell'ultimo task: quale colonna prende
il posto di `TotalRaised_Est` in `src/preprocessing.py` — `TotalRaised`
(zero dove l'importo non è dichiarato) oppure `TotalRaised_NA` (null lì,
così l'imputatore a valle vede il buco). Entrambe vengono prodotte.

**E2 — `TR_D` diventa una colonna mantenuta.** Calcolata dall'R alla
riga 1246 e poi buttata via da `vars_selected`, vale 1 quando l'anno-
azienda non ha nessun deal. Va aggiunta a `vars_selected`.

**E3 — il post-processing rientra nello scope.** Il file intermedio
mancante `db_master_panel.csv.gz` è stato fornito e le sue quattro
regole sono state ricostruite e verificate a divergenza zero su tutte le
882.324 righe. Nascono due stadi e due checkpoint:

| checkpoint | stadio | riferimento | righe |
|---|---|---|---|
| E | 6 | `db_master_panel.csv.gz` | 882.324 |
| F | 7 | `data/raw/panel.csv.gz` | 882.324 |

I checkpoint diventano sei; le regole dello stadio 6 e le note dello
stadio 7 stanno nei Task 16 e 17 del piano.

Conseguenza sui confronti: le sei colonne `_Est` esistono nei
riferimenti e non nel nostro output (**assenti attese**, escluse dal
confronto a C, D, E, F); `TR_D` esiste nel nostro output e non in
`db_selected.csv`, `db_master_panel.csv.gz`, `panel.csv.gz`
(**in più attesa**, esclusa a D, E, F); `StageBlock` non è riproducibile
ed è dichiarata divergenza attesa a E e F. Tutto il resto deve
coincidere esattamente.

## 2. Contesto: cosa fa la pipeline R

Due script in `src/RCode/`.

**`1_Arrange_DB.R`** (1257 righe) carica 51 CSV in un database SQLite
temporaneo e vi accede per indice posizionale (`tbl[N]`), poi costruisce:

- `db_master_1` — cross-section per azienda (116.920 righe,
  `YearFounded > 1999`);
- `db_master_2` — panel azienda × anno, generato espandendo
  `seq(YearFounded, MaxYear)`;
- `db3` — tabella persona-azienda (534.851 righe) con istruzione,
  esperienza e finestra temporale di permanenza.

**`2_Arrange_Final.R`** (269 righe) applica cumulate, deriva
`GrowthStage`, calcola medie ponderate cumulate, aggancia gli attributi
del CEO, seleziona le colonne e produce `db_selected` (1.001.625 righe)
e infine `db_final` = `db_selected` + 22 colonne di `db_master_1`.

Il database SQLite serve solo come meccanismo di caricamento. L'unica
informazione che fornisce è la corrispondenza fra indice e tabella,
già decodificata:

| idx | tabella | idx | tabella |
|----:|---------|----:|---------|
| 1 | `Company` | 19 | `Deal` |
| 2 | `CompanyAffiliateRelation` | 22 | `DealInvestorRelation` |
| 3 | `CompanyBoardTeamRelation` | 32 | `Investor` |
| 6 | `CompanyEmployeeHistoryRelation` | 43 | `Person` |
| 8 | `CompanyFinancialRelation` | 48 | `PersonEducationRelation` |
| 14 | `CompanyNewsRelation` | 49 | `PersonPositionRelation` |
| 17 | `CompanySimilarRelation` | | |

Nessun database è necessario nella versione Python.

## 3. Vincoli

- **Memoria: 7 GB di RAM**, contro 5,6 GB di CSV sorgenti.
  `Person.csv` pesa 956 MB, `CompanySimilarRelation.csv` 836 MB,
  `Deal.csv` 293 MB, `PersonPositionRelation.csv` 289 MB. Va usato
  `pl.scan_csv` in lazy con proiezione delle colonne e raccolta in
  streaming; nessuna tabella grande va materializzata per intero.
- **polars**, non pandas e non SQL. Coerente con `src/preprocessing.py`,
  che già lo usa.
- **Riproducibilità**: rieseguire la pipeline due volte deve dare lo
  stesso risultato. Questo è oggi violato dall'imputazione RandomForest
  (§7).

## 4. Architettura

```
src/panel/
  __init__.py
  config.py              PanelConfig: percorsi, flag dei bug, strategia di imputazione
  io.py                  scan_csv con schema espliciti; regola NA di R
  rutils.py              primitive R -> polars
  stage1_company.py      Company + Affiliate -> db_master_1 (v1), scheletro panel
  stage2_team.py         BoardTeam + Person + Education + Position -> db3, panel team
  stage3_relations.py    competitor, employee, financials, news
  stage4_deals.py        Deal + DealInvestor + Investor -> deals_panel
  stage5_final.py        cumulate, GrowthStage, CEO -> db_selected, db_final
  imputation.py          strategie di imputazione di TotalInvestedCapital
  validate.py            confronto con i CSV intermedi R
notebooks/build_panel.ipynb
```

Ogni stadio è una funzione pura `run(cfg) -> None` che legge i parquet
degli stadi precedenti e scrive i propri in `data/interim/`. Rieseguire
uno stadio non impone di rifare i precedenti — vincolo pratico
importante, dato il volume dei dati.

`data/interim/` e `src/RCode/` vanno aggiunti a `.gitignore`.

### Dataflow

```
stage1  Company, CompanyAffiliateRelation
          -> db_master_1_v1.parquet        (cross-section, YearFounded > 1999)
          -> db_master_2_skeleton.parquet  (azienda x anno + variabili time-varying)
                                           [verifica parziale per colonna]

stage2  CompanyBoardTeamRelation, Person, PersonEducationRelation,
        PersonPositionRelation, db_master_1_v1
          -> db3.parquet                   [CHECKPOINT A]
          -> db_master_2_team.parquet      (full_join del panel team)

stage3  CompanySimilarRelation, CompanyEmployeeHistoryRelation,
        CompanyFinancialRelation, CompanyNewsRelation
          -> db_master_1.parquet           [CHECKPOINT B]
          -> db_master_2_relations.parquet

stage4  Deal, DealInvestorRelation, Investor
          -> deals_panel.parquet
          -> db_master_2_deals.parquet     (+ TR_D)
                                           [verifica parziale per colonna]

stage5  tutto lo script 2
          -> db_master_2.parquet           [CHECKPOINT C]
          -> db_selected.parquet           [CHECKPOINT D]
          -> db_final.parquet              (output della pipeline)
```

### Notebook

`notebooks/build_panel.ipynb` richiama gli stadi in sequenza. Per ogni
cella il notebook documenta tre cose:

1. cosa fa la cella e a quali righe dell'R corrisponde;
2. quali bug del registro (§6) sono attivi in quel punto, cosa
   provocano e perché sono bug;
3. l'impatto misurato sul resto della pipeline, non stimato.

Dopo ogni checkpoint il notebook stampa il report di validazione (§8).

## 5. Gli stadi in dettaglio

Per ogni stadio si elencano le regole R che richiedono attenzione
nella traduzione. Le regole ovvie (select, rename, left join semplici)
non sono elencate: la fonte resta lo script R, che va seguito riga per
riga.

### stage1 — Company e affiliate

- Sostituzione NA su tutte le colonne per i valori `""`, `"NA"`,
  `"N/A"`, `"NULL"`, `"NaN"` (per `db1` include `"NaN"`, per `db2` no:
  differenza reale fra le righe 43 e 113 dell'R, da replicare).
- Flag booleani `Website_d`, `Linkedin`, `Facebook`, `Twitter` come
  `nchar(x) > 0`.
- Parsing date con la regola dei due formati: `nchar == 10` →
  `%m/%d/%Y`; `nchar == 8` → `mdy` con `cutoff_2000 = 24` (anni 00–24
  → 2000–2024, 25–99 → 1925–1999); **ogni altra lunghezza → NA**.
- `FiscalPeriod` → `Quarter` via `sub("TTM (\\dQ)(\\d{4})", "\\1", .)`.
  Attenzione: `sub` senza match restituisce la stringa originale, non
  NA; `Month` diventa NA per i valori non riconosciuti.
  `FiscalDate` = `Year-Month-30`.
- `MaxYear = pmax(..., na.rm = TRUE)` sui sei anni disponibili, poi
  espansione `seq(YearFounded, MaxYear)` → vedi bug B6.
- Le variabili time-varying di `db_master_1` si agganciano al panel con
  cinque join su `(CompanyID, anno)`.
- `db_master_1` e `db_master_2` filtrati a `YearFounded > 1999`.

### stage2 — team

- Deduplica per `(CompanyID, PersonID)`: si selezionano le righe
  duplicate, si filtra `db3` per `PersonID %in% duplicati` (per
  `PersonID`, non per la coppia — bug B9), si aggrega con
  `coalesce(first(x), last(x))`, si concatena in testa a `db3` e si
  applica `distinct(CompanyID, PersonID, .keep_all = TRUE)`, che tiene
  quindi le righe ricomposte.
- Join con `Person` per 16 attributi più la chiave `PersonID`; indici
  di esperienza calcolati con
  `scale(log(x + 1))` **globale su tutto `db3`**, non per gruppo.
- Join con `PersonEducationRelation`: classificazione `DegreeLevel`,
  `MBA`, `Major_Concentration`, `Field` con la stessa sequenza di
  `grepl` case-insensitive dell'R, nello stesso ordine (le `case_when`
  sono a corto circuito: il primo match vince).
- Aggregazione per `PersonID` con la gerarchia
  `c("Other", "Diploma/Certificate", "Bachelor's", "Master's", "PhD/Doctorate")`;
  `Highest_Degree` è l'indice numerico del massimo. Vedi bug B2 per
  `Is_Other`.
- Override da `PersonName`: `Ph.D` / ` JD` / ` MD` forzano
  `Highest_Degree = 5` e, per gli ultimi due, `Is_Law` / `Is_Med`.
- Join con `PersonPositionRelation` (deduplicata per
  `(EntityID, PersonID)`) per `PositionLevel`; `IsFounder` da
  `str_detect(paste(FullTitle, PositionLevel, sep = "; "), "(?i)Found")`.
  Nota: `paste` con NA produce la stringa `"NA"`, quindi `IsFounder`
  non è mai NA.
- Imputazione di `StartDate` / `EndDate` in quattro passaggi
  (righe 451–505), incluso `PermanenzaMedia` — vedi bug B5 e B10.
- `EndDate < StartDate` → `StartDate`; poi `DeltaStart` / `DeltaEnd`;
  `DeltaStart = 0` per i founder.
- Espansione: griglia anni per azienda (`DeltaStart_min`..`DeltaEnd_max`)
  in left join con l'espansione persona-anno; aggregazione per
  `(CompanyID, Years)`; filtro `YearFounded > 2000` — vedi bug B1.
- Loop di sostituzione NA sull'aggregato che include anche `"Inf"` e
  `"-Inf"`: converte in NA i `max()` su gruppi tutti-NA (che in R danno
  `-Inf`) e le `mean()` su gruppi tutti-NA (che danno `NaN`). Da
  replicare come espressione esplicita, non come `null_values` in
  lettura.
- `full_join` con lo scheletro su `(CompanyID, Delta = Years)`,
  poi `fill(YearFounded, "downup")` per azienda e ricostruzione di
  `Year_Delta = YearFounded + Delta`.

### stage3 — competitor, dipendenti, financials, news

- `CompanySimilarRelation` (836 MB): join con `HQCountry` da
  `db_master_1`, continente via `countrycode`, aggregazione per
  `CompanyID`. `mean(SimilarityScore)` e `sum(IsCompetitor == "Yes")`
  sono **senza** `na.rm`; `Same_Country` usa `any()` senza `na.rm`
  (bug B4); `N_Europe` conta tutte le righe mentre `N_Outside_Europe`
  solo quelle con `SimilarityScore > 90` (bug B8).
  In Python la mappatura paese → continente va congelata in una tabella
  versionata nel repo, non presa da una libreria che può cambiare fra
  una release e l'altra.
- Dipendenti: ordinamento `(CompanyID, Year, desc(Date))` e
  `distinct(CompanyID, Year)` → si tiene l'ultima rilevazione dell'anno.
- Financials: stessa logica di deduplica su `PeriodEndDate`; i valori
  riempiono solo le celle già mancanti in `db_master_2` (coalesce).
- News: conteggio per `(CompanyID, anno)`, con 0 dove manca.

### stage4 — deal e investitori

- `DealInvestorRelation` + `Investor` → `InvestorCategory` con la
  mappatura a sette classi delle righe 861–871; aggregazione per
  `DealID` (`db22_s`). Molti aggregati sono condizionati a
  `any(InvestorStatus == "New Investor")` e vanno replicati alla
  lettera, incluse le asimmetrie fra i blocchi "New Investor" e
  "Lead Investor".
- `Deal`: tre imputazioni successive di `DealDate` da
  `OwnershipStatusDate` e da `YearFounded`, poi il riempimento
  dei buchi con la media arrotondata verso l'alto fra l'anno del deal
  precedente e quello successivo (`ceiling((Year_Prev + Year_Next) / 2)`),
  che richiede lag/lead ordinati per `(CompanyID, DealNo)`.
- Filtro `YearFounded > 2000` (bug B1).
- `Year_Delta = pmax(year(DealDate), YearFounded)`: i deal antecedenti
  la fondazione sono schiacciati sull'anno di fondazione.
- `UndisclosedAmountFlag` da regex su `DealSynopsis`.
- Regola `Zero_Invested`: i `DealType` con oltre il 90% di
  `TotalInvestedCapital` mancante ottengono 0; quelli con meno di 200
  occorrenze diventano `"Other"`. È una regola derivata da statistiche
  globali sull'intero dataset — non è leakage nel senso stretto, ma
  appartiene alla stessa famiglia dei problemi di §7 e va annotata.
- Imputazione di `TotalInvestedCapital_Est` — §7.
- Classificazione dei `DealType` in 15 flag booleani secondo le liste
  delle righe 1125–1131, poi aggregazione in `deals_panel` per
  `(CompanyID, Year_Delta)`. Le sei varianti di `TotalRaised` hanno
  semantiche di missing diverse e vanno distinte con cura:
  `sum(na.rm = TRUE)`; NA se **tutti** NA; NA se **almeno uno** NA.
- `TR_D = 1` quando tutte e sei le varianti sono NA.
- Ultimo loop di sostituzione NA, di nuovo con `"Inf"` / `"-Inf"`.

### stage5 — finalizzazione

- `NA -> FALSE` su 24 flag, poi `cumany` per azienda ordinata per
  `Year_Delta`.
- `GrowthStage` con la cascata di sette condizioni della riga 82: è a
  corto circuito, l'ordine è vincolante.
- Dove `TR_D == 1`, le sei varianti di `TotalRaised` vanno a 0.
- Blocco di cumulate. Attenzione: `cumsum` in R propaga NA fino in
  fondo al gruppo; `TotalRaised_NA_cum` e `TotalRaised_any_cum` sono
  quindi NA dal primo NA in poi. Va verificata esplicitamente la
  semantica dei null di `cum_sum` in polars e forzata a coincidere.
- `NewInvestors` è calcolato **prima** che `TotalInvestors` venga
  sovrascritto dalla propria cumulata, nello stesso `mutate`: dplyr
  valuta in sequenza, e la traduzione deve preservare l'ordine.
- Media ponderata cumulata per quattro variabili, pesata su
  `NewInvestors`, sulle sole righe con valore e peso non nulli e peso
  positivo. L'implementazione R è O(n²); l'equivalente algebrico
  `cum_sum(x*w) / cum_sum(w)` sulle righe valide è O(n) ed è esatto.
- `CEO_ID` riempito in avanti per azienda, poi join di 18 attributi da
  `db3` su `(CompanyID, PersonID)`. Gli attributi del CEO sono
  invarianti nel tempo: variano solo quando cambia `CEO_ID`.
- `Age = Delta`; selezione delle colonne di `vars_selected`.
- `StageBlock` (bug B3), `YearsInStage` via
  `sequence(rle(GrowthStage)$lengths)` — dove `rle` tratta **ogni NA
  come un run a sé**, comportamento verificato sui dati e da replicare.
- `GrowthNextStage` / `TimeNextStage`: primo stadio futuro diverso da
  quello corrente, ignorando i NA; `TimeNextStage` è la distanza in
  righe. L'indicizzazione `(i+1):n()` sull'ultima riga di ogni gruppo
  produce in R un vettore invertito, ma il risultato resta comunque NA:
  non serve replicare l'anomalia, solo il risultato.
- `db_final` = `db_selected` in left join con 22 colonne di
  `db_master_1`.

## 6. Registro dei bug

Tutti i flag vivono in `PanelConfig`, sono booleani e valgono `False`
di default. `False` significa **comportamento R**. Il notebook stampa
la configurazione attiva a ogni esecuzione.

Ordinati per impatto decrescente.

| id | flag | descrizione | impatto misurato |
|---|---|---|---|
| B1 | `fix_founding_year_threshold` | `db_master_1` e `db_master_2` filtrano `YearFounded > 1999`, ma il panel team (riga 566) e i deal (riga 935) filtrano `YearFounded > 2000` | **ALTO.** La coorte 2000 conta 29.603 righe (3,0% del panel): 0% ha dati di team, 3,2% ha `GrowthStage` contro ~67% delle coorti adiacenti. A valle `preprocessing.py` scarta le righe con `Total_People` nullo, quindi l'effetto netto è che un'intera coorte di fondazione sparisce silenziosamente dal dataset. |
| B5 | `fix_permanenza_media_per_company` | `PermanenzaMedia` è calcolata con una `summarise` senza `group_by`, quindi è un singolo valore globale, nonostante il commento dichiari "per ciascuna CompanyID" | **DA MISURARE.** Determina l'imputazione di `EndDate`, quindi `DeltaEnd`, quindi in quali anni ogni persona viene conteggiata: tocca potenzialmente tutte le feature di team. La misura va fatta appena la pipeline gira. |
| B2 | `fix_is_other_label` | `Is_Other` verifica `Field %in% "Other/Unknown"`, ma `Field` produce `"Other"` e mai `"Other/Unknown"` | MEDIO. La colonna è `False` su tutte le 885.142 righe valorizzate: è inutilizzabile. Non è fra le feature di `preprocessing.py`, quindi non contamina i risultati pubblicati. |
| B4 | `fix_same_country_narm` | `any()` senza `na.rm` restituisce NA quando nessun confronto è vero e almeno uno è NA | BASSO come volume — 1.563 aziende (1,34% di quelle con competitor) — **ma `Same_Country` è una feature dei modelli**. |
| B3 | `fix_stageblock_na` | `cumsum` su un confronto che vale NA propaga NA fino in fondo al gruppo: `StageBlock` diventa NA per l'intera azienda dal primo `GrowthStage` mancante | BASSO. Verificato sull'azienda `100026-46`. La colonna non è usata a valle. |
| B8 | `fix_europe_asymmetry` | `N_Europe` somma su tutte le righe, `N_Outside_Europe` solo su quelle con `SimilarityScore > 90` | BASSO. Nessuna delle due è feature dei modelli. |
| B6 | `fix_negative_delta` | se `MaxYear < YearFounded`, `seq()` genera una sequenza decrescente e produce `Delta` negativi | TRASCURABILE. 245 righe (0,024%) su 106 aziende. |
| B7 | `fix_institute_na_literal` | `paste(unique(Institute), collapse = "; ")` include i NA come stringa letterale `"NA"` | COSMETICO. Presente nell'83,3% delle stringhe `Institute` non vuote; non altera il flag top-50, che cerca nomi di atenei. |
| B9 | `fix_dup_coalesce` | due anomalie nello stesso blocco di deduplica: i duplicati si selezionano filtrando per `PersonID` invece che per la coppia `(CompanyID, PersonID)`, e `coalesce(first(x), last(x))` ignora le righe intermedie, così con tre o più duplicati un valore presente solo in mezzo va perso | BASSO sul risultato; il primo dei due allarga inutilmente il lavoro di aggregazione. |
| B10 | `fix_is_out_na` | dopo il left join con `ownership_out`, `Is_Out` è NA (non `FALSE`) per le aziende non fallite; le condizioni `Is_Out == FALSE` valgono NA e neutralizzano due imputazioni di `EndDate` | TRASCURABILE. Un catch-all successivo (riga 501) recupera i casi, quindi il bug è auto-sanato. |

Due anomalie **senza** flag, perché non serve:

- L'indicizzazione `(i+1):n()` sull'ultima riga di ogni gruppo produce
  in R un vettore invertito, ma il risultato è comunque sempre NA. Si
  replica il risultato, non il meccanismo.
- Le righe "fantasma" nel panel team (un anno senza nessuna persona
  otterrebbe `Total_People = 1` perché `.N` conta la riga di join
  vuota) sono teoricamente possibili ma **empiricamente zero** su
  885.142 righe. Va aggiunta un'asserzione di validazione, non un flag.

## 7. Imputazione di `TotalInvestedCapital`

### Il problema

Alla riga 1087 lo script R imputa `TotalInvestedCapital` con
`randomForest(ntree = 50)` **senza `set.seed`**: non è riproducibile
nemmeno rieseguendo l'R.

Peggio, è una fonte di look-ahead bias dentro un lavoro che denuncia il
look-ahead bias. Tre problemi distinti:

1. **Temporale.** Il modello è addestrato su deal di tutti gli anni e
   imputa un deal del 2013 usando pattern del 2020. `Age` è fra i
   predittori.
2. **Train/test.** L'imputazione è fittata sull'intero dataset prima di
   qualsiasi split. Si aggiungono altre tre statistiche globali: la
   winsorizzazione al p95 (riga 1057), `q3_by_group` (riga 1095) e la
   regola `Zero_Invested` (righe 994–1002).
3. **Prossimità al target.** `DealTypeGrouped` è un predittore, ma i
   deal type (`Merger/Acquisition`, `Bankruptcy`, `IPO`) sono ciò che
   determina `GrowthStage` e quindi il target.

Non c'è invece circolarità: il training usa `TotalInvestedCapital`
originale su `complete.cases`, quindi nessun valore imputato rientra
nel modello che imputa.

### Propagazione

I valori imputati non alimentano nessun'altra feature. Il percorso è
lineare e si esaurisce in sei colonne su ~97:

```
RF -> TotalInvestedCapital_Est (livello deal)
      -> TotalRaised_Est, TotalRaised_Est_NA, TotalRaised_Est_any  (deals_panel)
         -> azzerate dove TR_D == 1
         -> cumsum -> TotalRaised_Est_cum, _Est_any_cum, _Est_NA_cum
```

Di queste sei, **una sola raggiunge i modelli**: `TotalRaised_Est`
(`src/preprocessing.py:475`). E `TotalRaised`, la variante non
imputata calcolata dalla stessa pipeline, non è fra le feature.

### Dimensione

Sui deal di aziende fondate dopo il 2000: 285.336 deal totali, 89.627
(31,4%) con `TotalInvestedCapital` mancante dopo la regola zero, di cui
al massimo 48.000 (16,8%) candidati all'imputazione. È un limite
superiore: manca il filtro `complete.cases` sui predittori, e quattro di
essi esistono solo per i deal con investitori registrati. Il numero
esatto va misurato appena la pipeline gira.

Da notare che i ~41.600 deal che restano NA dopo la RF diventano 0,
perché `TotalRaised_Est = sum(..., na.rm = TRUE)`: la pipeline attuale
già tratta come zero la maggioranza degli importi non dichiarati.

### Interfaccia

`imputation.py` espone una funzione con tre implementazioni
selezionabili da `PanelConfig.imputation_strategy`:

- **`r_legacy`** — `RandomForestRegressor(n_estimators=50,
  random_state=cfg.seed)` con la stessa logica dell'R: stessi
  predittori, stessa winsorizzazione p95, stesso vincolo al terzo
  quartile per `DealTypeGrouped`. Rende la pipeline autonoma e serve a
  **misurare l'arbitrarietà del metodo attuale**: quanto si sposta la
  feature rieseguendo lo stesso metodo con un seed diverso. È un numero
  di per sé utile al paper.
- **`r_injected`** — legge `TotalRaised_Est`, `TotalRaised_Est_NA` e
  `TotalRaised_Est_any` da `db_master_2.csv`. Serve solo alla
  validazione: isola la RF come **unica** divergenza nota, così le tre
  colonne cumulate a valle si verificano in modo esatto e si può
  affermare con prove che tutto il resto coincide.
- **`leakage_free`** — segnaposto che solleva `NotImplementedError`.
  Fase 2. La direzione raccomandata è una mediana condizionata
  strettamente causale (solo deal con data precedente a quello da
  imputare, per `DealTypeGrouped` × `PrimaryIndustrySector` con
  gerarchia di fallback): deterministica, spiegabile in due righe,
  elimina le fonti 1 e 3 e rende la 2 risolvibile ricalcolandola dentro
  il fold. L'alternativa è non imputare e passare ai modelli la
  missingness esplicita.

`PanelConfig.imputation_strategy` vale `r_legacy` di default, così la
pipeline è autonoma dai CSV R. Il notebook la commuta esplicitamente a
`r_injected` nella cella di validazione dei checkpoint C e D.

## 8. Fasi di verifica

Ogni stadio termina con una fase di verifica che confronta il proprio
output con il file R corrispondente e dice, in modo esplicito, **quali
colonne divergono e su quante righe**. La pipeline non prosegue allo
stadio successivo se la verifica precedente non è passata, salvo
forzatura esplicita.

### 8.1 I file di riferimento

I CSV in `data/reference/` sono la ground truth.

| checkpoint | fine stadio | file di riferimento | righe attese | chiave |
|---|---|---|---|---|
| A | 2 | `db3.csv` | 534.851 | `(CompanyID, PersonID)` |
| B | 3 | `db_master_1.csv` | 116.920 | `CompanyID` |
| C | 5 | `db_master_2.csv` | 1.001.625 | `(CompanyID, Year_Delta)` |
| D | 5 | `db_selected.csv` | 1.001.625 | `(CompanyID, Year_Delta)` |

Tre avvertenze sui riferimenti:

- **`db3.csv` è `db3` non filtrato.** Il filtro `YearFounded > 2000` si
  applica a `db3_expanded`, cioè al panel team, non alla tabella
  persona-azienda che viene salvata.
- **`db_master.csv` non è un target.** I due script R non scrivono
  alcun CSV: fanno solo `save()` di `.RData`, e i quattro file sopra
  sono stati esportati a parte. `db_master.csv` ha una colonna indice
  di R, 113.890 righe e uno schema diverso (`Is_Startup`, `Is_IPO`,
  `Time`, `Out_of_Business`, `Employee`, `NetDebt`): proviene da una
  versione precedente della pipeline e va ignorato.
- **`db_final` non ha un riferimento**, ma è `db_selected` in left join
  con 22 colonne di `db_master_1`: entrambi verificati, quindi è
  verificato per costruzione. La verifica si limita a controllare che
  il join non abbia duplicato righe.

### 8.2 Verifiche parziali negli stadi senza riferimento

Gli stadi 1 e 4 non hanno un file dedicato, ma molte colonne di
`db_master_2` sono già definitive prima della fine: `EmployeeCount` e
`N_News`, per esempio, non cambiano più dopo lo stadio 3. Aspettare il
checkpoint C significherebbe scoprire alla fine un errore nato
all'inizio.

`validate.py` mantiene quindi una mappa `COLUMN_FINALISED_AT_STAGE`,
che associa a ogni colonna di `db_master_2` lo stadio dopo il quale non
viene più toccata. La mappa si compila mentre si implementa ciascuno
stadio, perché in quel momento si sa esattamente quali colonne si
scrivono, e va tenuta allineata al codice.

Ogni verifica parziale confronta l'output corrente con
`db_master_2.csv` e classifica le colonne in tre gruppi:

- **verificate** — già definitive a questo stadio e coincidenti;
- **attese diverse** — non ancora definitive; la divergenza è normale e
  viene riportata ma non conta come errore;
- **regressioni** — già definitive a uno stadio precedente ma ora
  divergenti. Questo è il caso interessante: significa che uno stadio
  successivo ha sporcato una colonna che doveva essere ferma.

### 8.3 Cosa produce una verifica

`verify(actual, reference, key) -> VerificationReport` restituisce tre
livelli, che il notebook stampa in ordine.

**Livello 1 — sommario.** Una riga di esito, `PASS` o `FAIL`, seguita
da: righe attese e ottenute; numero di chiavi presenti solo in R e solo
in Python; numero di colonne presenti solo da un lato; numero di
colonne divergenti su totale confrontate.

**Livello 2 — dettaglio per colonna.** Una tabella con una riga per
colonna, **ordinata per numero di righe divergenti decrescente**, così
il problema peggiore è il primo che si legge:

| colonna | tipo R | tipo PY | righe confrontate | righe diverse | % | solo R è NA | solo PY è NA | max diff |
|---|---|---|---|---|---|---|---|---|

Le colonne che coincidono al 100% sono riassunte in una riga di
conteggio invece che elencate una per una: con ~97 colonne, l'elenco
completo nasconderebbe le poche che contano.

Le due colonne `solo R è NA` / `solo PY è NA` sono separate dal
conteggio dei valori diversi ed è deliberato: il grosso del rischio di
traduzione sta nella semantica dei missing, e un disallineamento di NA
è un sintomo diverso da un valore sbagliato. Serve poterli distinguere
a colpo d'occhio.

**Livello 3 — esempi.** Per ogni colonna divergente, le prime `N`
righe (default 10) con la chiave, il valore R e il valore Python.
Con la chiave si risale immediatamente all'azienda e all'anno e si
riesegue il singolo caso a mano.

Il report è anche serializzato in `data/interim/reports/<checkpoint>.json`,
così due esecuzioni successive sono confrontabili e si vede se una
correzione ha migliorato o peggiorato la situazione.

### 8.4 Regole di confronto

- **Allineamento per chiave, mai per posizione.** Il confronto avviene
  dopo un join sulla chiave naturale: l'ordine delle righe non deve
  contare. Le chiavi non appaiate si contano a parte e non inquinano le
  statistiche per colonna.
- **Float.** I CSV sono stati scritti da R con precisione finita:
  l'uguaglianza bit a bit non si darà mai. Si confronta con tolleranza
  relativa `1e-9`, che distingue il rumore di serializzazione da una
  divergenza reale. Si riporta comunque `max|diff|`, perché una
  colonna "entro tolleranza" con un massimo sospetto merita
  un'occhiata.
- **Stringhe e il problema del token `NA`.** `write.csv` scrive un
  missing come `NA` non quotato e la stringa letterale `"NA"` come `NA`
  quotato, ma il parser CSV di polars rimuove le virgolette prima del
  confronto: **i due casi sono indistinguibili nel riferimento.** Non è
  teorico — `Institute` è costruita con `paste(unique(...))` che
  include i NA come testo, e una persona con soli istituti mancanti
  produce una cella esattamente uguale a `"NA"`.
  Risoluzione: le colonne stringa si confrontano in forma
  **normalizzata alla serializzazione R** — il riferimento si legge con
  `null_values=[]`, e sul lato Python i null si mappano al token `NA`
  prima del confronto. Così il confronto è ben definito da entrambi i
  lati. Il report conta a parte le celle esattamente uguali a `NA`,
  perché su quelle la verifica non può distinguere null da stringa: è
  un limite del formato del riferimento, non della pipeline, e va
  dichiarato invece che nascosto.
- **Booleani.** R serializza `TRUE`/`FALSE`; il parsing va forzato,
  non lasciato all'inferenza, per non confrontare stringhe con
  booleani e ottenere un falso 100% di divergenza.

### 8.5 Esito atteso

- **A** e **B** — coincidenza integrale in ogni caso: non dipendono dai
  deal, quindi la strategia di imputazione è irrilevante.
- **C** e **D** — coincidenza integrale con `r_injected`. Con
  `r_legacy` coincidono tutte le colonne tranne le sei di
  `TotalRaised_Est*`, per le quali il report dice quante righe
  divergono e di quanto: quella misura è il risultato utile, non un
  fallimento della verifica. Il report marca quelle sei colonne come
  divergenza attesa, così `FAIL` resta un segnale affidabile.

### 8.6 Interfaccia

- `python -m src.panel.validate --checkpoint C` esegue una verifica
  singola da riga di comando e stampa i tre livelli.
- `report.assert_clean()` solleva un'eccezione se ci sono divergenze
  non attese: è la chiamata che blocca l'avanzamento allo stadio
  successivo.
- `PanelConfig.ignore_failed_checks = True` forza l'avanzamento. Serve
  quando si vuole vedere l'effetto a valle di una divergenza nota
  invece di fermarsi al primo stadio che la produce.

## 9. Primitive R → polars

Vivono in `rutils.py`, ciascuna con un test unitario che ne verifica il
comportamento sui casi limite (input vuoto, tutto NA, un solo elemento).

| primitiva | semantica R da riprodurre |
|---|---|
| `as_na(df, cols, extra)` | il loop `%in% c("", "NA", "N/A", "NULL", ...)`; l'insieme dei valori cambia fra i punti di applicazione e va passato esplicitamente |
| `parse_date_r(col)` | `nchar == 10` → `%m/%d/%Y`; `nchar == 8` → `mdy` con `cutoff_2000 = 24`; ogni altra lunghezza → NA |
| `cumany(col)` | `cum_max` su booleano, con la stessa gestione dei null di dplyr |
| `rle_sequence(col)` | `sequence(rle(x)$lengths)`, dove ogni NA apre un run a sé |
| `weighted_cumulative(x, w)` | media ponderata cumulata sulle righe con `x` e `w` non nulli e `w > 0` |
| `quantile_type7(col, p)` | il default di `quantile()` in R, che non coincide con quello di numpy per tutti i `p` |
| `scale_r(col)` | `scale()` di R usa la deviazione standard campionaria (denominatore `n-1`); numpy e polars usano `n` di default. Su `db3` la differenza è minima ma sistematica, e va eliminata, non tollerata |
| `coalesce_first_last(col)` | `coalesce(first(x), last(x))` |
| `tail_na_omit(col)` | `tail(na.omit(x), 1)`, cioè l'ultimo non-NA nell'ordine di riga |

## 10. Rischi di fedeltà

Sono i punti in cui una traduzione plausibile diverge silenziosamente.
Ognuno va coperto da un test o da un'asserzione, non dall'attenzione di
chi scrive.

1. **Inferenza dei tipi.** R faceva `read.csv` sull'intero file; polars
   inferisce sulle prime righe. Servono schemi espliciti almeno per
   `CompanyID` (è stringa: `"100020-70"`), gli importi e gli anni. Una
   colonna inferita come intera invece che stringa cambia i join in
   silenzio.
2. **Timing della sostituzione NA.** Il loop R gira in punti precisi, e
   due volte gira *dopo* i calcoli includendo `"Inf"` e `"-Inf"`: serve
   a convertire in NA i `max()` e `mean()` su gruppi tutti-NA.
   Tradurlo come `null_values` in lettura perde questo effetto.
3. **Ordine.** Molte aggregazioni R dipendono dall'ordine di riga
   (`first`, `last`, `tail(na.omit(x), 1)`, `fill`). Servono
   `maintain_order=True` e sort espliciti, senza affidarsi all'ordine
   naturale dell'input.
4. **Null in `cum_sum`.** R propaga; polars va verificato e forzato a
   coincidere.
5. **Regex.** `grepl` usa POSIX ERE, polars usa la crate `regex` di
   Rust. Le espressioni in gioco sono semplici, ma `ignore.case = TRUE`
   va tradotto in `(?i)` e le alternanze vanno verificate una a una.
6. **`countrycode`.** La mappatura paese → continente va congelata in
   una tabella versionata nel repo: dipendere dalla libreria significa
   che una sua release futura cambia il panel.
7. **Campi quotati con newline.** `Description`, `DealSynopsis` e
   `Biography` contengono testo libero. Il conteggio righe dopo la
   lettura di ogni CSV va confrontato con quello atteso, come
   asserzione, non come controllo a vista.

## 11. Fase 2

Non fa parte di questo lavoro, ma il design lo predispone:

- misurare `fix_permanenza_media_per_company`, l'unico impatto ancora
  ignoto;
- decidere e implementare `leakage_free`;
- valutare la regola `Zero_Invested`, che forza a 0 anche `Spin-Off`,
  `Product Crowdfunding` e `University Spin-Out` — eventi di raccolta
  capitale, non uscite, e per giunta `Is_SpinOff` e `Is_CrowdFunding`
  sono entrambe feature dei modelli;
- decidere quali flag di correzione attivare, uno alla volta,
  misurandone l'effetto sul dataset finale;
- aggiungere le lavorazioni che portano a `panel.csv.gz`: competitor
  temporizzati anno per anno, `*Group`, e il filtro delle righe.
