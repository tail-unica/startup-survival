# Panel — errori logici, differenze implementative, peso morto

Data: 2026-09-10 · riverificato il 2026-09-11 · fase 2 chiusa nel leggero il 2026-09-14 · M0 corretto il 2026-09-15
Stato: documento di lavoro per la revisione fase per fase

Questo file è il registro di tutto ciò che, nella pipeline di costruzione del
panel, è **sbagliato, discutibile o inutile**. Serve alla revisione fase per
fase: quando il codice di una fase entra in `build_panel.ipynb`, si rileggono
le voci di quella fase e si decide cosa farne.

Non è un elenco di cose da correggere subito. È un elenco di cose **da
decidere**, e la decisione è di chi scrive l'articolo.

## Come si legge

Quattro categorie, con id stabili da citare nelle discussioni:

| prefisso | significato |
|---|---|
| **B1–B10** | **Difetto dell'R riprodotto di proposito**, dietro un flag `fix_*` in `PanelConfig` che vale `False`. Gli id sono quelli del registro nella spec e non vanno rinumerati. |
| **T#** | **Trappola di traduzione**: un punto in cui la semantica R e quella polars divergono. Sono **già risolte** nel codice attuale; sono qui perché una riscrittura le può reintrodurre in silenzio. Da leggere come «non toccare questo senza sapere perché è così». |
| **M#** | **Scelta metodologica discutibile**: il codice fa quello che intendeva fare, ma quello che intendeva fare è opinabile. Sono le voci che contano per l'articolo. |
| **X#** | **Candidato all'eliminazione**: calcolato e apparentemente mai usato. Marcati «da confermare» dove non ho verificato i lettori a valle. |

Dove un numero è stato **misurato** lo scrivo. Dove non l'ho misurato scrivo
«da misurare», e vale come tale: non è una stima.

## La regola di lavoro

Le correzioni **non sostituiscono** il comportamento R: aggiungono un flag.
Con tutti i flag spenti i sei checkpoint restano verdi e si può affermare di
aver riprodotto la pipeline originale; con i flag accesi si ottiene il panel
corretto, e per ciascuna correzione si può dire quante righe ha spostato.

Le **eliminazioni** sono l'eccezione: una colonna morta si cancella, non si
mette dietro a un flag. Ma serve prima la prova che nessuno la legge.

## Fin dove arriva un difetto: il cancello 2008–2017

Ogni voce di questo file dichiara **fin dove si propaga**, perché non tutto
quello che sporca il panel tocca i risultati. Il percorso è:

```
panel.parquet            882.324 righe x 117 colonne   (una riga = azienda-anno)
  └─ build_windowed_dataset
        filtri: StartingAge <= 2  e  2010 <= YearFounded + StartingAge <= 2017
  └─ preprocess_dataset
        dataset_window.csv       30.300 righe x 47 colonne  (una riga = azienda)
```

*Misurato su `data/processed/dataset_window.csv`: gli anni di fondazione
presenti vanno da **2008 a 2017**, senza eccezioni.* Un difetto che colpisce
solo aziende fuori da quella finestra sporca il panel ma **non ha toccato un
solo numero dei risultati pubblicati**. Dove rilevante, ogni voce lo dice.

Le tre destinazioni possibili, usate come etichetta nel riepilogo finale:

| etichetta | significato |
|---|---|
| **arriva alle 47** | la colonna difettosa è una feature dei modelli |
| **arriva al panel** | resta in `panel.parquet` ma muore nel preprocessing |
| **muore prima** | una fase successiva la sovrascrive o la elimina |

## Decisioni prese

**2026-09-11 — l'imputazione RandomForest è eliminata, non sospesa.**
Il modello di `1_Arrange_DB.R:1037-1119` è sbagliato alla radice (vedi M11):
nessun `set.seed`, addestrato su tutti gli anni, fittato prima di qualsiasi
split, e con il tipo di deal fra i predittori quando il tipo di deal determina
il target. **Non va portato.** Di conseguenza le sette colonne `*_Est` non
esistono più in nessun output, e la colonna che le sostituisce a valle è
**`TotalRaised`**: `src/preprocessing.py` leggerà quella al posto di
`TotalRaised_Est` una volta finita la correzione del notebook. Vedi M13.

**2026-09-14 — fase 2 del notebook leggero, corretta senza flag.** Dettaglio e
numeri in `docs/panel_revisione_stato.md`, «Registro: fase 2 del leggero».
- **B9** sostituita da una deduplica con regole esplicite: stesso incarico
  fuso scegliendo le date con `LastUpdated`, ruoli diversi fusi nell'unione dei
  periodi. Una riga per coppia.
- **B5, M4, M6** chiuse: ogni `EndDate` mancante diventa l'ultimo anno di vita
  dell'azienda. **B10** sparisce con la tabella delle aziende fallite.
- **T17** chiusa e **M3** decisa: finestre tagliate all'ultimo anno di vita,
  left join in 2b.3. Il panel coincide con lo scheletro.
- In `db3` entrano solo le aziende dello scheletro.

**2026-09-16 — M7 corretto, M8 dichiarata.** `Percent_Females` ha come
denominatore le persone di genere noto; M8 resta com'è, con i numeri misurati
nella sua voce.

**2026-09-16 — B7 corretto e pipeline ripulita.** `Institute` non contiene più il
testo `"NA"`. Il notebook leggero non legge più nessun flag `fix_*`: restano in
`PanelConfig` solo per `build_panel.ipynb`, che è congelato.

**2026-09-15 — M0 corretto nel leggero.** Esperienza e istruzione delle persone
sono ricostruite anno per anno dalle tabelle di dettaglio e agganciate dopo
l'espansione, per il team (2b.1bis) e per il CEO (5.6). Vedi il registro in
`docs/panel_revisione_stato.md`.

---

## Trasversale — la voce più importante

### M0 — gli attributi delle persone non sono temporizzati

> **Corretto nel notebook leggero il 2026-09-15**, per esperienza e istruzione,
> sia del team sia del CEO. Dettaglio, numeri ed effetto sul panel in
> `docs/panel_revisione_stato.md`, «Registro: M0 nel leggero». Resta da valutare
> il CEO ricavato dai titoli del board team, che riempirebbe gli anni senza CEO.

**Dove:** fasi 2a, 2b e 5.

La finestra di permanenza di ogni persona **è** temporizzata: `DeltaStart` e
`DeltaEnd` dicono in quali anni di vita dell'azienda quella persona era
presente, e l'aggregazione della fase 2b la rispetta. Ma gli **attributi** di
quella persona no.

`CurrentPositionsCount`, `FormerPositionsCount`, `CurrentBoardSeatsCount`,
`AffiliatedDealsCount` e compagnia sono conteggi **alla data di estrazione**,
letti da `Person.csv`. Da lì nascono `RolesCount_Total`, `Positions`,
`BoardSeats`, `OtherRoles` e `WorkExperienceIndex`, che vengono attaccati a
**ogni anno** del panel in cui quella persona compare. Lo stesso vale per
`Highest_Degree` e `Earliest_Year`.

In pratica: l'esperienza attribuita al team di una startup nel 2010 è
l'esperienza che quelle persone hanno **oggi**, comprese le posizioni assunte
dopo il 2010. Vale per le colonne di team della fase 2b (`WorkExp_Idx_Mean`,
`WorkExp_Idx_Max`, `Highest_Degree_Mean`, `Highest_Degree_Max`,
`RolesCount_Max`, `RolesCount_Mean`, `Avg_Earliest_Year`, `Positions`,
`BoardSeats`, `OtherRoles`) e per tutte e 18 le colonne `*_CEO` della fase 5.

**Propagazione: arriva alle 47.** Le feature contaminate sono **tre**:
`WorkExp_Idx_Mean`, `Highest_Degree_Mean`, `WorkExperienceIndex_CEO`. Più
`HasTop50Institute` in modo debole, perché un ateneo frequentato è quasi sempre
precedente alla startup.

Due precisazioni rispetto alla prima stesura, che ne elencava quattro:

- **`Highest_Degree_CEO` non è fra le 47.** *Misurato: è nullo sul 64,7% del
  panel finale*, quindi `handle_missing_values` lo scarta con la soglia del 40%
  di mancanti. È selezionato in `preprocess_dataset` ma non sopravvive.
- **`Avg_Earliest_Year` non è look-ahead.** È l'anno di laurea **più antico**
  della persona: guarda indietro, non avanti.

**Perché la segnalo per prima:** l'articolo è sul look-ahead bias. Questa è
informazione dal futuro che entra nelle feature, e non passa da nessuno dei
dieci bug censiti — è nel disegno della pipeline, non in un suo difetto.

**Cosa comporterebbe correggerla.** Le fonti datate esistono già
nell'estrazione, *verificato sugli header*:

```
PersonPositionRelation.csv  PersonID, EntityID, FullTitle, PositionLevel,
                            IsCurrent, StartDate, EndDate
PersonEducationRelation.csv PersonID, Degree, Major_Concentration, Institute,
                            GraduatingYear
```

Con `StartDate`/`EndDate` si contano le posizioni **attive all'anno Y** invece del
totale corrente; con `GraduatingYear` si prende il titolo più alto **conseguito
entro l'anno Y**.

> **Correzione rispetto alla prima stesura.** Qui c'era scritto che gli otto
> contatori di `Person.csv` «vanno abbandonati, non corretti» perché la tabella
> non ha date. **È falso**: ognuno si ricostruisce esattamente dalla sua tabella
> di dettaglio (`PersonPositionRelation`, `PersonBoardSeatRelation`,
> `PersonAdvisoryRelation`, `PersonAffiliatedDealRelation`,
> `PersonAffiliatedFundRelation`), che le date ce l'ha. *Verificato sulle 404.468
> persone di `db3`: i conteggi coincidono con i contatori per 404.461, e le 7
> differenze sono incoerenze interne a PitchBook.* L'intervento è quindi molto
> meno costoso di quanto stimato, ed è stato fatto il 2026-09-15.

### M1 — `GrowthStage` mescola stato attuale e storia

**Dove:** fase 5, `2_Arrange_Final.R:80-92`.

La cascata che definisce `GrowthStage` — e quindi il target — ha due fonti:

- `OwnershipStatus`, che è lo stato **attuale** dell'azienda, agganciato
  all'anno della sua `OwnershipStatusDate`;
- i flag cumulativi ricavati dai deal, che sono storici.

Nelle prime tre condizioni la prima fonte vince con un `|`. Va guardata con
attenzione: `OwnershipStatus` è nullo su quasi tutte le righe del panel (è
agganciato solo all'anno della sua data), quindi nella pratica decide poco, ma
dove decide sta proiettando lo stato di oggi su un anno passato.

*Misurato su `db_master_2`: la condizione è vera per via di `OwnershipStatus` e
falsa per via dei flag su **2.764 righe su 1.001.625**, una per azienda —
`OwnershipStatus` è valorizzato su 116.619 righe in tutto.*

**Propagazione: arriva al panel, in modo indiretto ma reale.** Quelle 2.764
righe ricevono uno stadio terminale, e alla fase 6 il troncamento taglia
l'azienda lì. *Misurato: le aziende troncate sono 39.622 in tutto, e **2.764 di
queste sono troncate su una riga decisa dal solo `OwnershipStatus`**, con
**9.620 righe eliminate**.*

**Attenuante, e va detta.** `OwnershipStatus` è agganciato all'anno della
**propria** data (`OwnershipStatusDate`), non proiettato su tutti gli anni.
Per «Out of Business» o «Acquired/Merged» quella data è, di fatto, la data
dell'evento: il salto temporale è quasi nullo. Il problema resta di principio —
una fonte di stato corrente dentro una colonna storica — ma è più piccolo di
quanto la prima stesura lasciasse intendere.

### M2 — il vintage dell'estrazione competitor

**Dove:** fase 7.

Le sei colonne competitor di `data/raw/panel.csv.gz` sono state calcolate da un
download di `CompanySimilarRelation.csv` **diverso** da quello in
`data/raw/pitchbook/`. Il checkpoint B dimostra che l'estrazione che abbiamo è
quella su cui girava l'R; eppure **655.869 righe su 84.138 aziende** hanno lo
stesso numero di concorrenti del panel pubblicato e una media di similarità
diversa, in entrambe le direzioni.

I punteggi di similarità sono output di un modello che PitchBook ricalcola fra
un download e l'altro. Conseguenza per l'articolo: il panel ricostruito e
quello pubblicato **non hanno gli stessi valori competitor**, e i risultati
pubblicati non sono riproducibili su quelle tre feature con i dati presenti.

---

## Fase 1 — aziende, affiliate, scheletro del panel

`1_Arrange_DB.R:37-238` · `src/panel/stage1_company.py`

### Difetti riprodotti

**B6 — `seq()` conta all'indietro** (`fix_negative_delta`).
Se `MaxYear < YearFounded`, `seq(2010, 2008)` produce `2010, 2009, 2008` e
quindi `Delta` **negativi**: anni-azienda precedenti alla fondazione.

`MaxYear` è il più recente fra sei date disponibili in `Company.csv`. Per un
pugno di aziende i dati sono sporchi — per esempio una data di bilancio del
2008 su un'azienda registrata come fondata nel 2010 — e la sequenza parte
dopo il suo punto d'arrivo.

**Propagazione: arriva al panel, non arriva alle 47.** *Misurato lungo la
catena:*

| tappa | righe con età < 0 | aziende |
|---|---:|---:|
| scheletro (fase 1) | 245 | 106 |
| `panel.parquet` (fase 7) | **186** | **90** |
| `dataset_window.csv` | 0 | 0 |

*Su tutte le 186 righe `Total_People` e `GrowthStageGroup` sono nulli*, quindi
il preprocessing le scarta. Impatto nullo sui modelli; nel panel restano 186
righe che affermano che un'azienda esisteva prima di essere fondata, e
qualunque statistica descrittiva sul panel le include.

**La correzione dietro il flag** sostituisce la sequenza all'indietro con
**una riga sola**, quella dell'anno di fondazione, invece di scartare
l'azienda: quelle 90 aziende restano nel panel con un panel lungo 1.

### Trappole di traduzione

**T1 — il momento della sostituzione NA.**
`nchar(x) > 0` gira **dopo** la sostituzione dei placeholder, quindi un URL
mancante dà **null**, non `False`. *Misurato: `Website_d` vale `True` su
107.256 aziende e null su 9.664; `False` mai.* Tradurlo come «se manca allora
False» cambia 9.664 righe.

**T2 — `pmax(..., na.rm = TRUE)` su sei anni.**
Se tutte e sei le date mancano il risultato è NA e l'azienda viene **scartata
dallo scheletro**. Non è un errore: è il filtro che decide chi entra nel panel.

**T3 — `sub()` senza match restituisce la stringa originale.**
`Quarter` nasce da `sub("TTM (\\dQ)(\\d{4})", "\\1", FiscalPeriod)`: per un
valore non riconosciuto non dà NA, dà la stringa intera. `Month` diventa NA
solo perché nessun `case_when` la riconosce. `FiscalDate` è `Year-Month-30`.

**T4 — il parsing delle date ha una regola a due rami.**
`nchar == 10` → `%m/%d/%Y`; `nchar == 8` → `mdy` con `cutoff_2000 = 24` (anni
`00`–`24` → 2000-2024, `25`–`99` → 1925-1999); **ogni altra lunghezza → NA**.
*Misurato: in `Company.csv` tutte le date hanno 10 caratteri, quindi il secondo
ramo non scatta mai su questa estrazione.* La regola è latente, non attiva: va
tenuta perché un'estrazione futura potrebbe attivarla.

**T5 — `db1.parquet` esiste per una ragione precisa.**
Alla riga 448 l'R aggancia `YearFounded` alla tabella del team prendendolo da
`db1`, cioè dalla versione **non filtrata**. Usare `db_master_1` (filtrato
`> 1999`) farebbe sparire le persone delle aziende più vecchie e cambierebbe
`db3`.

### Scelte discutibili

**M3 — `MaxYear` come limite del panel.** — *decisa il 2026-09-14: il panel si ferma a `MaxYear` anche per il team; allargarlo con le date del board e dei deal resta un'alternativa non adottata*
`MaxYear` è l'anno più recente fra sei date disponibili, cioè **l'ultimo anno
con dati**. Il panel di un'azienda finisce lì. Un'azienda ben coperta da
PitchBook ha quindi un panel più lungo di una coperta male, a parità di vita
reale. (Alla fase 7 lo stesso `MaxYear` viene riusato come proxy di «azienda
viva», dove il problema è più grave: vedi M12.)

### Candidati all'eliminazione

**X1 — `D_OS_FFD` e `D_OS_YF`.** L'R le calcola alla riga 97 e non le usa mai:
non entrano in `db_master_1`. **Nella traduzione non le ho portate**: è una
differenza implementativa deliberata da confermare.

**X2 — colonne di `db_master_1` che non arrivano a `db_final`.**
`db_final` prende 22 colonne da `db_master_1`. Non ci arrivano:
`CompanyLegalName`, `HQPostCode`, `OwnershipStatusDate`, `ParentCompanyID`,
`ActiveInvestors`, `FormerInvestors`, `Website` (la stringa; resta `Website_d`),
`Affiliated`, `N_Parent`, `N_Sister`, `N_Subsidiary`.
*Da confermare* che nessuna serva ad altro.

**X3 — `Affiliated` duplica `N_Affiliated > 0`.** È lo stesso dato in due
forme.

**X4 — colonne lette da `Company.csv` e usate solo per `MaxYear`.**
`CompanyFinancingStatusDate`, `BusinessStatusDate`, `LastKnownValuationDate`,
`FirstFinancingDate` e `FiscalPeriod` servono a calcolare `MaxYear` e i cinque
join per anno, poi sparisco­no. `FirstFinancingSize`,
`FirstFinancingValuation`, `FirstFinancingDealType`, `FirstFinancingDebt`,
`FirstFinancingDebtSize` sono selezionate e **mai usate**.

---

## Fase 2a — la tabella persona-azienda (`db3`)

`1_Arrange_DB.R:240-524` · `src/panel/stage2_team.py` · **CHECKPOINT A**

### Difetti riprodotti

**B2 — `Is_Other` cerca un valore che non esiste** (`fix_is_other_label`).
Confronta `Field` con `"Other/Unknown"`, mentre `Field` produce `"Other"` e mai
`"Other/Unknown"`. *Misurato: 0 righe vere su tutto `db3`.* La colonna è
inutilizzabile. Non è fra le 47 feature, quindi non ha contaminato i risultati
pubblicati.

**B5 — `PermanenzaMedia` è un solo numero globale** — *chiusa il 2026-09-14 nel notebook leggero, vedi «Decisioni prese»*
(`fix_permanenza_media_per_company`). L'R la calcola con un `summarise` senza
`group_by`, nonostante il commento dichiari «per ciascuna CompanyID»: è la
permanenza media su tutto il dataset, usata per imputare l'`EndDate` di
chiunque. *Misurato: quel numero vale **7 anni** (media grezza 6,606065,
calcolata sulle 124.277 righe che hanno entrambe le date **prima**
dell'imputazione).* Ogni persona di cui non si sa quando ha lasciato l'azienda
riceve «entrata + 7 anni», che sia una startup morta in due anni o una
sopravvissuta quindici.

*Misurato: la regola scatta su **115.399 righe, il 21,6% di `db3`**.* Non e'
un ritocco marginale: e' una riga su cinque la cui finestra di presenza e'
inventata da un solo numero.

Quella data decide **in quali anni una persona viene contata nel team**, quindi
tocca tutte le colonne della fase 2b. *Misurato: con la media per azienda il
`DeltaEnd` medio passa da 11,57 a 12,04 anni, le righe con dati di team da
885.142 a 892.534 (+0,84%), `Total_People` +0,91%, `Total_Founders` +1,23%.*
Sistematico ma piccolo.

> **⚠ Il ramo corretto ha un difetto suo: l'arrotondamento.**
> `round()` di R arrotonda **al pari** (`round(7.5)` è 8, `round(8.5)` è 8);
> `.round(0)` di polars arrotonda per eccesso (8 e 9). Con la media globale non
> si nota, perché 6,61 non è un pareggio. Con la media **per azienda** i
> pareggi diventano normali — due persone con 7 e 8 anni fanno esattamente 7,5.
> Va sistemato **prima** di accendere il flag, altrimenti il ramo corretto
> arrotonda diversamente dal ramo di riferimento senza farlo notare.

**B9 — due anomalie nella deduplica** (`fix_dup_coalesce`) — *chiusa il 2026-09-14 nel notebook leggero, vedi «Decisioni prese»*.
I duplicati si riselezionano filtrando per `PersonID` invece che per la coppia
`(CompanyID, PersonID)`, e `coalesce(first(x), last(x))` non vede un valore
presente solo in una riga intermedia.

*Misurato sull'estrazione (2026-09-11):* 535.568 righe grezze, 534.851 coppie
uniche, **713 coppie duplicate** su 1.430 righe.

- **Anomalia 1 — il filtro per `PersonID`.** Tira dentro all'aggregazione anche
  le righe di una persona duplicata *in un'altra azienda*: **2.175 righe invece
  di 1.430, cioè 745 elaborate per niente**. *Verificato isolandola: il
  contenuto di `db3` è **identico**; cambia solo l'**ordine** delle righe —
  1.171 coppie su 272 aziende.*

  Quell'ordine arriva alla fase 2b, dove `Institute` è un `paste(unique(...))`.
  *Misurato: la stringa `Institute` cambia su **155 aziende**, e in **tutte e
  155 cambia solo l'ordine**: l'insieme degli atenei è identico.* Il solo
  consumatore di `Institute` è `get_top50_institute_flag`, che fa `split(";")`
  e cerca per appartenenza: **l'ordine non porta informazione e il flag
  `HasTop50Institute` non cambia mai**.

  Quindi l'anomalia 1 è **inerte per i dati**. L'unica cosa che la rende non
  rimovibile è che i checkpoint C–F confrontano `Institute` **come testo**
  contro l'export R: toglierla li farebbe diventare rossi su quella colonna
  senza che nulla di sostanziale sia cambiato. È un vincolo di prova di
  fedeltà, non un vincolo di correttezza. Quando la fedeltà smette di essere
  l'obiettivo, l'anomalia 1 si toglie senza pensarci.
- **Anomalia 2 — `coalesce(first, last)`.** Serve almeno tre righe per la
  stessa coppia. *Misurato: 709 coppie hanno 2 righe, **solo 4 ne hanno 3**.*

*Effetto totale della correzione, verificato eseguendo i due rami e
diffandoli: **3 celle su una sola riga**, su 534.851.* Una sola coppia,
`186214-78 / 57142-36P`:

| colonna | valori nelle tre righe | R tiene | corretto |
|---|---|---|---|
| `RepresentingName` | `[null, "Self", null]` | `null` | `"Self"` |
| `RoleOnBoard` | `[null, "Board Member", "CFO & Board Member"]` | `"CFO & Board Member"` | `"Board Member"` |
| `StartDate` | `[null, 2023-01-01, 2024-01-01]` | `2024-01-01` | `2023-01-01` |

**Propagazione: una cella.** `RepresentingName` non è fra le nove colonne che
`db3` seleziona. `RoleOnBoard` lo è ma non è letta da nessuno (X5). Resta
`StartDate`, che guida `DeltaStart`: **una persona, in un'azienda, entra nel
team un anno dopo.** E sulle altre due la versione «corretta» non è nemmeno
ovviamente migliore — `"CFO & Board Member"` è più informativo di
`"Board Member"`.

**Nessuna modifica al codice.** Il flag fa già la cosa giusta; quello che
mancava era sapere che l'effetto è di tre celle.

**B10 — `Is_Out` resta NA per le aziende non fallite** (`fix_is_out_na`) — *chiusa il 2026-09-14 nel notebook leggero, vedi «Decisioni prese»*.
Dopo il left join con `ownership_out`, la condizione `Is_Out == FALSE` vale NA
e **neutralizza** una delle imputazioni di `EndDate`. *Misurato: 471.770 righe
con `Is_Out` nullo.* Un catch-all successivo recupera i casi, quindi il bug si
auto-sana.

### Trappole di traduzione

**T6 — `if_else` di dplyr restituisce NA se la condizione è NA.**
La più insidiosa di tutte. Alla riga 452 la condizione è
`is.na(StartDate) | year(StartDate) < YearFounded`: per un'azienda senza
`YearFounded` il confronto vale NA, l'intera condizione vale NA, e `if_else`
**cancella una `StartDate` perfettamente valida**. `pl.when` tratterebbe la
condizione nulla come falsa e terrebbe il valore. *Misurato: 5.357 righe di
`db3`; 11.264 righe appartengono ad aziende senza `YearFounded` e su tutte la
`StartDate` risulta cancellata.* Risolta con `rutils.r_if_else`.

**T7 — `scale()` di R usa la deviazione standard campionaria** (denominatore
`n-1`); numpy e polars usano `n`. E la standardizzazione è **globale su tutta
`db3`**, non per azienda: rifarla per gruppo darebbe numeri completamente
diversi.

**T8 — `paste` con NA produce la stringa `"NA"`.**
`IsFounder` nasce da
`str_detect(paste(FullTitle, PositionLevel, sep = "; "), "Found")`: siccome
`paste` stringifica i mancanti, la stringa non è mai NA e `IsFounder` non è mai
nullo. *Misurato: 0 nulli, 218.790 founder.*

**T9 — le catene `case_when` sono a corto circuito.**
La classificazione del titolo di studio e quella del campo sono sequenze di
`grepl`: il primo che matcha vince. **L'ordine dei test è vincolante** e
riordinarli cambia le classificazioni. Esempio concreto: `"MD"` compare nella
prima regola (`PhD/Doctorate`) e `"Master"` nella seconda, quindi un `"MD"`
non arriva mai a `Master's`.

**T10 — `DegreeLevel` non è mai NA.**
Il ramo finale `TRUE ~ "Other"` cattura anche un `Degree` mancante, quindi la
guardia `ifelse(sum(!is.na(DegreeLevel)) > 0, ...)` è sempre vera. Una persona
senza titolo dichiarato riceve `Highest_Degree = 1` («Other»), non NA.

**T11 — `paste(na.omit(Institute))` qui *rimuove* i NA.**
Nell'aggregazione per persona di questa fase i mancanti vengono scartati. È
nell'aggregazione della fase 2b che vengono invece inclusi come testo `"NA"`
(bug B7). Due `paste` diversi a due righe di distanza, con comportamenti
opposti.

**T12 — gli override dal nome sono case-sensitive.**
`grepl(" JD", PersonName)` e `grepl(" MD", PersonName)` sono senza
`ignore.case`, a differenza di tutte le altre `grepl` del blocco.

**T13 — `rowSums(..., na.rm = TRUE)` su una riga tutta mancante dà 0, non NA.**
Una persona assente da `Person.csv` ottiene quindi `Positions = 0`,
`BoardSeats = 0`, `OtherRoles = 0` — indistinguibile da una persona
effettivamente senza posizioni.

**T14 — il join con `db1` è un *natural join*.**
`db3 %>% left_join(db1[,c("CompanyID","YearFounded")])` è senza `by=`: dplyr
unisce su tutte le colonne in comune. Qui è solo `CompanyID`, ma è fragile: se
`db3` avesse anche `YearFounded`, il join cambierebbe da solo.

### Scelte discutibili

**M4 — `EndDate` imputata a `2024-12-31`.** — *chiusa il 2026-09-14 nel notebook leggero, vedi «Decisioni prese»*
Chi ha `IsCurrent == "Yes"` riceve come fine permanenza una data fissa legata
al **vintage dell'estrazione**. Rieseguire la pipeline su un download del 2026
darebbe finestre diverse per le stesse persone.

**M5 — `DeltaStart = 0` per i founder** — *decisa il 2026-09-16: si tiene.*
Sovrascrive la data di inizio reale: si assume che un founder ci sia dal primo
giorno. Ragionevole come euristica, ma cancella un dato che in alcuni casi
c'era. *Misurato con la pipeline di oggi: 205.884 coppie founder; fra quelle con
una data d'ingresso vera, 17.834 (8,9%) risultano entrate dopo la fondazione, in
media 3 anni.*

> **Corretto invece il riconoscimento del founder.** L'R cerca `Found` nella
> qualifica, quindi prende anche "Foundation", "Foundry" e gli assistenti
> ("Founder's Associate"). Nel leggero si cerca `founde|founding` dopo aver
> tolto la frase degli assistenti: **42 coppie perdono il flag**, `Total_Founders`
> cala di 252 anni-persona.

**M6 — `PermanenzaMedia` come imputazione.** — *chiusa il 2026-09-14 nel notebook leggero, vedi «Decisioni prese»*
Anche corretta per azienda (B5), imputare la fine di una permanenza con una
media è un'assunzione forte, e determina **in quali anni una persona viene
contata**: tocca tutte le feature di team.

### Candidati all'eliminazione

**X5 — colonne di `db3` mai usate a valle.**
`db3` ha 53 colonne. Alla fase 5 ne servono 18 per il CEO; alla fase 2b una
dozzina per l'aggregazione. Sembrano non servire a nulla: `Prefix`, `City`,
`PostCode`, `Country`, `IsOnBoard`, `RoleOnBoard`, `PositionLevel` (serve solo
a costruire `IsFounder`), `OwnershipStatus`, `OwnershipStatusDate`, `OutDate`,
`RolesCount` (diverso da `RolesCount_Total`).

**X6 — `Biography`.** Campo di testo grande, letto da `Person.csv`, portato in
`db3` e **mai usato**. La spec originale lo cita come «azione potenziale».

**X7 — `Positions_z`, `BoardSeats_z`, `OtherRoles_z`.** Intermedie: servono
solo a calcolare `WorkExperienceIndex`, poi restano in `db3`.

**X8 — `MBA` e `Major_Concentration`.** `MBA` è calcolata in `db48` e mai
usata. `Major_Concentration` e `Field` sono intermedie della classificazione.

---

## Fase 2b — le colonne di team, anno per anno

`1_Arrange_DB.R:527-664` · `src/panel/stage2_team.py`

### Difetti riprodotti

**B1 — la soglia di fondazione cambia** (`fix_founding_year_threshold`).
La fase 1 filtra `YearFounded > 1999` (`1_Arrange_DB.R:236`); questa fase
(`:566`) e la fase 4 (`:935`) filtrano `YearFounded > 2000`. Risultato:
l'intera coorte di aziende fondate nel **2000** entra nel panel come un guscio
vuoto — nello scheletro sì, con i dati di team e i deal no.

*Misurato sul panel finale:*

| coorte | righe | aziende | con dati di team | con `GrowthStage` |
|---|---:|---:|---:|---:|
| **2000** | 24.430 | 1.443 | **0,0%** | **0,0%** |
| 2001 | 19.745 | 1.186 | 91,1% | 58,8% |
| 2002 | 19.017 | 1.175 | 91,8% | 58,4% |

Non è solo il team: senza deal non c'è niente che possa accendere
`GrowthStage`. Sono **24.430 righe, il 2,77% del panel**, che non dicono nulla.

> **Propagazione: muore prima delle 47. Correzione rispetto alla prima
> stesura.** Questa voce era classificata come «il difetto più grave del
> registro», con la motivazione che `preprocessing.py` scarta le righe a
> `Total_People` nullo e la coorte sparisce dal dataset. È vero che sparisce,
> ma **sarebbe sparita comunque**: il cancello 2008–2017 di
> `build_windowed_dataset` esclude la coorte 2000 a monte. **B1 non ha toccato
> nessun risultato pubblicato.** Resta un difetto di completezza del panel, e
> diventa bloccante solo allargando la finestra temporale.

*Correggendolo si recuperano 1.386 aziende su 1.443* (le altre 57 non hanno
comunque nessuna persona in `CompanyBoardTeamRelation.csv`).

**B7 — `paste(unique(Institute))` include i NA come testo**
(`fix_institute_na_literal`) — *corretto il 2026-09-16 nel notebook leggero*.
*Misurato: 390.544 righe del riferimento hanno un `Institute` che comincia con
`"NA; "`.* Cosmetico: non altera il flag «ateneo fra i primi 50», che cerca nomi
di università. Nel leggero i mancanti ora si scartano, come in 2a.5: le righe con
`"NA"` fra gli atenei passano da 343.460 a 0 e nessun'altra colonna si muove.

### Trappole di traduzione

**T15 — `max()` su un gruppo tutto-NA dà `-Inf`, `mean()` dà `NaN`.**
Il ciclo di sostituzione NA della riga 650 include anche `"Inf"` e `"-Inf"`, e
li converte in NA veri. **È per questo che quel ciclo gira dopo
l'aggregazione**: spostarlo prima perde l'effetto.

**T16 — le righe fantasma non esistono, e non per fortuna.**
L'aggregazione conta le persone con `.N`, che conterebbe anche la riga vuota
prodotta da un join senza corrispondenze, dando `Total_People = 1` con tutti i
campi della persona nulli. Non succede perché `YearFounded` arriva dal lato
persona: per un anno-azienda che non ha trovato nessuno vale NA, e il filtro
sull'anno di fondazione elimina la riga **prima** che l'aggregazione la veda.

> **Correzione rispetto alla prima stesura.** Qui c'era scritto che correggere
> B1 avrebbe rimosso questa protezione e fatto ricomparire le righe fantasma.
> **È falso.** La protezione non dipende dal *valore* della soglia: dipende dal
> fatto che `YearFounded` è **null** sulla riga non abbinata, e sia in R
> (`filter`) sia in polars un confronto `> qualcosa` scarta i null comunque.
> *Verificato eseguendo `expand_team` con entrambe le soglie sullo stesso
> sottoinsieme: `> 2000` dà 90.431 righe espanse e **0 fantasma**, `> 1999` ne
> dà 198.357 e **0 fantasma**.* Correggere B1 è sicuro su questo fronte.

**T17 — `full_join`, non `left_join`.** — *chiusa il 2026-09-14 nel notebook leggero, vedi «Decisioni prese»*
Il panel del team può contenere anni-azienda che lo scheletro non ha. È così
che si arriva alle 1.001.625 righe: con un left join sarebbero meno.

**T18 — `unnest(Years, keep_empty = TRUE)`** tiene un'azienda senza finestra
utilizzabile come **una riga con `Years = NA`**, che poi non si aggancia a
niente.

### Scelte discutibili

**M7 — `Percent_Females` ha come denominatore tutte le persone** — *corretta il
2026-09-16 nel notebook leggero.*
Usa `.N`, cioè conta anche le persone con genere ignoto. Se il genere è
sconosciuto per metà del team, la quota di donne è **diluita verso il basso**
invece di essere calcolata sui soli casi noti. `Percent_Females` è una delle 47
feature.

> **Corretta.** Il denominatore sono ora le persone di genere noto, e dove
> nessuno ha un genere noto la colonna resta vuota, perché la quota non è
> definita. *Misurato sul panel: cambia su 8.582 righe, in media di 5,33 punti;
> la media della colonna passa da 14,217 a 14,288; le righe valorizzate scendono
> da 749.892 a 749.093. Nessun'altra colonna si muove.* La distorsione non era
> casuale: colpiva le aziende con i team documentati peggio.

> **Declassata.** Era «da misurare», ed era in nona posizione nel riepilogo.
> *Misurato su `db3`: `Gender` è nullo su **3.012 righe su 534.851 = 0,6%**
> (449.943 Male, 81.896 Female).* La diluizione esiste su mezzo punto
> percentuale di casi. **Fuori dal riepilogo per priorità**: resta qui come
> nota di trasparenza, non come cosa da correggere.

**M8 — founder e chiunque altro pesano uguale** — *decisa il 2026-09-16: si
dichiara nell'articolo, non si corregge.*
L'aggregazione mette sullo stesso piano i founder e ogni altro membro del board
team. `Total_Founders` li conta, ma gli indici di esperienza e di istruzione
sono medie su tutti.

> **Misurato il 2026-09-16.** I founder sono **1.544.565 anni-persona su
> 3.710.269 (41,6%)**, e hanno *meno* esperienza degli altri (indice medio 0,083
> contro 0,199), perché fra i non founder ci sono investitori e dirigenti esterni
> con molte cariche. Il titolo di studio è simile (3,82 contro 3,80), la quota di
> donne più bassa (11,6% contro 16,7%). Calcolando le medie sui soli founder, la
> media di team passerebbe da 0,007 a 0,046, con correlazione 0,725 rispetto a
> oggi, e resterebbe **vuota nel 14,5% degli anni-azienda**, quelli senza nessun
> founder presente. Non è un errore ma una definizione: la feature misura il
> vertice nel suo complesso, e va detto nell'articolo.

Vedi anche **M0**: gli attributi aggregati qui non sono temporizzati.

### Candidati all'eliminazione

**X9 — `RolesCount_Mean`, `Highest_Degree_Max`, `WorkExp_Idx_Max`,
`Positions`, `BoardSeats`, `OtherRoles`.** Nel panel; *da confermare* quali
arrivino alle 47 feature. `Highest_Degree_Mean` e `WorkExp_Idx_Mean` ci
arrivano.

**X10 — `Is_Other`** (già morta per B2).

---

## Fase 3a — le colonne competitor

`1_Arrange_DB.R:681-708` · `src/panel/stage3_relations.py` · **CHECKPOINT B**

> **⚠ Leggere prima di lavorare su questa fase: le sue sei colonne competitor
> non arrivano al panel.** La fase 7 ne butta tre e le ricalcola anno per anno,
> e ne elimina due:
>
> ```python
> finale = panel.drop("N_Competitors", "Same_Country", "SimilarityScoreMean",
>                     "N_Europe", "N_Outside_Europe")   # buttate
>              .join(stat_concorrenti, ...)             # ricalcolate
> ```
>
> *Verificato su `panel.parquet`: 117 colonne, `N_Europe` e `N_Outside_Europe`
> assenti, `Same_Country` con **zero nulli**.* Sopravvive alla fase 3a solo
> `SimilarityScoreMax`. Quindi **tutti i difetti di questa fase muoiono
> prima delle 47**, e la fase esiste soltanto per far passare il checkpoint B.

### Difetti riprodotti

**B4 — `any()` senza `na.rm`** (`fix_same_country_narm`).
`Same_Country` restituisce NA quando nessun confronto è vero *e* almeno uno è
mancante, invece di `False`. *Misurato: 1.809 aziende, l'1,55% di quelle con un
aggregato competitor.*

> **Declassato.** Era in sesta posizione con la motivazione «`Same_Country` è
> una delle 47 feature». La feature esiste, ma è **quella della fase 7**, che
> ha anche un significato diverso (vedi M21): era un booleano, è diventata un
> conteggio. La colonna con il bug viene buttata. **Propagazione: muore
> prima.** Fuori dal riepilogo per priorità.

**B8 — `N_Europe` e `N_Outside_Europe` non sono simmetriche**
(`fix_europe_asymmetry`). `N_Europe` somma su **tutte** le righe,
`N_Outside_Europe` solo su quelle con similarità sopra 90. Non sono due facce
dello stesso conteggio. **Propagazione: muore prima** — entrambe le colonne
sono eliminate alla fase 7. Fuori dal riepilogo per priorità.

### Trappole di traduzione

**T19 — tre aggregati senza `na.rm`.**
`mean(SimilarityScore)`, `max(SimilarityScore)` e `sum(IsCompetitor == "Yes")`
diventano NA se *un solo* valore manca. In polars gli aggregati saltano i null
per default e darebbero un numero. *Su questa estrazione non scatta mai*, ma il
comportamento è riprodotto.

**T20 — il subset con indice NA produce elementi NA, non li elimina.**
`SimilarCompanyHQCountry[SimilarityScore > 90]` con uno score mancante produce
un **elemento NA nel sottoinsieme**, che arriva a `any()`. È il meccanismo che
genera B4.

**T21 — `countrycode` ha bisogno di un terzo stato.**
Restituisce NA per un nome che non sa collocare, e NA **non è `False`**:
`N_Europe` non conta né l'uno né l'altro, ma `N_Outside_Europe` conta solo
`False`. Quattro nomi cadono in questo caso — **Kosovo, Polynesia, Micronesia,
British Indian Ocean Territory** — e trattarli come «non europei» sbagliava
`N_Outside_Europe` su 44 aziende. La mappa è stata **ricavata dai dati** e
congelata in `src/panel/data/europe.csv` per non dipendere da una libreria che
può cambiare classificazione fra due release.

### Scelte discutibili

**M9 — `HQCountry` viene dalla tabella già filtrata.**
`Same_Country` confronta il paese del concorrente con `HQCountry` preso da
`db_master_1`, che è filtrato `YearFounded > 1999`. Per le aziende fuori dal
panel `HQCountry` è NA, quindi il confronto è NA **per costruzione**.

**M10 — la soglia 90 è arbitraria e usata in modo incoerente.**
`Same_Country` e `N_Outside_Europe` la applicano, `SimilarityScoreMean`,
`N_Competitors` e `N_Europe` no.

### Candidati all'eliminazione

**X11 — `N_Europe` e `N_Outside_Europe`.** Sono **eliminate alla fase 7** e non
sono feature dei modelli. *Verificato: assenti da `panel.parquet`.*
Conseguenza: **tutta la derivazione di `src/panel/data/europe.csv` e
`scripts/derive_europe_mapping.py` esiste solo per far passare il checkpoint
B.** È il candidato più netto della lista: se la fedeltà non è più l'obiettivo,
spariscono ~100 righe di codice e un file di dati. Insieme a X11 cadono anche
B4 e B8, che sono difetti di colonne che nessuno legge.

**X12 — `SimilarityScoreMax`.** Nel panel; *da confermare* se arriva alle 47.

---

## Fase 3b — dipendenti, financials, news

`1_Arrange_DB.R:710-813` · `src/panel/stage3_relations.py`

### Trappole di traduzione

**T22 — «ultima rilevazione dell'anno» dipende dall'ordinamento.**
`arrange(CompanyID, anno, desc(Date))` più `distinct` tiene la prima riga di
ogni anno, cioè la rilevazione più recente. Le date mancanti vanno in coda.

**T23 — i financials riempiono, non sovrascrivono.**
`if_else(is.na(Revenue), Revenue_db8, Revenue)` è un `coalesce`: i bilanci che
la fase 1 aveva già agganciato da `Company.csv` **vincono**. *Misurato: 0 celle
sovrascritte.*

**T24 — `N_News` è 0 dove il join non trova niente**, non NA.

**T25 — `left_join` di dplyr abbina NA a NA**, polars no.
Riguarda la riga con `Year_Delta` nullo (azienda `807550-48`): in R un
`EmployeeCount` con anno mancante potrebbe agganciarsi a quella riga, in polars
no. Effetto su 1 riga.

### Candidati all'eliminazione

**X13 — `N_News`.** *Misurato: l'estrazione contiene 1.858 news su 371 aziende;
sul panel finale `N_News > 0` su **97 righe su 882.324 = 0,011%**.* È di fatto
una costante e non può portare informazione. Non è nemmeno fra le 47.
Candidata certa all'eliminazione, non «forte».

**X14 — `EmployeeCount`.** *Misurato: valorizzata su 300.371 righe su
1.001.625, il 30%.* **Non è peso morto**: la voce si chiude qui. Da 600.805
rilevazioni grezze la deduplica per anno ne tiene 392.055.

---

## Fase 4 — deal e investitori

`1_Arrange_DB.R:826-1251` · `src/panel/stage4_deals.py`

### Difetti riprodotti

**B1 — di nuovo la soglia `> 2000`** (`fix_founding_year_threshold`).
Stesso flag della fase 2b: i deal della coorte 2000 sono esclusi.

### Trappole di traduzione

**T26 — `pmax(year(DealDate), YearFounded)` non ha `na.rm`.**
Un deal che non ha mai ottenuto una data mantiene quindi `Year_Delta = NA`,
finisce in un gruppo nullo e **esce dal panel** al join.
`pl.max_horizontal` ignora i null e restituirebbe l'anno di fondazione,
parcheggiando quei deal sull'anno zero dell'azienda e **inventando deal in
7.335 anni-azienda**. *Misurato: 13.184 gruppi azienda con `Year_Delta` nullo
in `deals_panel`.* Era l'unica divergenza di questa fase al primo tentativo.

**La trappola è risolta, ma il comportamento riprodotto resta discutibile: vedi
M25.** Nessuno dei due comportamenti — scartare il deal (R) o parcheggiarlo
sull'anno zero (polars ingenuo) — è quello giusto.

**T27 — tre semantiche di missing sullo stesso numero.**
`TotalRaised` è `sum(na.rm = TRUE)` (un importo ignoto vale zero);
`TotalRaised_NA` è NA se **tutti** gli importi mancano; `TotalRaised_any` è NA
se **almeno uno** manca. Le tre `_Est` corrispondenti erano le versioni con
l'imputazione RandomForest e non vengono più prodotte.

**T28 — gli aggregati per investitore sono condizionati senza `na.rm`.**
Sono gestiti da `ifelse(any(InvestorStatus == "New Investor"), ..., NA)`, e
`any()` senza `na.rm` vale NA quando nessuno corrisponde e qualcosa manca:
l'aggregato diventa NA.

**T29 — asimmetria fra il blocco «nuovi investitori» e quello «lead».**
Il blocco lead filtra su `IsLeadInvestor == "Yes"` ma **conserva la condizione
sui nuovi investitori**. L'asimmetria è nell'R ed è riprodotta.

**T30 — `TR_D` con tre varianti invece di sei.**
L'R testa se tutte e sei le varianti di `TotalRaised` sono NA; con tre il
risultato è identico, perché le `_Est` erano nulle esattamente dove sono nulle
queste. Riprodotto, ma è una equivalenza da tenere a mente se si cambiano le
varianti.

**T31 — `tbl1` elenca solo i `DealType` che appaiono con importo mancante.**
Quindi `cat_other` (i tipi accorpati in `"Other"`) considera solo quelli, non
tutti i tipi di deal.

### Scelte discutibili — è la fase con più problemi

**M11 — l'imputazione RandomForest. ELIMINATA (decisione del 2026-09-11).**
`1_Arrange_DB.R:1037-1119` addestra un `randomForest(ntree = 50)` **senza
`set.seed`**: non è riproducibile nemmeno rieseguendo l'R. Ed è una fonte di
look-ahead in un lavoro che denuncia il look-ahead:

- **temporale**: addestrata su deal di tutti gli anni, imputa un deal del 2013
  usando pattern del 2020, con `Age` fra i predittori;
- **train/test**: fittata sull'intero dataset prima di qualsiasi split;
- **prossimità al target**: `DealTypeGrouped` è un predittore, ma i tipi di
  deal determinano `GrowthStage`, cioè il target.

*Misurato sui dataset pubblicati: nel dataset con finestra temporale 5.914
righe su 30.300 (**19,5%**) avevano un valore imputato, pari al **29,7% della
massa della feature**; senza finestra circa 9.739 aziende su 30.300 (32,1%).*
*Misurato a livello deal: le righe che soddisfano i criteri dell'R — importo
mancante e sinossi che parla di importo non dichiarato — sono **53.975 su
332.818**, prima del filtro sui predittori completi. La spec ne stimava «al
massimo 48.000» come limite superiore, ed era una sottostima.*
**Non è portata, e non va portata.** Non è più una sospensione in attesa di
valutazione: il modello è sbagliato alla radice per i tre motivi qui sopra, e la
decisione è di eliminarlo. Le sette colonne che creava —
`TotalInvestedCapital_Est`, `TotalRaised_Est`, `TotalRaised_Est_NA`,
`TotalRaised_Est_any` e le tre cumulate — **non esistono in nessun output**.

*Verificato: `data/raw/panel.csv.gz` (pubblicato) ha 122 colonne e le sei
`*_Est`; `data/interim/panel.parquet` (ricostruito) ne ha 117 e non le ha.*
I checkpoint C–F le dichiarano come `expected_missing` e restano verdi.

Conseguenza operativa: cadono con la RF anche X15 (`UndisclosedAmountFlag`) e
X16 (`DealSynopsis`), che servivano solo a selezionare le righe da imputare.

**M12 — la regola `Zero_Invested`.**
I tipi di deal in cui l'importo manca in oltre il 90% dei casi ricevono 0
invece di NA. È una regola derivata da **statistiche globali su tutto il
dataset**: stessa famiglia di M11, più mite. È **mantenuta** perché alimenta
`TotalRaised` e perché «questo tipo di deal non dichiara mai l'importo, quindi
è zero» non è un'imputazione modellistica. Da decidere se tenerla.

**M13 — `TotalRaised` confonde tre situazioni diverse. DECISA.**
Vale 0 sia quando non c'è stato nessun deal, sia quando c'è stato un deal a
importo zero, sia quando c'è stato un deal di importo **non dichiarato**.
`TR_D` distingue il primo caso; `TotalRaised_NA` e `TotalRaised_any` il terzo.

> **Decisione del 2026-09-11: la colonna che sostituisce `TotalRaised_Est` in
> `src/preprocessing.py` è `TotalRaised`.** La sostituzione si fa **dopo** aver
> finito di correggere il notebook, in un passaggio solo, perché comporta
> rigenerare `data/processed/` e tutti i run.
>
> Il costo della scelta, dichiarato: `TotalRaised` scrive **0** dove l'importo
> non è dichiarato, quindi «non ha raccolto niente» e «non sappiamo quanto»
> diventano lo stesso numero. `TotalRaised_NA` avrebbe lasciato un nullo, e
> l'imputazione che gira prima del training avrebbe visto il buco. Da citare
> nell'articolo se si discute la distribuzione di questa feature.
>
> **Resta aperto**, e si decide alla fase 4: se tenere `TotalRaised_NA` e
> `TotalRaised_any` nel panel o eliminarle. Oggi servono a `TR_D` e sono
> confrontate dai checkpoint C–F, quindi toglierle non è gratis.

**M14 — la riparazione delle date dei deal inventa date.**
`Deal.csv` ha **61.169 deal su 385.481 senza `DealDate`**, e i deal sono ciò
che determina `GrowthStage`, cioè il target. Quattro passaggi successivi
provano a inventarne una (`1_Arrange_DB.R:917-976`). *Misurato:*

| passaggio | deal riparati | quota | che assunzione è |
|---|---:|---:|---|
| 1 — data dallo stato «Out of Business» | 1.210 | 0,4% | plausibile |
| 2 — data dallo stato «Acquired/Merged» | 234 | 0,1% | plausibile |
| 3 — primo round → 1° gennaio dell'anno di fondazione | **23.147** | **7,0%** | assunzione forte |
| 4 — media fra l'anno del deal precedente e del successivo | **11.615** | **3,5%** | invenzione pura |
| **totale con data inventata** | **36.206** | **10,9%** | |

*(quote su 332.818 deal, cioè dopo il filtro `YearFounded > 2000` che sta fra
il passaggio 2 e il passaggio 3.)*

**Propagazione: arriva alle 47, attraverso il target e attraverso il campione.**
Il passaggio 3 schiaccia 23.147 primi round sull'**età 0**, ed è proprio l'età
del primo stadio `Early` che `build_windowed_dataset` usa per decidere chi
entra nel campione (`StartingAge <= 2`). Un'azienda il cui primo round non ha
data viene dichiarata d'ufficio «Early a zero anni» ed entra.

**M25 — 16.690 deal escono dal panel in silenzio.** *(voce nuova, 2026-09-11;
il meccanismo era censito come trappola di traduzione T26, ma non è solo una
trappola.)*
Dopo i quattro passaggi di M14 restano **16.690 deal senza data, il 5,0% del
totale, su 13.184 aziende**. `pmax` senza `na.rm` li manda in un gruppo ad anno
nullo e **non si agganciano a niente**: spariscono senza errore e senza traccia.

*Misurato, per tipo di deal fra quelli persi:*

| tipo | deal persi |
|---|---:|
| Accelerator/Incubator | 6.760 |
| Secondary Transaction - Private | 3.235 |
| **Later Stage VC** | **1.912** |
| **Early Stage VC** | **1.272** |
| Equity Crowdfunding | 762 |
| terminali (uscita o fallimento), in tutto | 299 |

**Propagazione: arriva alle 47, sul target.** Un'azienda il cui unico round
«Later Stage VC» non ha data non raggiunge mai lo stadio `Later`: viene
etichettata **non-successo**. Il bias del target è sistematico e va in una sola
direzione, il pessimismo.

Le opzioni sono tre e vanno decise alla fase 4: (a) tenere il comportamento R e
dichiararlo nell'articolo; (b) collocare il deal sull'anno di fondazione, che
è ciò che `pl.max_horizontal` farebbe da solo e che inventa deal in 7.335
anni-azienda; (c) tenere il deal con un flag «data ignota» e escluderlo solo
dalle colonne temporali. Nessuna delle tre è gratis.

**M15 — i deal antecedenti la fondazione vengono spostati nel tempo.**
`Year_Delta = pmax(year(DealDate), YearFounded)` schiaccia sull'anno di
fondazione i deal datati prima. È un evento **spostato**, non scartato.
*Misurato: **143 deal su 332.818, su 122 aziende**.* Trascurabile — è M14 e
M25 che contano, non questa.

### Candidati all'eliminazione

**X15 — `UndisclosedAmountFlag`.** Serviva **solo** a selezionare le righe da
imputare con la RandomForest. Con la RF eliminata (M11) è **calcolata e mai
usata**: peso morto nato dall'eliminazione. Da cancellare alla fase 4.

**X16 — `DealSynopsis`.** Letta solo per costruire `UndisclosedAmountFlag`. Se
cade X15, cade anche questa lettura.

**X17 — colonne di `Investor` selezionate e mai usate.**
`db_inv` seleziona `Description`, `YearFounded`, `HQCity`, `HQCountry`,
`PrimaryInvestorType` (usata), `MostLikelyFundraisIng`. Di queste,
`Description`, `YearFounded`, `HQCity`, `HQCountry` e `MostLikelyFundraisIng`
non vengono mai usate. *Nella traduzione non le ho portate.*

**X18 — `PercentAcquired`.** Selezionata in `deals` e mai aggregata.

**X19 — `DealType` concatenato.** L'aggregazione produce una stringa tipo
`"Accelerator/Incubator; Angel (individual)"`. *Da confermare* se serve a
qualcosa oltre alla leggibilità.

**X20 — `Other_Deal`, `PreferredVerticals`, `MeanMedianValuation_cum`,
`MeanTotalActivePortfolio_cum`, `InvestorOwnership`, `PremoneyValuation`.**
Nel panel; *da confermare* quali arrivino alle 47.

---

## Fase 5 — finalizzazione

`2_Arrange_Final.R` (intero) · `src/panel/stage5_final.py` ·
**CHECKPOINT C e D**

### Difetti riprodotti

**B3 — `StageBlock` diventa NA per l'intera azienda** (`fix_stageblock_na`).
`cumsum` su un confronto che vale NA propaga fino in fondo al gruppo, quindi
dal primo `GrowthStage` mancante in poi `StageBlock` è NA. La colonna non è
usata a valle.

### Trappole di traduzione

**T32 — `cumsum` di R propaga NA fino in fondo al gruppo.**
`cumsum(c(1, NA, 3))` è `c(1, NA, NA)`. Quello di polars lascia un null al suo
posto e **continua ad accumulare**, dando `c(1, null, 4)`. Riguarda
`TotalRaised_NA_cum` e `TotalRaised_any_cum`, cioè proprio le due varianti che
diventano NA quando un importo non è dichiarato. Risolta con
`rutils.r_cum_sum`.

**T33 — `NewInvestors` è calcolato prima che `TotalInvestors` sia
sovrascritto** dalla propria cumulata, dentro lo **stesso** `mutate`. dplyr
valuta in sequenza; invertire i due passaggi cambia silenziosamente il peso di
tutte le medie ponderate.

**T34 — `rle` tratta ogni NA come una sequenza a sé.**
`YearsInStage` nasce da `sequence(rle(GrowthStage)$lengths)`: due anni
consecutivi con stadio mancante non formano una sequenza di lunghezza due, ma
due sequenze di lunghezza uno.

**T35 — la media ponderata cumulata dell'R è O(n²).**
L'equivalente algebrico `cumsum(x·w) / cumsum(w)` sulle sole righe valide è
O(n) ed è **esatto**, non un'approssimazione.

**T36 — `next_different` con stadio corrente NA dà NA.**
`which(future != NA)[1]` è NA, quindi sia lo stadio futuro sia la distanza sono
nulli. In più l'indicizzazione `(i+1):n()` sull'ultima riga di ogni gruppo
produce in R un vettore invertito, ma il risultato resta comunque NA: si
riproduce il risultato, non il meccanismo.

**T37 — la cascata di `GrowthStage` è a corto circuito.**
Sette condizioni, il primo match vince: **l'ordine è vincolante**. E
`OwnershipStatus` è nullo su quasi tutte le righe, quindi `NA | TRUE` vale TRUE
e il flag vince, mentre `NA | FALSE` vale NA e la condizione non matcha —
comportamento identico in polars, ma dipende dalla logica a tre valori e non va
«semplificato».

### Scelte discutibili

Vedi **M1** (`GrowthStage` mescola stato attuale e storia) e **M0** (attributi
del CEO non temporizzati).

**M16 — `cumany` rende gli stadi monotoni.**
Un'azienda non può retrocedere di stadio. Ragionevole per uscita e fallimento;
discutibile per la sequenza seed → early → later, dove un round successivo di
tipo «inferiore» non abbassa mai lo stadio.

**M17 — dove `TR_D == 1` tutte le varianti di `TotalRaised` vanno a 0.**
Un anno senza deal non ha raccolto niente: corretto. Ma azzera anche
`TotalRaised_NA` e `TotalRaised_any`, che erano le due colonne costruite per
**distinguere** lo zero dal non-noto. Con M13 decisa a favore di `TotalRaised`
questa voce perde rilevanza per le 47, ma resta valida se un giorno si vuole
usare `TotalRaised_NA`: dopo questo passaggio quella colonna non distingue più
niente sugli anni senza deal.

### Candidati all'eliminazione

**X21 — `StageBlock`.** Non riproducibile alla fase 6, non letta da nessuno.

---

## Fase 6 — raggruppamento degli stadi e troncamento

Nessuno script R · `src/panel/stage6_panel.py` · **CHECKPOINT E**

Il codice che produceva questo passaggio è andato perduto: esisteva solo il suo
output. Le quattro regole sono state **ricostruite** dal file e ognuna
verificata contro di esso a divergenza zero su tutte le 882.324 righe.

### Trappole di traduzione

**T38 — lo stadio futuro va calcolato prima del troncamento.**
Dopo, `Out` ed `Exit` non sarebbero più raggiungibili come stadio futuro, e il
senso della colonna è esattamente quello. È l'errore più facile da fare
riscrivendo questa fase.

**T39 — `"Stay"` non è un null.**
Quando non esiste un gruppo futuro diverso, il valore è la stringa letterale
`"Stay"` e la distanza è quella dall'**ultima riga non troncata**
dell'azienda — non nulla. La fase 7 sostituisce `"Stay"` con il gruppo
corrente.

**T40 — `YearsInStage` viene ricalcolata e sovrascritta.**
Il valore che la fase 5 aveva calcolato sullo stadio **non raggruppato** viene
buttato e rifatto sul gruppo.

### Scelte discutibili

**M18 — il troncamento elimina anche le righe non terminali che seguono una
terminale.** *Misurato: 5.364 righe (il troncamento ne elimina 119.301 in
tutto, da 1.001.625 a 882.324, su 39.622 aziende).* Un'azienda che risulta
«Out» e poi ha un altro deal viene troncata al primo Out, e la sua vita
successiva scompare. È coerente con l'idea che l'uscita sia assorbente (M16),
ma va detto. **Da leggere insieme a M1**: di quelle 39.622 aziende, 2.764 sono
troncate su una riga il cui stadio terminale viene dal solo `OwnershipStatus`.

**M19 — `StageBlock` non è riproducibile.** *Misurato: 81.954 righe fuori da
qualsiasi variante provata.* Dichiarata come divergenza attesa ai checkpoint E
e F.

### Candidati all'eliminazione

**X22 — `GrowthStage`, `GrowthNextStage`, `TimeNextStage` non raggruppate.**
Restano nel panel **accanto** alle versioni raggruppate: è una duplicazione. Le
versioni raggruppate sono quelle che `preprocessing.py` usa.

---

## Fase 7 — competitor temporizzati

`notebook_temporizzazione_competitors.ipynb` celle 4 e 6 ·
`src/panel/stage7_competitors.py` · **CHECKPOINT F**

### Trappole di traduzione

**T41 — le date usano `parse_date_r`, non il parser del notebook originale.**
Il notebook leggeva `%m/%d/%Y` con fallback `%m/%d/%y`, e chrono interpreta un
anno a due cifre `25`–`69` come **2025-2069** mentre R con `cutoff_2000 = 24`
lo interpreta come **1925-1969**. Su questa estrazione tutte le date hanno
quattro cifre, quindi la differenza è **latente e non attiva**; è stata
uniformata perché è la colonna che decide se un concorrente è vivo, e due
convenzioni diverse nella stessa pipeline sono un problema che aspetta di
succedere.

**T42 — `company_life` è costruita su tutte le 134.355 aziende**, non solo
sulle coorti del panel: un concorrente può essere più vecchio del 2000.

**T43 — il remap `CompanyID` → interi è l'ultima operazione della pipeline**,
perché distrugge ogni possibilità di join con i file di riferimento.

### Scelte discutibili

**M20 — `MaxYear` come proxy di «azienda viva».**
È l'ultimo anno con **dati**, non l'anno in cui l'azienda è morta. Un'azienda
ben coperta da PitchBook risulta viva più a lungo di una coperta male, quindi
la temporizzazione **sovrappesa sistematicamente i concorrenti grandi e ben
coperti**. È il difetto metodologico principale di questa fase e va dichiarato
nell'articolo.

**M21 — `Same_Country` cambia significato mantenendo il nome.**
Era un booleano — «esiste un concorrente sopra 90 di similarità nel nostro
paese» — e diventa un **conteggio** di concorrenti attivi nello stesso paese.
La feature nei modelli si chiama uguale in entrambi i casi.

**M22 — `SimilarityScoreMean` riempita con 0.**
Dove nessuna azienda simile era attiva quell'anno il valore è 0, che è il
**minimo della scala**, non un valore neutro: un'azienda senza concorrenti vivi
appare a un modello come un'azienda i cui concorrenti sono massimamente
diversi. Alternative: NA da imputare, oppure una colonna separata «nessun
concorrente attivo».

**M23 — `N_Competitors_All` non è il `N_Competitors` dell'R.**
Conta solo i concorrenti che hanno una finestra di vita utilizzabile, quindi è
il minore dei due. L'esperimento senza finestra la usa come se fosse la
versione statica dell'altra.

**M24 — la relazione competitor è orientata.**
Se A dichiara B come simile ma B non dichiara A, allora B conta fra i
concorrenti di A e A non conta fra quelli di B. La direzione inversa non viene
aggiunta, né qui né alla fase 3a. Da decidere se è quello che si vuole.

Vedi anche **M2** (vintage dell'estrazione).

### Candidati all'eliminazione

**X23 — `N_Europe` e `N_Outside_Europe` vengono eliminate qui.** Vedi X11: se
si eliminano già alla fase 3a, sparisce anche tutta la mappa Europa.

---

## Riepilogo per priorità

Riordinato il 2026-09-11 su quello che è stato **misurato**, non su quello che
sembrava grave. La colonna «arriva» usa le tre etichette del cancello
2008–2017 spiegate all'inizio del file.

| # | voce | perché conta | arriva |
|---|---|---|---|
| 1 | **M0** attributi delle persone non temporizzati — **corretto il 2026-09-15** | era informazione dal futuro dentro **tre** delle 47 feature; ruoli e titoli ora si contano fino all'anno della riga | **alle 47** |
| 2 | **M14 + M25** date dei deal | 10,9% delle date inventate, 5% dei deal che evaporano; tocca il target e il campione | **alle 47** |
| 3 | **M20 / M22 / M23** competitor temporizzati | `MaxYear` come proxy di «viva», zero come riempimento: bias sistematico su tre feature | **alle 47** |
| 4 | **M2** vintage competitor | i risultati pubblicati non sono riproducibili su quelle tre feature | **alle 47** |
| 5 | **M1** `GrowthStage` mescola stato attuale e storia | 2.764 aziende troncate lì, 9.620 righe perse; ma il salto temporale è quasi nullo | al panel |
| 6 | **M12** regola `Zero_Invested` | statistiche globali che decidono un valore per riga; da tenere o togliere | **alle 47** |
| 7 | **B1** soglia `1999` / `2000` | 2,77% del panel è rumore puro, ma nessun risultato pubblicato è toccato | al panel |
| 8 | **B6** `seq()` all'indietro | 186 righe che dicono che un'azienda esisteva prima di nascere | al panel |
| 9 | **X11 / X23** mappa Europa | ~100 righe e un file di dati che esistono solo per il checkpoint B | muore prima |
| 10 | **X13** `N_News` | costante di fatto: `> 0` su 97 righe su 882.324 | al panel |
| 11 | **X15 / X16** `UndisclosedAmountFlag` e `DealSynopsis` | morti con l'eliminazione della RandomForest | al panel |
| 12 | **B2** `Is_Other`, **B3** `StageBlock`, **X21** | colonne inutilizzabili, nessuna fra le feature | al panel |
| 13 | **M15** deal spostati sull'anno di fondazione | 143 deal su 122 aziende | **alle 47** |

**Chiuse, decise o declassate** — restano nel file per tracciabilità, fuori
dalla classifica:

| voce | perché è uscita |
|---|---|
| **M11** RandomForest | **eliminata**, decisione del 2026-09-11 |
| **M13** `TotalRaised_Est` | **decisa**: la sostituisce `TotalRaised` |
| **B4** `Same_Country` nulla | la colonna col bug è buttata alla fase 7: **muore prima** |
| **B8** asimmetria Europa | entrambe le colonne sono eliminate alla fase 7 |
| **M7** `Percent_Females` diluita | *misurato: genere ignoto sullo 0,6% delle righe* |
| **X14** `EmployeeCount` | *misurato: valorizzata sul 30% delle righe*, non è peso morto |

Le voci **T** non sono in questa classifica: sono già risolte. Vanno rilette
prima di riscrivere la fase corrispondente, perché una riscrittura le
reintroduce senza far rumore. **Due eccezioni da leggere comunque:** T16, la
cui conclusione era sbagliata ed è stata corretta; e T26, che è anche una
scelta metodologica e ora ha una voce sua (M25).

## Difetti del codice Python, non dell'R

Due cose trovate rileggendo la traduzione. Nessuna delle due cambia un output
di oggi; la prima cambia un output appena si accende un flag.

1. **`round()` nel ramo corretto di B5.** R arrotonda al pari, polars per
   eccesso. Irrilevante sulla media globale (6,61), decisivo sulle medie per
   azienda, dove i pareggi a `.5` sono normali. Vedi il riquadro in B5.
2. **`config/config.yaml` ha una chiave morta.** È stato aggiunto
   `first_year: 2010`, ma `build_windowed_dataset` ha ancora `>= 2010` scritto
   a mano nel codice. O si legge la chiave o si toglie.
