# Panel — errori logici, differenze implementative, peso morto

Data: 2026-09-10
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

---

## Trasversale — la voce più importante

### M0 — gli attributi delle persone non sono temporizzati

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

Nell'elenco delle 47 feature dei modelli ci sono `WorkExp_Idx_Mean`,
`Highest_Degree_Mean`, `Avg_Earliest_Year` e `WorkExperienceIndex_CEO`.

**Perché la segnalo per prima:** l'articolo è sul look-ahead bias. Questa è
informazione dal futuro che entra nelle feature, e non passa da nessuno dei
dieci bug censiti — è nel disegno della pipeline, non in un suo difetto.

**Cosa comporterebbe correggerla:** `PersonPositionRelation` e
`PersonEducationRelation` hanno delle date. In principio si può ricostruire il
conteggio delle posizioni *alla data*, invece di prendere il totale corrente.
È l'intervento più costoso di tutta la lista e va valutato a sé: **da
misurare** quanto cambierebbe.

### M1 — `GrowthStage` mescola stato attuale e storia

**Dove:** fase 5, `2_Arrange_Final.R:80-92`.

La cascata che definisce `GrowthStage` — e quindi il target — ha due fonti:

- `OwnershipStatus`, che è lo stato **attuale** dell'azienda, agganciato
  all'anno della sua `OwnershipStatusDate`;
- i flag cumulativi ricavati dai deal, che sono storici.

Nelle prime tre condizioni la prima fonte vince con un `|`. Va guardata con
attenzione: `OwnershipStatus` è nullo su quasi tutte le righe del panel (è
agganciato solo all'anno della sua data), quindi nella pratica decide poco, ma
dove decide sta proiettando lo stato di oggi su un anno passato. **Da
misurare:** su quante righe la condizione è vera per via di `OwnershipStatus` e
falsa per via dei flag.

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
*Misurato: 245 righe su 106 aziende.* Impatto trascurabile in volume, ma sono
righe prive di senso.

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

**M3 — `MaxYear` come limite del panel.**
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

**B5 — `PermanenzaMedia` è un solo numero globale**
(`fix_permanenza_media_per_company`). L'R la calcola con un `summarise` senza
`group_by`, nonostante il commento dichiari «per ciascuna CompanyID»: è la
permanenza media su tutto il dataset, usata per imputare l'`EndDate` di
chiunque. *Misurato: con la media per azienda il `DeltaEnd` medio passa da
11,57 a 12,04 anni, le righe con dati di team da 885.142 a 892.534 (+0,84%),
`Total_People` +0,91%, `Total_Founders` +1,23%.* Sistematico ma piccolo.

**B9 — due anomalie nella deduplica** (`fix_dup_coalesce`).
I duplicati si riselezionano filtrando per `PersonID` invece che per la coppia
`(CompanyID, PersonID)`, e `coalesce(first(x), last(x))` non vede un valore
presente solo in una riga intermedia. Il primo allarga inutilmente il lavoro,
il secondo può perdere un dato con tre o più duplicati.

**B10 — `Is_Out` resta NA per le aziende non fallite** (`fix_is_out_na`).
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

**M4 — `EndDate` imputata a `2024-12-31`.**
Chi ha `IsCurrent == "Yes"` riceve come fine permanenza una data fissa legata
al **vintage dell'estrazione**. Rieseguire la pipeline su un download del 2026
darebbe finestre diverse per le stesse persone.

**M5 — `DeltaStart = 0` per i founder.**
Sovrascrive la data di inizio reale: si assume che un founder ci sia dal primo
giorno. Ragionevole come euristica, ma cancella un dato che in alcuni casi
c'era. *Misurato: 19.458 founder avevano una `DeltaStart` reale diversa da 0.*

**M6 — `PermanenzaMedia` come imputazione.**
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
**È il difetto più grave del registro.** La fase 1 filtra
`YearFounded > 1999`; questa fase e la fase 4 filtrano `YearFounded > 2000`.
Risultato: l'intera coorte di aziende fondate nel **2000** entra nel panel
**senza nessun dato di team**, e siccome `preprocessing.py` scarta le righe con
`Total_People` nullo, quella coorte **spa­risce silenziosamente** dal dataset
finale. *Misurato: coorte 2000 con 0,0% di righe dotate di dati di team, contro
l'87,1% del 2001 e l'88,2% del 2002.*

**B7 — `paste(unique(Institute))` include i NA come testo**
(`fix_institute_na_literal`). *Misurato: 390.544 righe del riferimento hanno un
`Institute` che comincia con `"NA; "`.* Cosmetico: non altera il flag «ateneo
fra i primi 50», che cerca nomi di università.

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
`> 2000` elimina la riga **prima** che l'aggregazione la veda. *Misurato: 0
righe fantasma.* **Conseguenza da tenere presente: correggere B1 rimuove anche
questa protezione**, e le righe fantasma potrebbero comparire.

**T17 — `full_join`, non `left_join`.**
Il panel del team può contenere anni-azienda che lo scheletro non ha. È così
che si arriva alle 1.001.625 righe: con un left join sarebbero meno.

**T18 — `unnest(Years, keep_empty = TRUE)`** tiene un'azienda senza finestra
utilizzabile come **una riga con `Years = NA`**, che poi non si aggancia a
niente.

### Scelte discutibili

**M7 — `Percent_Females` ha come denominatore tutte le persone.**
Usa `.N`, cioè conta anche le persone con genere ignoto. Se il genere è
sconosciuto per metà del team, la quota di donne è **diluita verso il basso**
invece di essere calcolata sui soli casi noti. `Percent_Females` è una delle 47
feature. *Da misurare:* quanto è frequente il genere ignoto.

**M8 — founder e chiunque altro pesano uguale.**
L'aggregazione mette sullo stesso piano i founder e ogni altro membro del board
team. `Total_Founders` li conta, ma gli indici di esperienza e di istruzione
sono medie su tutti.

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

### Difetti riprodotti

**B4 — `any()` senza `na.rm`** (`fix_same_country_narm`).
`Same_Country` restituisce NA quando nessun confronto è vero *e* almeno uno è
mancante, invece di `False`. *Misurato: 1.809 aziende, l'1,55% di quelle con un
aggregato competitor.* Poche, **ma `Same_Country` è una delle 47 feature**.

**B8 — `N_Europe` e `N_Outside_Europe` non sono simmetriche**
(`fix_europe_asymmetry`). `N_Europe` somma su **tutte** le righe,
`N_Outside_Europe` solo su quelle con similarità sopra 90. Non sono due facce
dello stesso conteggio.

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
sono feature dei modelli. Conseguenza: **tutta la derivazione di
`src/panel/data/europe.csv` e `scripts/derive_europe_mapping.py` esiste solo
per far passare il checkpoint B.** È il candidato più netto della lista: se la
fedeltà non è più l'obiettivo, spariscono ~100 righe di codice e un file di
dati.

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
il riferimento ne conta 1.137 distribuite su 248 righe di panel.* `N_News` è
quindi **zero sul 99,98% delle righe**: è di fatto una costante e non può
portare informazione. Candidata forte.

**X14 — `EmployeeCount`.** *Da misurare* su quante righe è valorizzata: se è
sparsa come `N_News`, vale la stessa considerazione.

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

**M11 — l'imputazione RandomForest, sospesa.**
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
**Non è portata.** Il commento `RF_SUSPENDED` in cima al modulo elenca le sette
colonne che creava e dove.

**M12 — la regola `Zero_Invested`.**
I tipi di deal in cui l'importo manca in oltre il 90% dei casi ricevono 0
invece di NA. È una regola derivata da **statistiche globali su tutto il
dataset**: stessa famiglia di M11, più mite. È **mantenuta** perché alimenta
`TotalRaised` e perché «questo tipo di deal non dichiara mai l'importo, quindi
è zero» non è un'imputazione modellistica. Da decidere se tenerla.

**M13 — `TotalRaised` confonde tre situazioni diverse.**
Vale 0 sia quando non c'è stato nessun deal, sia quando c'è stato un deal a
importo zero, sia quando c'è stato un deal di importo **non dichiarato**.
`TR_D` distingue il primo caso; `TotalRaised_NA` e `TotalRaised_any` il terzo.
Con la RF sospesa questa è **la decisione aperta più urgente**: quale colonna
prende il posto di `TotalRaised_Est` in `src/preprocessing.py`.

**M14 — la riparazione delle date dei deal inventa date.**
Quattro passaggi successivi: dallo stato di proprietà per fallimenti e
acquisizioni (plausibile), dall'anno di fondazione per il primo round
(assunzione), e infine la **media arrotondata per eccesso fra l'anno del deal
precedente e quello del successivo** (invenzione pura). *Da misurare:* su
quanti deal scatta ciascuno dei quattro.

**M15 — i deal antecedenti la fondazione vengono spostati nel tempo.**
`Year_Delta = pmax(year(DealDate), YearFounded)` schiaccia sull'anno di
fondazione i deal datati prima. È un evento **spostato**, non scartato.

### Candidati all'eliminazione

**X15 — `UndisclosedAmountFlag`.** Serviva **solo** a selezionare le righe da
imputare con la RandomForest. Con la RF sospesa è **calcolata e mai usata**:
peso morto che ho introdotto io sospendendo la RF.

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
**distinguere** lo zero dal non-noto.

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
terminale.** *Misurato: 5.365 righe.* Un'azienda che risulta «Out» e poi ha un
altro deal viene troncata al primo Out, e la sua vita successiva scompare. È
coerente con l'idea che l'uscita sia assorbente (M16), ma va detto.

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

Ordinato per quanto conta, non per fase.

| priorità | voce | perché |
|---|---|---|
| 1 | **M0** attributi delle persone non temporizzati | informazione dal futuro dentro quattro delle 47 feature, in un articolo sul look-ahead bias |
| 2 | **B1** soglia `1999` / `2000` | un'intera coorte di fondazione sparisce dal dataset |
| 3 | **M13** `TotalRaised` confonde tre situazioni | decisione aperta e bloccante: cosa sostituisce `TotalRaised_Est` |
| 4 | **M1** `GrowthStage` mescola stato attuale e storia | tocca il target |
| 5 | **M20** `MaxYear` come proxy di «viva» | bias sistematico su tre feature competitor |
| 6 | **B4** `Same_Country` nulla | 1.809 aziende su una feature dei modelli |
| 7 | **M22** `SimilarityScoreMean` riempita con 0 | zero è il minimo, non il neutro, su una feature |
| 8 | **M7** `Percent_Females` diluita dal genere ignoto | feature dei modelli |
| 9 | **M2** vintage competitor | i risultati pubblicati non sono riproducibili su tre feature |
| 10 | **X11 / X23** mappa Europa | ~100 righe di codice e un file di dati che non servono a niente |
| 11 | **X13** `N_News` | costante di fatto: zero sul 99,98% delle righe |
| 12 | **X15 / X16** `UndisclosedAmountFlag` e `DealSynopsis` | morti da quando la RandomForest è sospesa |
| 13 | **B2** `Is_Other` | colonna inutilizzabile, non fra le feature |
| 14 | **M14 / M15** date dei deal inventate e spostate | eventi collocati nel tempo per assunzione |

Le voci **T** non sono in questa classifica: sono già risolte. Vanno rilette
prima di riscrivere la fase corrispondente, perché una riscrittura le
reintroduce senza far rumore.
