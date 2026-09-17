# Panel — stato della revisione fase per fase

Ultimo aggiornamento: 2026-09-17 (fase 7: competitor dietro interruttore,
colonne `_All` eliminate, `N_Similar` aggiunta — la revisione fase per fase è
arrivata in fondo; più la chiusura dei due residui di M0)

Questo file è il punto di ripresa. Se la sessione di lavoro si interrompe,
basta questo file più `git log` per riprendere senza ricostruire niente.
**Va aggiornato alla fine di ogni fase.**

## Obiettivo

Portare tutta la logica di costruzione del panel dentro `build_panel.ipynb`,
correggendo gli errori logici ed eliminando le parti inutili. Il notebook deve
leggere i CSV grezzi e produrre il panel finale, con solo poche chiamate a
script esterni.

Il repository accompagna un articolo scientifico: ogni passaggio deve essere
comprensibile e verificabile leggendo il notebook.

## L'accordo di lavoro

Una fase alla volta, in ordine di pipeline (1 → 7). Per ciascuna:

1. il codice della fase entra nel notebook, **un blocco per passaggio logico**,
   ognuno con il suo commento;
2. **si rileggono le voci di quella fase** in
   `docs/panel_errori_logici.md`, citate per id (B/T/M/X);
3. Giulio legge, corregge, e chiede cosa comporta una modifica;
4. si riesegue da quella fase in poi e si riporta il registro: quali colonne si
   sono mosse e su quante righe;
5. **si cancella il modulo `src/panel/stageN_*.py`** nello stesso commit in cui
   il suo codice entra nel notebook — nessuna finestra con due sorgenti;
6. si passa alla fase successiva.

### Regole non negoziabili

- **Le correzioni non sostituiscono il comportamento R: aggiungono un flag**
  `fix_*` in `PanelConfig`, spento per default. A flag tutti spenti i sei
  checkpoint restano verdi, e si può affermare di aver riprodotto la pipeline
  originale; a flag accesi si ottiene il panel corretto, e per ciascuna
  correzione si sa quante righe ha spostato.
- **Le eliminazioni** sono l'eccezione: una colonna morta si cancella, ma serve
  prima la prova che nessuno la legge. **Le decide Giulio, fase per fase.**
- **Restano negli script**: `rutils.py` (le semantiche R), `io.py`,
  `validate.py`, `config.py`, `data/europe.csv`. Sono le cose che nessuno vuole
  leggere inline e che i test coprono bene.
- **Restano chiamate a funzione** l'espansione della fase 2b e quella della
  fase 7: sono le due più affamate di memoria e le meno interessanti da leggere
  riga per riga.
- **Memoria**: la pipeline tocca 5,66 GB di picco su una macchina da 7 GB. I
  rapporti di verifica girano in sottoprocesso, le celle di ispezione
  proiettano solo le colonne che servono e liberano le variabili con `del`. Se
  il kernel muore non si perde niente: gli stadi comunicano via parquet e si
  riparte dalla cella della fase interrotta.

## Dove siamo

**Tutte e sette le fasi sono nel notebook**, e tutti i moduli degli stadi sono
stati cancellati. In `src/panel/` restano solo `config.py`, `io.py`,
`rutils.py`, `validate.py` ed `expansions.py`. `scripts/build_panel.py` non
esiste piu': il notebook e' la pipeline.

| fase | blocchi nel notebook |
|---|---|
| 1 — aziende, affiliate, scheletro | 1.1 – 1.10 |
| 2a — tabella persona-azienda | 2a.1 – 2a.16 |
| 2b — colonne di team | 2b.1 – 2b.3 |
| 3a — competitor | 3a.1 – 3a.5 |
| 3b — dipendenti, financials, news | 3b.1 – 3b.3 |
| 4 — deal e investitori | 4.1 – 4.10 |
| 5 — finalizzazione | 5.1 – 5.8 |
| 6 — raggruppamento e troncamento | 6.1 – 6.4 |
| 7 — competitor temporizzati | 7.1 – 7.5 |

**La migrazione e' dimostrata identica.** Eseguendo il notebook da
`data/interim` vuoto, tutte le 91 celle di codice girano senza errori in circa
otto minuti con un picco di 5,69 GB, e **ognuno dei quattordici confronti con
la base congelata da' zero colonne divergenti** — dalle 12 dello scheletro alle
115 del panel finale, senza esclusioni. I sei checkpoint contro i file R sono
verdi.

**Ora si puo' iniziare a correggere.** Da qui in avanti ogni modifica va dietro
a un flag `fix_*`, e il confronto con la base dice esattamente quali colonne si
sono mosse e su quante righe. Se un checkpoint diventa rosso senza che tu
l'abbia voluto, e' un errore di implementazione e non un effetto della
correzione.

> **Aggiornamento 2026-09-14.** La regola dei flag vale per il notebook
> completo, che e' congelato. Nel **leggero** le correzioni si scrivono
> direttamente nel codice, senza flag, e il loro effetto misurato si registra
> in questo file.

## Le correzioni, fase per fase

Si riparte dall'inizio della pipeline e si corregge ogni errore quando si
presenta. Per ciascuno, prima di toccare il codice: **qual e' l'errore, perche'
e' stato commesso, come si risolve, cosa cambia.** Spuntare man mano.

Sul notebook **leggero**, e in ordine di pipeline.

| fase | voci da affrontare | stato |
|---|---|---|
| 1 | **B6 `seq()` all'indietro** | **fatto** (2026-09-11) |
| 2a | **B9** deduplica · **B5** permanenza media · **B10** `Is_Out` · **M4** fine 2024 · **M6** | **fatto** (2026-09-14) |
| 2a | **M5** founder a eta' zero | **decisa** (2026-09-16): si tiene; `IsFounder` corretto per non prendere "Foundation" e gli assistenti |
| 2a-2b | **T17** full join · **M3** `MaxYear` come fine del panel | **fatto** (2026-09-14): taglio e left join |
| 2b | **B1 soglia** | **fatto** (2026-09-11) |
| 2b | **B7** `Institute` con "NA" | **fatto** (2026-09-16) |
| 2b | **M7** denominatore di `Percent_Females` | **fatto** (2026-09-16) |
| 2b | **M8** founder pesati come gli altri | **decisa** (2026-09-16): si dichiara nell'articolo |
| 4 | **B1 soglia, seconda occorrenza** | **fatto** (2026-09-11) |
| 4 | **M12** `Zero_Invested` | **eliminata** (2026-09-16): leakage, e il panel non cambia |
| 4 | **M14** date inventate · **M25** deal che evaporano | **fatto** (2026-09-16): passaggio 4 esteso, elenco delle aziende salvato |
| 4 | **M15** deal datati prima della fondazione | **decisa** (2026-09-17): si tiene lo spostamento all'anno di fondazione e si dichiara · 139 deal in 121 aziende, e nel 68% dei casi il deal precede davvero la costituzione legale |
| 4 | **M26** attributi degli investitori non temporizzati | **implementata** (2026-09-16) come `TEMPORIZZA_INVESTITORI`, spento di default · resta aperto se accenderlo: a valle le due colonne superano il 40% di mancanti e verrebbero scartate |
| 5 | **M1** `GrowthStage` mescola stato e storia | **decisa** (2026-09-16): la cascata si tiene, si dichiara |
| 5 | **M16** `cumany` rende gli stadi monotoni | **decisa** (2026-09-16): si tiene, si dichiara |
| 6 | **M18** troncamento | **decisa** (2026-09-16): si tiene, e' cio' che rende corretto il target |
| 7 | **M20, M21, M22, M23, M24** competitor | **fatto** (2026-09-17): M22 e M23 risolte, M20/M21/M24 decise e da dichiarare · M2 misurata |
| trasversale | **M0** attributi delle persone non temporizzati | **fatto** (2026-09-15): esperienza e istruzione, team e CEO · **residui chiusi** (2026-09-17): anno dei ruoli senza data e standardizzazione · resta solo il CEO dal board team |
| dopo | M13: `TotalRaised` in `preprocessing.py` | **fatto** (2026-09-17): codice aggiornato e provato in memoria · la rigenerazione di `data/processed/` e dei run si fa **una volta sola, a pipeline del panel completata** |

**Cadute con le fasi 3a e 3b**, che nel leggero non esistono: B4, B8, X11/X23
(la mappa Europa e `scripts/derive_europe_mapping.py`), X13, X14, T19-T25.
**Cadute con le colonne eliminate**: B2 (`Is_Other`), B3 e X21 (`StageBlock`),
X22, M17, M19, X15-X20.

## `build_panel_light.ipynb` — la pipeline di lavoro

**Dal 2026-09-11 `build_panel.ipynb` e' un artefatto congelato**: ha dimostrato
che la traduzione dall'R e' fedele (sei checkpoint verdi, quattordici confronti
a divergenza zero) e non si tocca piu'. La pipeline su cui si lavora e'
`build_panel_light.ipynb`.

Produce **50 delle 53 colonne di `data/raw/example_panel.csv`** — con
`TotalRaised` al posto di `TotalRaised_Est`, e senza le tre `*_All`, eliminate
il 2026-09-17 — piu' le due aggiunte qui, `UndisclosedAmountShare` e
`N_Similar`: **52 in tutto**. Scrive in `data/interim_light/` per non toccare i
parquet del panel completo. Ogni riga e' commentata.

**Passo 1 fatto: a flag spenti il panel leggero e' identico al completo.**
*Verificato eseguendo tutte e 45 le celle: 882.324 righe x 53 colonne,
116.327 aziende, **zero divergenze su tutte e 51 le colonne condivise**.*
`Institute` e' confrontata come insieme di atenei e non come stringa, perche'
l'ordine della concatenazione non porta informazione.

**Passo 2 fatto: a flag accesi il panel leggero e' ancora identico al
completo.** *Verificato eseguendo entrambi i notebook con
`fix_founding_year_threshold` e `fix_negative_delta` accesi: 880.473 righe,
116.313 aziende, **zero divergenze su tutte e 51 le colonne condivise**, zero
chiavi spaiate.* Le due correzioni sono quindi implementate allo stesso modo
nei due notebook.

Effetto misurato delle due correzioni sul panel finale:

| | flag spenti | flag accesi | |
|---|---:|---:|---:|
| righe del panel | 882.324 | 880.473 | −1.851 |
| aziende | 116.327 | 116.313 | −14 |
| righe con dati di team | 795.698 | 816.297 | **+20.599** |
| righe con `GrowthStageGroup` | 623.359 | 637.043 | **+13.684** |
| righe con `Age < 0` | 186 | **0** | −186 |
| coorte 2000, righe con team | **0,0%** | **90,5%** | |
| coorte 2000, righe con stadio | **0,0%** | **60,1%** | |

Le righe totali calano mentre i dati crescono: dando alla coorte 2000 i suoi
deal, quelle aziende acquistano per la prima volta uno stadio, e molte vengono
di conseguenza troncate a un'uscita o a un fallimento. Prima restavano nel
panel per intero solo perche' non avevano nessuno stadio.

> **Nota operativa: il notebook completo a flag accesi non gira in un processo
> solo su questa macchina.** Correggere B1 aggiunge la coorte 2000
> all'espansione della fase 2b, e il picco supera i 7 GB: il processo viene
> ucciso dall'OOM. E' stato eseguito **una fase per processo** (gli stadi
> comunicano via parquet) e con la fase 2b spezzata in otto blocchi di aziende
> — `expand_team` e la sua aggregazione sono entrambe per `CompanyID`, quindi
> spezzare per azienda e' esattamente equivalente. Il notebook **leggero** non
> ha questo problema: gira intero in 91 s con un picco di 2,29 GB, perche'
> porta 22 colonne attraverso l'espansione invece di 117.

**Passo 3 fatto: i due flag non esistono piu' nel notebook leggero.**
`fix_negative_delta` e `fix_founding_year_threshold` sono state promosse a
comportamento: la soglia e' `SOGLIA_FONDAZIONE = 1999` in tutte e tre le fasi,
e il limite dello scheletro e' sempre `max_horizontal("MaxYear", "YearFounded")`.
*Verificato: il panel prodotto e' **identico riga per riga** a quello del passo
2 (880.473 x 53).* I due flag restano in `PanelConfig` perche' li usa
`build_panel.ipynb`, che e' congelato; gli altri otto flag restano spenti nel
leggero e riproducono il difetto dell'R.

La cella di confronto e' stata sostituita da una **verifica di invarianti**:
righe, aziende, colonne, righe con team e con stadio, piu' due assert sulle
correzioni (`Age < 0` deve essere zero, la coorte 2000 deve avere oltre l'80%
di righe con dati di team) e il controllo che le colonne coincidano con
`example_panel.csv`. Rieseguire il notebook completo a ogni giro non ha senso:
a correzioni accese non sta in memoria in un processo solo, e l'equivalenza e'
gia' dimostrata e registrata qui.

### Il panel corrente

| | valore |
|---|---:|
| righe | 802.148 |
| aziende | 116.312 |
| colonne | 52 |
| righe con dati di team | 749.847 |
| righe con `GrowthStageGroup` | 561.333 |
| righe con `Age < 0` | 0 |
| tempo di esecuzione | 2 min 41 s, picco 2,11 GB |

*Aggiornato il 2026-09-17, dopo la fase 7 (registri qui sotto).*

**La revisione fase per fase e' arrivata in fondo** (1 → 7). Quel che resta
non e' piu' una fase del notebook ma una lista di questioni aperte: vedi
«Decisioni aperte» in coda.

| | completo | leggero |
|---|---:|---:|
| celle di codice | 91 | **45** |
| CSV letti | 13 | **9** |
| colonne di `Company.csv` | 39 | **11** |
| colonne di `db3` | 54 | **10** |
| aggregati di team | 23 | **13** |
| colonne del panel | 117 | **52** |
| tempo di esecuzione | ~8 min | **72 s** |

Spariscono **due fasi intere**: la 3a (competitor statici, tutti sovrascritti
dalla 7) e la 3b (dipendenti, bilanci, news). Con loro cadono B4, B8, la mappa
Europa (X11/X23 e `scripts/derive_europe_mapping.py`), X13, X14, T22-T25, piu'
B3/X21 (`StageBlock`), X22, M17, M19 e B2 (`Is_Other`).

### Registro delle correzioni applicate

**B6 — `fix_negative_delta`, fase 1, blocco 1.8** (2026-09-11).
Il ramo del fix è stato riscritto: invece di fabbricare una lista di un
elemento con `pl.int_ranges` dentro un `pl.when`, alza il limite della sequenza
con `pl.max_horizontal("MaxYear", "YearFounded")`. Sei righe diventano due e il
codice dice quello che significa: il panel di un'azienda non può finire prima
di iniziare. **Comportamento identico su entrambi i rami**, verificato.

*Verificato eseguendo i blocchi 1.1–1.10 sui dati veri:*

| | flag spento | flag acceso |
|---|---|---|
| scheletro | 908.179 righe, **identico** al pre-modifica | 907.934 righe (−245) |
| righe con `Delta < 0` | 245 | **0** |
| aziende | 116.919 | 116.919 (nessuna persa) |
| `db_master_1_v1` | identico | identico (non lo tocca) |

Flag **spento** per default, come da accordo.

### Registro: fase 2 del leggero, 2026-09-14

Tutte senza flag, scritte direttamente in `build_panel_light.ipynb`.

1. **Solo le aziende dello scheletro (2a.1).** Si leggono 466.312 incarichi su
   535.568. `YearFounded` e l'ultimo anno di vita arrivano dallo scheletro:
   `db1` e la cella 1.3 sono eliminate. Effetto collaterale: la
   standardizzazione di `WorkExperienceIndex` e' calcolata su queste persone.
2. **Deduplica del board team (2a.1), sostituisce B9.**
   - *Stesso incarico* (stessa coppia e stesso titolo, 158 coppie): `IsCurrent`
     Yes se una riga lo e'; `EndDate` nulla se la riga fusa e' Yes; altrimenti
     vale la data della riga con `LastUpdated` piu' recente fra quelle che ne
     hanno una, e a parita' di `LastUpdated` l'intervallo piu' ampio.
   - *Ruoli diversi* (293 righe fuse): unione dei periodi. `StartDate` nulla se
     un ruolo non ha inizio (quindi dalla fondazione), altrimenti la piu'
     vecchia; `EndDate` nulla se un ruolo e' Yes o non ha fine, altrimenti la
     piu' recente; titoli concatenati, cosi' un founder resta founder (nell'R
     55 coppie su 713 lo perdevano).
   - Limite accettato: 2 persone e 5 anni-azienda con un buco fra due ruoli
     risultano presenti anche negli anni scoperti. Scartata l'alternativa dei
     ruoli separati fino all'espansione (piu' codice in 2b.2 e 5.6).
3. **`EndDate` mancante → 31/12 dell'ultimo anno di vita (2a.10)**, qualunque
   sia `IsCurrent`: 379.728 righe su 462.287. Chiude B5, M4, M6 e B10 (la
   tabella delle aziende fallite non serve piu'). Costo misurato: `Total_People`
   circa +7% rispetto alla media globale. Negli anni del panel la persona c'e'
   comunque, quindi la fine imputata non porta l'esito futuro dell'azienda.
   Scartate: la media per azienda (usa le permanenze future dei colleghi) e
   «solo l'anno di inizio» (`Total_People` −21%).
4. **Taglio all'ultimo anno di vita (2a.10) e left join (2b.3)**, chiude T17 e
   decide M3. Ruoli iniziati dopo l'ultimo anno scartati (3.574), `EndDate`
   esplicite successive tagliate (3.502). Motivo: dopo `MaxYear` la fase 1 non
   ha `OwnershipStatus`, quindi quegli anni non possono dare un target. Nell'R
   il full join aggiungeva 95.132 righe in 39.439 aziende (in 372 lo stadio
   cambiava proprio li'), spostando in avanti `LastAge` e l'eta' a cui si legge
   il target solo per le aziende con un team datato. Conseguenze: i 2.097 deal
   datati dopo `MaxYear` non si agganciano; sparisce l'azienda fantasma
   `807550-48`, che esisteva solo per il full join.
5. **Diagnosi di M3 registrata:** i ruoli dopo `MaxYear` stanno quasi tutti in
   aziende attive (2.272 «Privately Held» su 2.604) con la scheda aziendale non
   aggiornata. Allargare `MaxYear` con le date del board e dei deal resta
   un'alternativa possibile, non adottata.

Effetto sul panel finale:

| | prima | dopo |
|---|---:|---:|
| righe | 880.473 | 802.193 |
| aziende | 116.313 | 116.312 |
| righe con dati di team | 816.297 | 749.892 |
| righe con `GrowthStageGroup` | 637.043 | 561.195 |

**Obiettivi successivi dichiarati:** temporizzare le variabili di team anno per
anno (M0) mantenendo la stessa costruzione; gli attributi datati andranno
agganciati per (persona, anno) dopo l'espansione della fase 2b.

### Registro: M0 nel leggero, 2026-09-15

Gli attributi delle persone non sono piu' fotografie alla data di estrazione.
Tutte le decisioni sono scritte direttamente nel codice, senza flag.

**Esperienza (2a.3bis, nuova cella).** Gli 8 contatori di `Person.csv` si
ricostruiscono da cinque tabelle con una riga per ruolo: `PersonPositionRelation`,
`PersonBoardSeatRelation`, `PersonAdvisoryRelation`,
`PersonAffiliatedDealRelation`, `PersonAffiliatedFundRelation` (+ `Fund.csv` per
il `Vintage` e `Investor.csv` per gli anni di fondazione). Ogni ruolo diventa un
evento con un anno, e l'esperienza all'anno Y e' il numero di ruoli iniziati
entro Y: la data di fine non serve. **918.698 eventi** per 404.468 persone,
riassunti in `esperienza_persona_anno.parquet` (663.769 righe).

L'anno di un ruolo: la data vera; se manca l'anno di fondazione dell'entita'
(limite inferiore, non data vera); se manca anche quello, anno 0 = "conta
sempre". Per i fondi, la piu' recente fra `Vintage` e l'ingresso della persona
nella societa' d'investimento.

| gruppo | data vera | fondazione dell'entita' | sempre |
|---|---:|---:|---:|
| posizioni | 347.122 | 110.062 | 18.788 |
| seggi | 142.653 | 52.288 | 12.593 |
| altri ruoli | 220.567 | 10.059 | 4.566 |

**Controllo (2a.3ter, nuova cella).** Ogni contatore di `Person.csv` contro le
righe della sua tabella (attuali con `IsCurrent` = "Yes", passati con "No").
Coincidono per 404.461 persone su 404.468. Le **7 differenze sono incoerenze di
PitchBook**, sempre di un ruolo in piu' nella tabella: 5 nascono da un contatore
vuoto (e `Person.csv` scrive zero lasciando la cella vuota: non esiste un solo
"0" nel file), 2 no (Dan Boneh e David Spitz, un ruolo passato da advisor).
Vince la tabella dei ruoli, che e' la fonte piu' completa.

**Istruzione (2a.5, riscritta; 2a.6; controllo in 2a.5bis).** Per ogni persona e
ogni anno in cui ha preso un titolo, l'aggregazione dell'R sui soli titoli
conseguiti entro quell'anno: `istruzione_persona_anno.parquet`, 254.196 righe per
173.282 persone, di cui **59.925 cambiano nel tempo**. Un titolo senza
`GraduatingYear` (35%) conta sempre. I titoli ricavati dal nome ("Ph.D", " JD",
" MD") diventano tre flag in `db3` e si applicano in ogni anno. Il controllo
verifica che l'ultima riga di ogni persona coincida **colonna per colonna** con
l'aggregazione di tutti i titoli, cioe' con la versione dell'R.

**Aggancio (2b.1bis, nuova cella; 5.6).** Dopo l'espansione, due join asof su
(persona, anno <= Y) danno esperienza e istruzione dell'anno. `WorkExperienceIndex`
si calcola con la formula dell'R invariata (log(x+1), standardizzazione, media
dei tre gruppi); media e deviazione si calcolano sulle coppie (persona, anno)
distinte e si salvano in `parametri_esperienza.parquet`, perche' 5.6 usa gli
stessi numeri per `WorkExperienceIndex_CEO` e `Highest_Degree_CEO`.

**Effetto sul panel.** Righe, aziende e colonne non cambiano; cambiano i valori.

| colonna | righe cambiate | prima -> dopo |
|---|---:|---|
| `WorkExp_Idx_Mean` | tutte | da piatta nel tempo a crescente con l'eta': a eta' 0 -0,157 -> -0,108, a 5+ anni da -0,11 a +0,06. Correlazione con i valori vecchi 0,88 |
| `WorkExperienceIndex_CEO` | tutte | correlazione 0,85 |
| `Highest_Degree_Mean` | 15.781 (2,0%) | valorizzata su 519.956 -> 517.369 righe |
| `Avg_Earliest_Year` | 12.194 (1,5%) | media 1999,42 -> 1999,18 |
| `Institute` | 19.884 (2,5%) | valorizzata su 405.096 -> 401.689 righe |
| flag delle aree | 0,0%-0,6% | `Is_Eco` vera su 328.185 -> 323.459 righe |
| `Highest_Degree_CEO` | 2.341 (0,3%) | media 3,809 -> 3,806 |

I livelli assoluti dell'indice non si confrontano con quelli vecchi: la
standardizzazione ora e' calcolata sulle coppie (persona, anno) invece che sulle
righe di `db3`. Conta il profilo per eta'.

**Un difetto trovato e corretto.** `unique()` restituisce le righe in un ordine
che cambia fra un'esecuzione e l'altra, e la media della standardizzazione
cambiava nelle ultime cifre (1e-15). *Verificato: 8 ripetizioni davano 8
risultati; ordinando le coppie prima del calcolo, uno solo.* Con l'ordinamento,
**due esecuzioni complete danno un panel identico in ogni colonna**.

**2026-09-16 — M5 confermata e `IsFounder` piu' preciso.** Giulio conferma la
regola dell'R: chi e' founder lo e' dal giorno zero, quindi `DeltaStart = 0`
resta. *Misurato con la pipeline di oggi: 205.884 coppie founder (44,5%); fra
quelle con una data d'ingresso vera, 17.834 (8,9%) risultano entrate dopo
l'anno di fondazione, in media 3 anni, mediana 2: sono loro a essere spostate
indietro dalla regola.*
Il riconoscimento invece cambia: l'R cercava `Found`, che prende anche
"Foundation" e "Foundry". Ora cerca `founde|founding` (i refusi presenti nei
dati - Co-Founde, Co-Founderf, Co-Foundder - restano) e toglie prima la frase
"Founder's Associate" / "Founders Associate", che indica un assistente del
fondatore. *Effetto: 42 coppie perdono il flag (34 "Foundation"/"Foundry", 8
assistenti); nel panel `Total_Founders` cala di 252 anni-persona e
`Total_People` di 63, perche' chi non e' piu' founder non viene piu' riportato
all'anno zero. Le altre colonne si muovono su poche decine di righe; gli indici
di esperienza cambiano ovunque nelle ultime cifre, perche' cambia di poco la
popolazione su cui si standardizza.*

**2026-09-16 — M7 corretto.** Il denominatore di `Percent_Females` sono ora le
persone di **genere noto**; dove nessuno ha un genere noto la colonna resta
vuota. *Misurato: cambia su 8.582 righe del panel, in media di 5,33 punti; media
della colonna 14,217 -> 14,288; righe valorizzate 749.892 -> 749.093; nessun'altra
colonna si muove.* Il genere e' ignoto sullo 0,55% degli anni-persona, ma si
concentra nelle aziende documentate peggio, quindi la diluizione non era casuale.

**2026-09-16 — M8 dichiarata, non corretta.** I founder sono il 41,6% degli
anni-persona e hanno un indice di esperienza piu' basso degli altri (0,083 contro
0,199), perche' fra i non founder ci sono investitori e dirigenti esterni.
Medie sui soli founder darebbero 0,046 invece di 0,007 (correlazione 0,725) e
sarebbero vuote nel 14,5% degli anni-azienda. Cambiare la definizione
cambierebbe l'oggetto misurato: si dichiara nell'articolo.

**2026-09-16 — B7 e pulizia.** In 2b.2 i mancanti di `Institute` si scartano
(righe con `"NA"` fra gli atenei: 343.460 -> 0; nessun'altra colonna si muove).
Con questo il notebook **non legge piu' nessun flag** `fix_*`: sono rimasti in
`PanelConfig` solo per `build_panel.ipynb`. Tolti anche gli import inutilizzati
(`scale_r`, `BUG_FLAGS`) e il parquet `data/interim_light/db1.parquet`, che
nessuna cella scrive o legge piu'.

**2026-09-16 — M18, il troncamento si tiene.** Elimina 105.786 righe: 39.698
sono l'anno dell'uscita, 64.952 anni successivi gia' terminali, **1.136 righe non
terminali che seguono un'uscita** (413 aziende). Togliere anche l'anno
dell'uscita non e' un difetto ma la condizione perche' il target sia corretto:
l'esito resta in `GrowthNextStageGroup`, calcolato in 6.2 **prima** del
troncamento, e il panel pubblicato infatti non ha nessuna riga `Out` o `Exit`.
*Controprova: tenendo quella riga, 198 aziende cambiano etichetta (133 da Out a
Early, 54 da Exit a Out, 9 da Exit a Early), perche' `TargetAge` si sposta di +1
e cade sulla riga terminale, il cui stadio futuro punta alle righe post-uscita.*

**2026-09-16 — M16, la cumulata degli stadi si tiene.** *Misurato: il 70,0%
degli anni-azienda non ha nessun round, e con i soli round dell'anno lo stadio
sarebbe nullo sul 70,2% delle righe invece che sul 26,5%.* Senza `cumany` il
target non starebbe in piedi. Le righe in cui i round dell'anno direbbero uno
stadio piu' basso sono 20.747 (2,3%) in 14.749 aziende: 2.727 hanno il cumulato
gia' terminale e le elimina il troncamento della fase 6; le altre 18.020 sono
aziende gia' cresciute che prendono un grant o un round da acceleratore o angel,
classificati `Preseed`. Alternativa non adottata: affiancare allo stadio cumulato
una colonna con lo stadio dei soli round dell'anno.

**2026-09-16 — M1, la cascata di `GrowthStage` si tiene.** Nei primi tre rami
`OwnershipStatus` (lo stato attuale, agganciato al solo anno della sua data)
concorre con i flag storici. *Misurato sulla pipeline corrente: lo stato e'
valorizzato su 116.527 righe (12,8%), una per azienda, e fa scattare un ramo
terminale da solo — con il flag storico falso — in **1.928 righe**: 1.664 «Out of
Business», 195 acquisizioni, 69 quotazioni.*

*Controprova eseguita con i tre rami basati solo sui flag: il panel passa da
802.148 a 805.104 righe (+2.956), le aziende da 116.312 a 116.444, il dataset da
32.752 a 32.797 aziende, il target «Out» da 8.659 a 8.634.*

**Si tiene** perche' quelle 1.664 aziende sono fallimenti reali che nessun deal
registra: senza lo stato resterebbero nel panel come se fossero vive, e il target
direbbe «non uscita» a un'azienda chiusa. Lo stato entra solo nell'anno della
propria data, che per fallimenti e acquisizioni e' la data dell'evento: nessun
salto temporale. L'unico ramo discutibile resta `Exit_Public` (69 righe), dove la
data dello stato puo' non essere quella della quotazione. Da dichiarare
nell'articolo.

**2026-09-16 — M26, investitori temporizzati dietro l'interruttore.** Il blocco
4.1 puo' ricostruire `TotalInvestments` e `MedianRoundAmount` anno per anno dai
deal datati dell'estrazione, agganciandoli a ogni partecipazione con un join asof
sull'anno del deal. `TEMPORIZZA_INVESTITORI` parte **spento**: il default resta la
fotografia. *Verificato: a interruttore spento il panel e' identico bit per bit; a
interruttore acceso cambiano solo le due colonne attese.*

| | fotografia | temporizzata |
|---|---:|---:|
| `MeanTotalInvestments_cum`, mediana | 129,00 | 26,00 |
| `MeanMedianRoundAmount_cum`, mediana | 1,05 | 0,95 |
| righe valorizzate nel panel | 64,4% e 62,2% | 50,8% e 46,9% |
| correlazione fra le due versioni | — | 0,676 e 0,176 |
| valori mancanti nel dataset dei modelli | 15,5% e 19,6% | **44,4% e 50,3%** |

**Conseguenza da tenere presente:** oltre il 40%, `handle_missing_values` scarta
la feature. Accendere l'interruttore oggi equivale quindi a togliere le due
feature dai modelli, a meno di alzare quella soglia. La decisione su quale
versione pubblicare si prende con i risultati dei modelli in mano.

**2026-09-16 — due interruttori per rifare il panel CON il look-ahead.** Nella
cella di import ci sono `TEMPORIZZA_PERSONE` e `TEMPORIZZA_INVESTITORI`, veri per
default. A False gli attributi tornano a essere la fotografia alla data di
estrazione: nei blocchi 2b.1bis e 5.6 il join per (persona, anno) diventa un join
per persona sull'**ultima riga** delle tabelle, che e' proprio il valore alla data
di estrazione (lo dimostrano i controlli 2a.3ter e 2a.5bis). A interruttore spento
anche la standardizzazione torna quella dell'R, calcolata per coppia
(azienda, persona) invece che per (persona, anno).

Servono al confronto che l'articolo puo' portare: stessa pipeline, stesse feature,
cambia solo cosa si sapeva all'epoca. *Verificato: con gli interruttori accesi il
panel e' identico bit per bit a prima; nel ramo spento l'indice di esperienza e il
titolo di studio tornano costanti nel tempo per ogni persona.*

**Attenzione:** riportano indietro solo la temporizzazione. Le altre correzioni
(deduplica, `EndDate`, taglio a `MaxYear`, date dei deal, `Zero_Invested`,
`Percent_Females`, `IsFounder`) restano. Il panel originale resta
`build_panel.ipynb`, congelato.

**2026-09-16 — M12 `Zero_Invested` eliminata.** La regola metteva a 0 gli importi
mancanti dei 21 tipi di round che non li dichiarano quasi mai, con una soglia
calcolata su tutti i deal del dataset, anni futuri compresi: e' leakage, della
stessa famiglia della RandomForest gia' eliminata. *Verificato che toglierla non
cambia nulla: il panel finale e' identico, nessuna colonna si muove, perche'
l'unica consumatrice di `TotalInvestedCapital` e' la somma di 4.8 che tratta i
mancanti come zero.* Il blocco 4.6 e' cancellato.

**2026-09-16 — M13, aggiunto `UndisclosedAmountShare`.** `TotalRaised` scrive 0
sia dove l'azienda non ha raccolto sia dove l'importo non e' dichiarato, e
*misurato: in meta' delle righe del dataset l'importo non e' dichiarato*. Le
varianti che conservano il nullo non sono praticabili: `TotalRaised_NA` sarebbe
vuota sul 50,5% delle righe e `TotalRaised_any` sul 57,5%, oltre la soglia del
40% di `handle_missing_values`. Il blocco 4.8 calcola quindi la **quota di round
dell'anno con importo non dichiarato** (0 negli anni senza round, mai nulla):
e' una delle **due** colonne che non vengono da `example_panel.csv` —
l'altra e' `N_Similar`, aggiunta il 2026-09-17 — e la verifica delle invarianti
le mette in conto. Per tornare indietro basta togliere
la colonna.

**Fatto a valle il 2026-09-17.** `src/preprocessing.py` seleziona ora
`TotalRaised` e `UndisclosedAmountShare`. Nella storia completa
(`build_full_history_dataset`) i due valori si aggregano in modo diverso:
`TotalRaised` con `sum()` come prima, la quota con `mean()` sugli anni
dell'azienda, perche' sommare una quota non vorrebbe dire niente e tenerne
l'ultimo anno nemmeno.

Nello stesso passaggio sono spariti i due punti che dipendevano dalle colonne
`*_All`, eliminate dal panel: il `drop` in coda a `build_windowed_dataset` e il
`drop` + `rename` in `build_full_history_dataset`. Il dataset senza finestra si
ottiene ora ricostruendo il panel con `TEMPORIZZA_COMPETITOR` spento, non
leggendo colonne parallele. *Provata in memoria la catena completa di `scripts/build_datasets.py` sul panel
corrente, senza scrivere niente: `build_windowed_dataset` 32.752 x 56 ->
`preprocess_dataset` 30.270 x 48 -> `build_full_history_dataset` 30.270 x 52 ->
`preprocess_dataset(flag_no_time_window=True)` 30.270 x 49, e l'allineamento
finale `select(window.columns)` funziona (30.270 x 48). `TotalRaised` e
`UndisclosedAmountShare` sono presenti in entrambi i dataset, `TotalRaised_Est`
in nessuno dei due. `UndisclosedAmountShare` sopravvive alla soglia del 40% di
`handle_missing_values`, che invece avrebbe scartato `TotalRaised_NA`.*

**`N_Similar` non e' stata aggiunta alle feature**: e' una scelta di modellazione,
non una conseguenza di M13, e va decisa a parte.

**Attenzione al percorso di input.** `scripts/build_datasets.py` legge il panel da
`paths.raw_dataset`, che in `config/config.yaml` punta a `data/raw/panel.csv.gz`:
NON e' il panel prodotto da questa pipeline (`data/interim_light/panel.csv.gz`),
ma il vecchio output del notebook di temporizzazione dei competitor, con 122
colonne, ID rinumerati e un'estrazione precedente. Va corretto prima di
ricostruire i dataset, altrimenti il run e' a vuoto.

**2026-09-16 — passaggio 4 della riparazione delle date, esteso (M14).** I
quattro passaggi sono stati messi alla prova sui round che una data ce l'hanno,
fingendo che mancasse:

| passaggio | casi di prova | stima esatta | entro 1 anno |
|---|---:|---:|---:|
| 1 — data del fallimento | 20.956 | 94,5% | 97,6% |
| 2 — data dell'acquisizione | 11.536 | 90,4% | 92,2% |
| 3 — primo round all'anno di fondazione | 67.542 | 30,9% | 59,6% |
| 4 — media fra i round vicini | 116.976 | 45,9% | 85,9% |

**I passaggi 1, 2 e 3 restano invariati.** Il 3 anticipa in media di 1,85 anni
(l'eta' reale al primo round ha mediana 1 e vale 0 solo nel 30,9% dei casi) e
decide anche chi entra nel campione, perche' schiaccia 23.492 round sull'eta' 0;
l'alternativa migliore misurata sarebbe «meta' fra fondazione e round
successivo» (76,1% entro un anno contro 59,6%), **ma la regola e' una scelta
dell'economista che ha scritto la pipeline e si tiene**.

**Il passaggio 4 e' stato riscritto**, perche' aveva un limite tecnico: guardava
solo la riga immediatamente precedente e successiva, quindi non faceva niente
quando i round senza data erano due o piu' di fila. Ora cerca il round datato
**piu' vicino** prima e dopo e distribuisce i round del buco **uniformemente**
nell'intervallo, invece di dare a tutti la stessa stima.

*Verificato sui round datati: con un buco da 2 round la stima esatta passa dal
37,3% al 48,7% (entro un anno dall'81,1% all'88,9%); con un buco da 3 dal 31,0%
al 44,6%. Con un solo round nel buco la formula coincide con quella dell'R,
quindi gli 11.757 round gia' riparati non si muovono: verificato, zero round
datati cambiano anno.* I limiti restano round veri: fondazione e ultimo anno
dell'azienda non si usano.

**Effetto:** 3.233 round recuperati in 1.453 aziende (435 round VC); i round
senza data scendono da 16.860 a **13.627**.

| | prima | dopo |
|---|---:|---:|
| righe del panel | 802.193 | 802.148 |
| righe con stadio | 561.195 | 561.333 |
| righe con dati di team | 749.892 | 749.847 |
| aziende nel dataset dei modelli | 32.735 | 32.752 |
| target Later | 6.549 | 6.588 |
| target Early | 13.987 | 13.958 |

**Cosa resta (M25):** 13.627 round senza data, di cui 11.561 dopo l'ultimo round
datato, 1.213 prima del primo e 853 in aziende senza nessun round datato (le 757
che spariscono dai modelli). Per questi servirebbero limiti artificiali
(fondazione, ultimo anno), che nel banco di prova danno circa il 30% di stime
esatte. *Verificato che le altre colonne di data di `Deal.csv` non aiutano:
`AnnouncedDate` e' valorizzata solo sullo 0,9% dei round senza `DealDate`,
`FiscalYear` sullo 0,2%, le date di registrazione su una manciata di casi.
Verificata anche la data di uscita o fallimento come limite destro: copre solo il
2,5% dei casi e, dove c'e', non e' migliore dell'ultimo anno di dati (23,4% di
stime esatte contro 22,7%, con un bias peggiore), perche' i due limiti coincidono
quasi sempre: `OwnershipStatusDate` e' una delle sei date che formano `MaxYear`.*

**Decisione: ci si ferma qui, e si registra chi perde i round.** Il blocco 4.5bis
scrive `aziende_round_senza_data.parquet` — una riga per azienda con
`round_totali`, `round_senza_data`, `round_vc_senza_data` e `perde_tutti` — fuori
dal panel, che non cambia di una riga ne' di una colonna. *Misurato: **12.042
aziende**, di cui **757 perdono tutti i round**; 13.627 round persi, di cui 3.380
VC.* Serve al controllo di robustezza dell'articolo.

**Cosa resta di M0.** Solo il CEO ricavato dal board team (vedi le decisioni
aperte). Gli altri due residui sono stati chiusi il 2026-09-17, qui sotto.

### Registro: i residui di M0, 2026-09-17

**1. L'anno dei ruoli senza data.** La cascata che data ogni ruolo aveva tre
livelli: data vera (77,3% degli eventi), anno di fondazione dell'entita' (18,8%),
altrimenti **anno 0**, cioe' "conta in ogni anno del panel" (3,9%). Ora i livelli
sono quattro: al posto dell'anno 0 c'e' il **primo anno noto della persona**, il
piu' antico fra gli anni risolti ai primi due livelli sui suoi altri ruoli.
L'anno 0 sopravvive solo per chi non ha nemmeno un altro ruolo datato.

*Perche' il livello 3 non poteva usare la fondazione: le entita' che ci finiscono
non hanno un anno di fondazione da nessuna parte. Misurato: il **93,1%** non sta
ne' in `Company.csv` ne' in `Investor.csv` (sono aziende fuori dalla nostra
estrazione, o entita'-persona come gli angel), il 6,9% c'e' ma ha `YearFounded`
vuoto, e **nessuna di loro e' un'azienda del panel**.*

*Effetto misurato: si spostano **35.947 eventi** (18.788 posizioni, 12.593 seggi,
4.566 altri ruoli) per 23.899 persone, il 6,0% del totale; la categoria "da
sempre" si svuota del tutto, perche' ogni persona con un ruolo senza data ne ha
almeno un altro datato. `esperienza_persona_anno` passa da 663.769 a **639.681
righe**. `WorkExp_Idx_Mean` va da 0,0588 a 0,0593, `Total_People` non si muove
(conta persone, non ruoli), il panel resta 802.148 x 52.*

*Controllo decisivo: **2a.3ter resta verde** — l'ultima riga dei conteggi coincide
con le tabelle per tutte le 404.468 persone, e le differenze con `Person.csv`
sono sempre le stesse 7 note. La modifica ha spostato gli ANNI, non il NUMERO di
ruoli.*

**La regola dei Founder era gia' applicata.** Un ruolo da Founder senza data cade
nel livello 2 e viene datato alla fondazione dell'azienda, che e' esattamente
"dal giorno 0". *Verificata anche la versione forte - forzare la fondazione pure
dove una data dichiarata c'e' - e scartata: sui 237.009 ruoli da founder datati
con fondazione nota, l'**89,1%** coincide gia', il 6,8% e' posteriore (si
sposterebbero 16.031 eventi) e il **4,2% e' anteriore** alla fondazione, e li'
forzare la regola cancellerebbe esperienza dichiarata.*

**2. La standardizzazione guardava tutti gli anni.** Media e deviazione standard
di `log1p` erano **un unico paio di numeri** calcolati su tutte le coppie
(persona, anno) del panel: il valore di una riga del 2005 dipendeva anche dalle
righe del 2020, cioe' un fit dello scaler sull'intero dataset, futuro compreso.
Ora sono **per anno, a finestra espansiva** (solo gli anni <= a quello della
riga, anno corrente incluso), calcolate in 2b.1bis e riusate in 5.6 per l'indice
del CEO agganciandole sull'anno della riga. Solo sul ramo temporizzato: a
interruttore spento la popolazione e' (azienda, persona) e non ha un asse
temporale su cui espandere la finestra.

*Verifica del metodo: all'ultimo anno la finestra contiene tutta la popolazione,
quindi deve riprodurre i vecchi parametri unici - e li riproduceva alla sesta
cifra (0,692745 / 0,320527 per le Posizioni, 0,110951 / 0,434557 per gli
AltriRuoli). Misurato prima che cambiasse anche la cascata degli anni: la
proprieta' resta, i valori di oggi differiscono perche' sono cambiati gli eventi
sottostanti. Ne segue che il cambiamento e' **confinato agli anni iniziali** e si
annulla verso la fine del panel.*

*Quanto contava: nel 2000 la popolazione e' di 4.467 coppie contro 3.394.785
all'ultimo anno, e la deviazione standard di `AltriRuoli` vale 0,194 contro
0,435 - le righe di inizio panel erano standardizzate con una scala **2,2 volte**
piu' larga di quella dei loro contemporanei, quindi risultavano troppo basse.
`WorkExp_Idx_Mean` per anno solare va ora da -0,0229 nel 2000 a +0,0984 nel 2024.*

**Non misurato:** il confronto riga per riga prima/dopo. Il panel precedente e'
sovrascritto e i conteggi grezzi vengono scartati dopo il calcolo dell'indice;
servirebbe una riesecuzione con il codice vecchio.

### Registro: fase 7, competitor, 2026-09-17

**L'interruttore.** `TEMPORIZZA_COMPETITOR` si affianca a `TEMPORIZZA_PERSONE` e
`TEMPORIZZA_INVESTITORI`, default acceso. **Le tre colonne `*_All` non esistono
piu'**: la terna competitor e' una sola e il flag ne decide il contenuto. Acceso
= concorrenti vivi quell'anno, con la controparte filtrata su `Company.csv`;
spento = ogni informazione disponibile, nessuna finestra e nessun filtro.

| | acceso | spento | R originale |
|---|---:|---:|---:|
| `N_Competitors` | 0,238 | 1,311 | 1,263 |
| `Same_Country` | 0,071 | 0,141 | (era booleana) |
| `SimilarityScoreMean` | 54,524 | 90,751 | 90,97 |
| righe con >= 1 concorrente | 12,3% | 21,3% | 20,7% |
| `N_Similar` medio | 1,127 | 9,992 | — |

*Verificato eseguendo il notebook su entrambi i rami: 802.148 righe x 52 colonne
e invarianti verdi in tutti e due, righe e aziende identiche. Il flag sposta
solo i tre valori, che e' la proprieta' che serve all'ablazione.*

**Perche' le `_All` sono sparite.** Erano calcolate **dopo** il filtro sulla
controparte, quindi non erano la versione statica di niente: `N_Competitors_All`
valeva 0,297 contro 1,263 dell'R, un fattore 4,3. Presentarla come «la versione
senza temporizzazione» avrebbe attribuito alla temporizzazione un effetto che
era in gran parte copertura del dato. Ora il ramo spento riproduce l'R.

**Il paese della controparte** si legge da `SimilarCompanyHQCountry` nella
tabella delle relazioni, non piu' da `Company.csv`: *e' presente nel 99,57%
delle righe e sulle 228.894 coppie verificabili coincide con `Company.csv` sul
100%*. Cosi' `Same_Country` resta calcolabile anche fuori estrazione.

**`N_Similar`** (voce M22) e' il denominatore di `SimilarityScoreMean`.
Additiva: eliminarla riporta allo schema a 51 colonne. Acceso vale 0 nel 41,3%
delle righe — esattamente dove la media e' fabbricata — e 1 o 2 nel 76% delle
righe in cui un valore c'e'. Spento e' **di fatto costante**: 10 su 801.507
righe e 0 su 641, le uniche aziende a cui PitchBook non attribuisce nessuna
simile.

**M2, vintage dell'estrazione.** Le colonne competitor di `data/raw/panel.csv.gz`
vengono da un download piu' vecchio. *Misurato ricalcolandole sugli stessi
anni-azienda con l'estrazione di oggi: `N_Competitors` identica sul 94,5% delle
righe (correlazione 0,896), `Same_Country` sul 98,0%, `SimilarityScoreMean` solo
sul 35,7% (correlazione 0,492). Ma la scomposizione assolve i dati: sulle
377.902 righe con un valore in entrambe le estrazioni il punteggio si sposta di
**1,05 punti su 100** (correlazione 0,907), e il crollo viene tutto dalle
**227.625 righe** che attraversano il confine 0 / non-0.* L'instabilita' era
nella convenzione, non nella sorgente.

**Attenzione a `data/raw/panel.csv.gz`:** non e' il panel R. E' l'output del
notebook di temporizzazione dei competitor di Giulio (122 colonne, 882.324
righe, `Same_Country` gia' come conteggio, ID rinumerati). Il panel R puro e'
`data/reference/db_master_panel.csv.gz`, 121 colonne. Vanno rinominati o
dichiarati: e' fra le decisioni aperte.

## Decisioni prese


1. **M11 — l'imputazione RandomForest è eliminata**, non sospesa. Il modello è
   sbagliato alla radice (nessun `set.seed`, addestrato su tutti gli anni,
   fittato prima di qualsiasi split, con il tipo di deal fra i predittori
   mentre il tipo di deal determina il target). Le sette colonne `*_Est` non
   esistono in nessun output.
2. **M13 — la colonna che sostituisce `TotalRaised_Est` in
   `src/preprocessing.py` è `TotalRaised`.** *Codice aggiornato il 2026-09-17*
   (registro della fase 4). La rigenerazione di `data/processed/` e dei run
   resta deliberatamente sospesa: si fa **una volta sola**, quando la pipeline
   del panel sarà completa.

3. **2026-09-14 — fase 2 del leggero**: deduplica, `EndDate` all'ultimo anno
   di vita, taglio del panel a `MaxYear`. Dettaglio e numeri nel registro qui
   sopra. **I dataset in `data/processed/` non sono stati rigenerati**: il
   panel ha 78.280 righe in meno e il target si legge su un `LastAge` diverso.

## Decisioni aperte

1. **Voce M0**: gli attributi delle persone non sono temporizzati. È la prima
   del riepilogo per priorità e l'intervento più costoso; le fonti datate
   (`PersonPositionRelation`, `PersonEducationRelation`) ci sono. Da valutare
   a sé.
2. **Voci M14 e M25**: il 10,9% delle date dei deal è inventato e il 5% dei
   deal esce dal panel senza traccia. Tre opzioni, si decide alla fase 4.
3. **Voce M2**: le colonne competitor di `data/raw/panel.csv.gz` vengono da un
   altro download di `CompanySimilarRelation.csv` — *confermato da Giulio il
   2026-09-17: quel notebook girava su un'estrazione precedente*. L'effetto e'
   misurato (registro fase 7): sui punteggi vale 1,05 punti, il resto e' il
   confine 0 / non-0. Il checkpoint F non si chiudera' mai su quel file; da
   decidere se rinominarlo, visto che **non e' nemmeno il panel R**.
4. **Voci X11/X23**: la mappa Europa serve solo al checkpoint B. Si elimina
   quando la fedeltà smette di essere l'obiettivo.

## I file da leggere, in ordine

1. `docs/panel_errori_logici.md` — 101 voci numerate: cosa è sbagliato,
   discutibile o inutile, fase per fase. **È il documento di lavoro.**
2. `build_panel.ipynb` — la pipeline spiegata passaggio per passaggio.
3. `docs/superpowers/specs/2026-09-04-panel-pipeline-r2py-design.md` —
   il disegno della traduzione, con l'emendamento del 2026-09-09.
4. `git log --oneline` su `src/panel/` — un commit per stadio, con la
   motivazione di ogni scelta non ovvia.
