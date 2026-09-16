# Panel — stato della revisione fase per fase

Ultimo aggiornamento: 2026-09-15 (fase 2 del notebook leggero: board team,
finestre di presenza, lunghezza del panel, e M0: esperienza e istruzione
temporizzate)

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
| 4 | M12 `Zero_Invested` · M14 date inventate · M25 deal che evaporano | da fare |
| 5 | M1 `GrowthStage` mescola stato e storia · M16 | da fare |
| 6 | M18 troncamento | da fare |
| 7 | M20, M21, M22, M23, M24 · M2 vintage | da fare |
| trasversale | **M0** attributi delle persone non temporizzati | **fatto** (2026-09-15): esperienza e istruzione, team e CEO · il CEO dal board team resta da valutare |
| dopo | M13: `TotalRaised` in `preprocessing.py`, rigenerare i dataset e i run | da fare |

**Cadute con le fasi 3a e 3b**, che nel leggero non esistono: B4, B8, X11/X23
(la mappa Europa e `scripts/derive_europe_mapping.py`), X13, X14, T19-T25.
**Cadute con le colonne eliminate**: B2 (`Is_Other`), B3 e X21 (`StageBlock`),
X22, M17, M19, X15-X20.

## `build_panel_light.ipynb` — la pipeline di lavoro

**Dal 2026-09-11 `build_panel.ipynb` e' un artefatto congelato**: ha dimostrato
che la traduzione dall'R e' fedele (sei checkpoint verdi, quattordici confronti
a divergenza zero) e non si tocca piu'. La pipeline su cui si lavora e'
`build_panel_light.ipynb`.

Produce **solo le 53 colonne di `data/raw/example_panel.csv`** — con
`TotalRaised` al posto di `TotalRaised_Est` — e scrive in `data/interim_light/`
per non toccare i parquet del panel completo. Ogni riga e' commentata.

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
| righe | 802.193 |
| aziende | 116.312 |
| colonne | 53 |
| righe con dati di team | 749.892 |
| righe con `GrowthStageGroup` | 561.195 |
| righe con `Age < 0` | 0 |
| tempo di esecuzione | 108 s, picco 1,76 GB |

*Aggiornato il 2026-09-14, dopo le correzioni della fase 2 (registro qui sotto).*

**Prossimo passo: riprendere le correzioni dal catalogo**, in ordine di
pipeline, sul notebook leggero. Le voci ancora aperte che lo riguardano sono
~12; quelle legate alle fasi 3a e 3b sono cadute con le fasi stesse.

| | completo | leggero |
|---|---:|---:|
| celle di codice | 91 | **45** |
| CSV letti | 13 | **9** |
| colonne di `Company.csv` | 39 | **11** |
| colonne di `db3` | 54 | **10** |
| aggregati di team | 23 | **13** |
| colonne del panel | 117 | **53** |
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

**Cosa resta di M0.** I ruoli senza data contano dall'anno di fondazione
dell'entita' (limite inferiore) o da sempre; la standardizzazione usa tutte le
coppie (persona, anno), quindi anche anni successivi di altre aziende; il CEO
dal board team e' rimandato (vedi le decisioni aperte).

## Decisioni prese


1. **M11 — l'imputazione RandomForest è eliminata**, non sospesa. Il modello è
   sbagliato alla radice (nessun `set.seed`, addestrato su tutti gli anni,
   fittato prima di qualsiasi split, con il tipo di deal fra i predittori
   mentre il tipo di deal determina il target). Le sette colonne `*_Est` non
   esistono in nessun output.
2. **M13 — la colonna che sostituisce `TotalRaised_Est` in
   `src/preprocessing.py` è `TotalRaised`.** La sostituzione si fa **dopo**
   aver finito di correggere il notebook, in un passaggio solo: comporta
   rigenerare `data/processed/` e tutti i run.

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
3. **Voce M2**: le sei colonne competitor di `data/raw/panel.csv.gz` vengono da
   un altro download di `CompanySimilarRelation.csv`. Se salta fuori
   l'estrazione giusta, il checkpoint F si chiude.
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
