# Panel — stato della revisione fase per fase

Ultimo aggiornamento: 2026-09-10 (fase 1 inlinata, in attesa di revisione)

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

| fase | modulo | nel notebook | stato |
|---|---|---|---|
| 1 — aziende, affiliate, scheletro | *cancellato* | **sì**, blocchi 1.1–1.10 | **in revisione** |
| 2a — tabella persona-azienda | `stage2_team.py` | no | da fare |
| 2b — colonne di team | `stage2_team.py` | no | da fare |
| 3a — competitor | `stage3_relations.py` | no | da fare |
| 3b — dipendenti, financials, news | `stage3_relations.py` | no | da fare |
| 4 — deal e investitori | `stage4_deals.py` | no | da fare |
| 5 — finalizzazione | `stage5_final.py` | no | da fare |
| 6 — raggruppamento e troncamento | `stage6_panel.py` | no | da fare |
| 7 — competitor temporizzati | `stage7_competitors.py` | no | da fare |

Punto di partenza: la traduzione è completa e verificata. Sei checkpoint,
tutti verdi, in circa 8 minuti.

**Fase 1** — inlinata in dieci blocchi (1.1 lettura e segnaposto NA, 1.2 flag
di presenza, 1.3 date e numeri, 1.4 `FiscalDate`, 1.5 `db1` non filtrato, 1.6
affiliati, 1.7 join e flag `Has_*`, 1.8 `MaxYear` e scheletro, 1.9 variabili
time-varying, 1.10 selezione, filtro e scrittura). Modulo
`src/panel/stage1_company.py` cancellato; `seq()` di R è diventata
`rutils.r_seq` col suo test, e le cinque colonne data sono passate in
`io.COMPANY_DATE_COLUMNS` perché servono anche alla fase 7 e due liste che
divergono darebbero due `MaxYear` diversi.
Riesecuzione da zero: `db_master_1` a 116.920 righe, **28 colonne su 28
identiche** al riferimento, i 245 `Delta` negativi su 106 aziende del bug B6 al
loro posto, e il checkpoint A ancora verde a valle. In attesa della revisione
di Giulio.

## Decisioni aperte

1. **Bloccante, voce M13.** `src/preprocessing.py` selezionava
   `TotalRaised_Est`, che non esiste più perché l'imputazione RandomForest è
   sospesa. Va sostituita con `TotalRaised` (zero dove l'importo non è
   dichiarato) oppure con `TotalRaised_NA` (nulla lì, così l'imputazione che
   gira prima del training vede il buco). Entrambe vengono prodotte. In ogni
   caso i due dataset processati e tutti i run vanno rigenerati.
2. **Voce M0**, la più grave del catalogo: gli attributi delle persone non sono
   temporizzati. Se e come correggerla è da valutare a sé, è l'intervento più
   costoso.
3. **Voce M2**: le sei colonne competitor di `data/raw/panel.csv.gz` vengono da
   un altro download di `CompanySimilarRelation.csv`. Se salta fuori
   l'estrazione giusta, il checkpoint F si chiude.

## I file da leggere, in ordine

1. `docs/panel_errori_logici.md` — 101 voci numerate: cosa è sbagliato,
   discutibile o inutile, fase per fase. **È il documento di lavoro.**
2. `build_panel.ipynb` — la pipeline spiegata passaggio per passaggio.
3. `docs/superpowers/specs/2026-09-04-panel-pipeline-r2py-design.md` —
   il disegno della traduzione, con l'emendamento del 2026-09-09.
4. `git log --oneline` su `src/panel/` — un commit per stadio, con la
   motivazione di ogni scelta non ovvia.
