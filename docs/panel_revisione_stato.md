# Panel — stato della revisione fase per fase

Ultimo aggiornamento: 2026-09-10 (tutte le fasi inlinate e verificate identiche)

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
