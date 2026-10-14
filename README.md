# PRC Data Challenge 2026

**Team:** Charming Eel

**Performance:** RMSE 214.7479 on the live leaderboard (final submission)

## Overview

This repository contains the complete pipeline of our submissions to the PRC Data Challenge 2026: one script, `taxi/main.py`, downloads every input, builds every feature table, trains every model and writes the file for the [final ranking](#final-ranking).

**Key Features:**
* **Three recording regimes:** A label is not always a taxi time. Some block times are copies of the schedule (P), some carry the previous day's date (S); the prediction is a mixture `pP * msched + pS * (86400 + v) + (1 - pP - pS) * v` of the three.
* **ADS-B surface tracks:** Every movement is matched to its aircraft in the adsb.lol archive in three passes, and about 190 features describe what the aircraft did on the ground before the take-off sequence.
* **Evidence from the other movements:** take-off sequences, out-of-month typical taxi times and what the neighbouring take-offs show, per runway and airport.
* **Diverse network ensemble:** 160 neural network jobs (ResNet, TabM, RealMLP, neighbour attention), mixed by manual and greedy-selected ensembles and corrected by XGBoost second stages trained on held-out week folds.
* **Corrections after the models:** a coverage-outage blend (Rome lanes), an airport-month calibration from EUROCONTROL's published monthly aggregates of the scored months, support caps and a Rome P / S / N model for the late departures without an NM link, each checked three ways on a 2025 out-of-fold pipeline.

## Libraries & Tools

The project relies on a small, pinned stack (`requirements.txt`); the models' outputs depend on the exact versions.

* **Data Processing:** `pandas`, `numpy`, `pyarrow` (parquet), `scipy`.
* **Gradient Boosting:** `xgboost` (value models, regime classifiers, second stages), `catboost` (value and classifier variants), `scikit-learn` (logistic regressions of the Rome corrections, metrics).
* **Neural Networks:** `torch`, `rtdl_num_embeddings` (piecewise-linear, periodic and PBLD embeddings), `tabm` (TabM).
* **Optimisation:** `optuna` (TPE studies of the tuned classifiers).
* **Data Access:** `minio` (the organisers' bucket) and every other source is fetched over plain HTTP.

## Project Structure

```
├── README.md, requirements.txt
└── taxi/                          # the code, the data and the results; one package (python -m taxi.<module> --help)
    ├── main.py                    # The whole pipeline
    ├── features/
    │   ├── prior_tables.py        # Step 2: the prior model's tables (also the first columns of the feature matrix)
    │   ├── matrix_tables.py       # Step 3: departure matrix, X-Plane geometry, arrival proxies
    │   ├── adsb_tracks.py         # Steps 4-12: ADS-B surface tracks: three matching passes, the stand map, adsb_pass1/2/3
    │   ├── feature_blocks.py      # Step 13: network feature blocks and the airport-state tables
    │   ├── airframe_turnaround.py # Step 14: aircraft turnaround of every Rome departure
    │   └── network_inputs.py      # Stage 2: network input caches (quantile-normalised numbers, missing flags, codes)
    ├── models/
    │   ├── regime_mixture.py      # Recording-regime classifiers, tree value models, stake-weighted classifiers, Rome prior
    │   ├── tuned_classifiers.py   # Optuna-tuned schedule-copy classifiers (Rome NM-linked rows, rows without a link)
    │   ├── tree_value_models.py   # Residual two-part model and CatBoost value model (full fits and week-fold runs)
    │   ├── neural_network.py      # One neural value model configuration (ResNet / TabM / RealMLP, attention, two-hot)
    │   ├── network_jobs.py        # The 160 network training jobs (240 runs)
    │   ├── network_ensembles.py   # Member mixes, ensembles and second stages
    │   ├── prior_model.py         # The prior: an older, independent model in eight steps, run in memory (v8 alone)
    │   ├── out_of_fold.py         # 2025 out-of-fold reconstruction of the building blocks
    │   ├── rome_regime_stack.py   # The P / S / N model of the late Rome departures
    │   ├── aggregate_calibration.py # Airport-month shifts from EUROCONTROL's monthly taxi-out aggregates
    │   └── predictions/, weights/, tune/   # the runs, saved models and tuned classifiers (stage 2's working files)
    ├── prediction/
    │   ├── build_submission.py    # Trained runs + the prior -> ensembles, second stages, lanes -> submission.parquet
    │   ├── late_rome_departures.py # The Rome recording mechanism and the late-Rome corrections
    │   ├── submission.parquet     # The final file (all four months), written by the last model step
    │   └── submission_initial_months.parquet   # its January / July rows (the months the live leaderboard scores)
    └── data/                      # Created by taxi/main.py on its first run, before anything is downloaded
        ├── downloads/             # all external data
        └── features/              # every feature table
```

## Usage

### 1. Environment Setup

* Linux, Python 3.12 (virtual environment with requirements.txt packages installed), one NVIDIA GPU with 8 GB, 32 GB of RAM, about 70 GB of free disk for the data, the features and the model outputs. Note that running this code WILL fail if you don't meet these requirements. Some cases like OOM errors are easily fixable on your machine, in which case you can decrease the model memory consumption (for example, reduce PREDICTION_BATCH_ROW_MEMBER_BUDGET), but won't be necessary if you meet the requirements.

### 2. Connecting to the Competition Bucket (MinIO)

The challenge files (the 2025 training months, the ranking files and the submission templates) live in the organisers' MinIO bucket `prc-2026-datasets`. `taxi/main.py` connects with the `minio` Python package, which is already in `requirements.txt`, and reads the endpoint and keys from the MinIO client configuration `~/.mc/config.json`, alias `opensky`.

1.  Take the endpoint URL, access key and secret key that the organisers sent to your team.
2.  Create the configuration, either with the MinIO client `mc`:
    ```bash
    mc alias set opensky <ENDPOINT_URL> <ACCESS_KEY> <SECRET_KEY>
    ```
3.  Verify the connection and download the files into `taxi/data/downloads/competition/` (files already there are skipped) - this will automatically happen if you run `main.py` if you don't
    ```bash
    python taxi/main.py --download-only competition
    ```

### 3. Pipeline Execution

Run everything from the repository root. Every step runs as its own process, one at a time.

**Full pipeline**
```bash
python taxi/main.py
```

**Step 1: Downloads**
Competition files, adsb.lol, OPDI, EUROCONTROL aggregates, ERA5 weather, METAR and X-Plane layouts
```bash
python taxi/main.py --download-only                     # every source, in order
python taxi/main.py --download-only eurocontrol metar   # only these sources
```

**Step 2: Feature Tables**
Prior tables, departure matrix, the three ADS-B matching passes, the feature blocks and the aircraft turnaround
```bash
python taxi/main.py --data-only                         # downloads + every table
python taxi/main.py --data-only --from-step prior_tables   # every table, from that step on
```

**Step 3: Models and Submission**
The network input caches, 160 network jobs, the regime fits, the tuned classifiers, the tree models, the out-of-fold steps and finally the submission. A step whose outputs exist is skipped, so an interrupted run resumes where it stopped. This includes the submission step, so delete `taxi/prediction/submission.parquet` to rebuild it after retraining.
```bash
python taxi/main.py --models-only                       # stage 2 only; the runs, weights and tuned classifiers are kept afterwards
python taxi/main.py --models-only --no-keep-outputs     # delete them once the submission is written
python taxi/main.py --list                              # every step, with the model steps already done marked 'done'
```

## Glossary

Terms used in the code, the column names and this README, for readers who do not work in aviation.

| Term | Meaning |
|---|---|
| ADEP / ADES | departure / destination airport, as a four-letter ICAO code (e.g. LIRF = Rome Fiumicino, LTFM = Istanbul, EDDF = Frankfurt) |
| ICAO code | the four-letter airport code of the International Civil Aviation Organization (airlines' three-letter codes are ICAO codes too) |
| stand | the parking position where an aircraft boards; it leaves the stand at off-block |
| off-block, block time | the moment the aircraft leaves its stand (the blocks in front of the wheels are removed); the label is measured from it |
| take-off | wheels leaving the runway; the challenge files give its time (`MVT_TIME_UTC_mvt`) |
| taxi-out time | take-off minus off-block, in seconds: the quantity predicted (`TAXITIME_SEC_mvt`) |
| taxi-in time | landing to on-block of an arriving flight; used only as context (e.g. how busy the stands were) |
| schedule (`SCHED_TIME_UTC_mvt`) | the timetable departure time; `msched` = take-off minus schedule, in seconds |
| LOBT | last known off-block time in the flight plan; filed before the flight, so it is known for ranking rows |
| AOBT | actual off-block time as reported to the Network Manager (`AOBT_3_flt`): a noisy proxy of the label |
| NM, NM link | EUROCONTROL's Network Manager (flight-plan data); a departure has an NM link when the challenge files join it to a flight plan |
| regime P / S / N | how a label was recorded: P = the block time is a copy of the schedule, S = the actual time with the previous day's date (labels near 86,400 s), N = a normal block time |
| runway, runway end | the strip used for take-off; airports with several runways use different taxi routes to each |
| typical taxi times (`te_*`) | mean taxi times of groups such as the stand, the runway or stand x runway, measured on other months (so a row never sees its own label) |
| take-off sequence | the order of departures on a runway; queue length and gaps between take-offs (`sq_*` columns) |
| neighbours (`nb_*`) | the take-offs at the same airport around the same time, and what their NM and ADS-B taxi times show |
| ADS-B | position reports broadcast by aircraft transponders; adsb.lol archives them (latitude, longitude, ground speed, on-ground flag) |
| transponder address, `icao` | the aircraft's 24-bit ADS-B identity (hex), used to follow one airframe through the day |
| callsign | the flight identifier broadcast in ADS-B (e.g. AZA1234), matched to the flight number |
| pass 1 / 2 / 3 | the three ADS-B matching passes: pass 1 by callsign (or the only take-off within 90 s), pass 2 the closest unused take-off within 120 s, pass 3 the same among aircraft of the departure's type |
| METAR | the routine airport weather report (wind, visibility, precipitation codes such as FZ = freezing, SN = snow) |
| ERA5 | hourly reanalysis weather (Open-Meteo), a gap-free complement to the METAR reports |
| OPDI | EUROCONTROL / OpenSky open flight data: flight lists and surface events (e.g. when an aircraft entered a taxiway) |
| X-Plane gateway | community airport layouts with stand positions and runway ends, used to place ADS-B points on stands |
| airframe turnaround | the time an aircraft spent on the ground between its arrival and this departure |
| run, fold | a model's predictions on every departure; fold `wkN` holds out the 2025 departures whose ISO week number modulo 4 is N, `full` trains on all of 2025 |

## Methodology

### 1. Data

| Source | What is used | Licence |
|---|---|---|
| Challenge files (organisers) | 12 monthly training files of 2025 (2.08 M departures), 2 ranking and submission files each | N/A |
| [adsb.lol globe_history](https://www.adsb.lol/docs/open-data/historical/) | one readsb archive per UTC day, reduced to the points within about 11 km of the ten airports, on the ground or below 4,000 ft | ODbL 1.0 |
| OPDI (EUROCONTROL / OpenSky) | monthly flight lists and 10-day flight-event files | EUROCONTROL copyright: copy with source, non-commercial; not an open licence (see below) |
| EUROCONTROL monthly taxi-out / taxi-in additional time (`www.eurocontrol.int/performance/data/download/csv/`) | per airport and month of 2025 and 2026: flights, valid flights, reference and additional minutes | EUROCONTROL data notice: non-commercial use, not modified; downloaded, not redistributed; not an open licence (see below) |
| ERA5 via Open-Meteo | hourly weather at each airport | CC BY 4.0 |
| METAR (Iowa Environmental Mesonet) | reports per airport | public domain |
| X-Plane Scenery Gateway | pinned airport layouts (stands, taxiway network, runway ends) | GPL v2+ |

Licences: two sources are not openly licensed, while the challenge's eligibility page asks for additional data under an open licence. OPDI and EUROCONTROL aggregate data is used on an informal confirmation by an organiser (E. Spinielli) on the challenge's Discord server.

### 2. The Target and its Three Recording Regimes

`TAXITIME = take-off time - recorded off-block (block) time`.

* **Problem:** The NM actual off-block time (`AOBT_3_flt`) is a noisy proxy: only about 20 % of NM AOBTs lie within 60 s of the airport's block time. Block times are minute-rounded with independent +-6 s jitter. And not every label is a taxi time at all. The training labels show three ways a block time gets recorded.
    1. **P, schedule copy:** the block equals the scheduled minute (within 6 s), so the label is `take-off - schedule`, which can be hours. It depends strongly on the operator (some carriers at 90 % or more, others at 0 %).
    2. **S, previous date** (Rome only): long-delayed flights departing on a later UTC day carry the actual time of day with the schedule's date, so the label is about 86,400 s + the taxi time.
    3.  **N, normal:** an ordinary block time.
* **Solution:** We predict the regime and the ordinary taxi time separately and combine them as a mixture (see [modeling strategy](#4-modeling-strategy)).
* **Why it matters:** rows without an NM link produce 30-50 % of the squared error, so this should be reduced as much as possible.

### 3. Feature Engineering

Going through all the feature engineering here would be a really arduous ordeal, so here is a brief overview:

A feature is one clue about a flight, written as a number. We made hundreds, and the most useful ones we found were:

* When ADS-B saw the plane leave its spot. At airports with good ADS-B, this is within one minute of the real answer for about 8 out of 10 flights. The flight plan's guess is that close only for 2 out of 10.
* What the plane did on the ground: when it stopped, how long it waited, how far it still had to go, about 190 clues.
* What the other planes were doing: how many planes were queuing for the same runway, and how long the planes leaving just before and after took.
* Typical times for this parking spot and runway, measured on other months, so a flight never sees its own answer.
* Distance from the parking spot to the runway along the taxiways.
* Weather, such as snow, fog and wind.
* Rest time for Rome flights: how long the plane had been parked since it last landed.

No feature uses a departure's own block time or label. 

#### A. Prior Tables (`prior_tables.py`, `taxi/data/features/prior/`)
* Categoricals; calendar and clock parts of every time field and every pairwise clock difference; baselines.
* Movement counts in 5 / 15 / 60 min windows; ERA5 weather and clock-precision indicators; queue lengths per airport and runway.
* Candidate block clocks (NM fields, the same flight's arrival record, the previous in-block at the stand); the matched OPDI flight; scheduled movements and runway use; OPDI ground events; a table for the rows without an NM link.
* Linked arrivals: the same flight's arrival record at its destination when that is one of the ten airports (closest NM off-block 0-6 h before the take-off).

#### B. Departure Matrix (`matrix_tables.py`)
* `base_features`: the prior's tables side by side, one row per departure in the order of `departure_metadata`.
* X-Plane geometry: stand position, straight-line and shortest taxi-network distance from stand to runway threshold.
* Arrival proxies: how often the operator's / airport's arrivals that day had their block copied from the schedule, and whether the previous arrival at the same stand did; at Rome the latter raises the departure copy rate from 16 % to 39 %.

#### C. ADS-B Surface Tracks (`adsb_tracks.py`)
* **Problem:** The challenge files do not say which aircraft flew a movement, and the ADS-B points carry only a transponder address and a broadcast callsign.
* **Solution:** Each movement is matched to an aircraft in three passes:
    1. By callsign at the airport, confirmed by a runway transition (slow to fast for a take-off, fast to slow for a landing) within 900 s of the recorded movement time; otherwise the only transition of its kind at the airport within +-90 s. Every transition goes to one movement at most.
    2. Departures still unmatched: unused take-off transitions within +-120 s, assigned one to one, closest first, only when the runner-up is at least 20 s farther away.
    3. As pass 2, restricted to aircraft whose readsb type matches the movement's aircraft type.
* **Stand map:** the median parked position of arrivals at their in-block time (arrival block times are known in the ranking data too; at least two observations, spread below 40 m), completed with X-Plane stands where it has none at seven airports.
* **Features:** about 190 per departure describe the surface sequence before the take-off roll: first observation, dwells, leaving the parked position, reception gaps, path still to go, and the state on a time grid from 60 s to 3,600 s before take-off and at the schedule / EOBT / NM AOBT clocks. At well-covered airports "left the stand" alone is within 60 s of the recorded block for 77-83 % of departures, against about 20 % for the NM AOBT. The three tables `adsb_pass1/2/3` add the second- and third-pass departures.

#### D. Feature Blocks: Evidence from the Other Movements (`feature_blocks.py`)
* `sq_*`: the take-off and landing sequence per runway and airport.
* `te_*`: out-of-month target statistics for a hierarchy of keys, shrunk towards the parent key (including `te_unimpeded`, the 20th percentile per stand x runway).
* `or_*`: overtaking.
* `nb_*`, `n2_*`, `nw_*`: what the neighbouring take-offs show (NM taxi, ADS-B taxi, dwell, delay, push delay) over +-5 min to +-12 h.
* `wx_*`: METAR regimes and runway-relative wind.
* The airport-state tables combine some of them for the tree models.

#### E. Aircraft Turnaround (`airframe_turnaround.py`, Rome departures only)
* **Problem:** the challenge files do not identify the aircraft.
* **Solution:** a departure is matched to its OPDI flight (same callsign after removing a doubled operator letter, first seen 10 min before to 40 min after the take-off) and that flight's transponder address to the same aircraft's latest OPDI flight into LIRF: `turn` = first seen of the departure - last seen of the inbound, `inbound_after_sched` = last seen of the inbound - schedule (95 % of Rome departures).
* **Fallback:** on ranking rows where OPDI gives no turnaround (above all on the days without a full flight list: 2-10 and 23 January 2026, partly 24 January), our ADS-B tables give the aircraft's previous landing, shifted by the gap measured on the 2025 departures with both (OPDI's turnaround = ADS-B's - 264 s, 90 % within 300 s); the Rome models are fitted on OPDI turnarounds only.

#### F. Network Inputs (`network_inputs.py`)
* The 250 numerical features with the highest gain in the XGBoost value model, each mapped to a standard normal by its 2025 quantiles (clipped to +-5, missing -> 0), one flag per distinct missingness pattern, and 14 categorical codes.

### 4. Modeling Strategy

#### The Prior (`prior_model.py`)
An older model without ADS-B and without regimes, in eight steps that each replace the last only where the held-out months (January, July 2025) improve: XGBoost on `label - (take-off - NM AOBT)`, a CatBoost specialist for rows without an NM link, OPDI / weather features, no-link CatBoost models (Rome and others), per-airport XGBoost models, an embedding MLP on linked residuals, per-airport models with OPDI ground events. Alone it scored 283.5 (v8).

#### Regime Runs (`regime_mixture.py`)
The mixture prediction is `pred = pP * msched + pS * (86400 + v) + (1 - pP - pS) * v` with `msched` = take-off minus schedule.
* **`pP`:** three XGBoost schedule-copy classifiers (all rows, Rome, rows without an NM link), later averaged with CatBoost classifiers and stake-weighted XGBoost classifiers: a classifier error costs `(msched - v)^2`, so rows are weighted by `(clip(|msched - vbase|, 300, 14400) / 300)`.
* **`pS`:** exists only for the Rome next-day no-link group (68 training rows); there a P / S / N logistic regression on the delay in hours is used, with the P-vs-S split of long delays shrunk toward operator- and flight-level counts.
* **`v`:** the ordinary taxi time, learned on regime-N rows only, as a residual on a schedule-independent baseline `vbase` (take-off - NM AOBT, else the airport median; residuals clipped at +-6000 s); the regime runs average an XGBoost value model with a two-part and a CatBoost variant.
* Two seeds (42, 7). Iterations are fixed: July-held-out early-stopping counts x1.1.
* The regime classifiers, the XGBoost value model and the prior keep the calendar columns (month, day, day of year); the networks, the tree value models and the final tuned classifiers drop them.

#### Tuned Classifiers (`tuned_classifiers.py`)
Optuna TPE studies (25 and 60 trials) of the schedule-copy classifiers for Rome NM-linked rows and for rows without an NM link, minimising the mixture's stake-weighted error in 4-fold month-group cross-validation; the studies keep the calendar columns and the final fits drop them. The Rome classifier of the final file also gets the reference network's taxi time and the implied push delay as inputs.

#### Tree Value Models (`tree_value_models.py`)
* **Residual two-part model:** treats long taxi-outs as a hidden regime (q = P(y - vbase > threshold), separate regressions above and below; 0.7 % of linked rows exceed the NM baseline by more than 20 min and carry 27 % of its squared error).
* **CatBoost value model.**
* Each is trained on all of 2025 and on the four week folds.

#### Networks (`neural_network.py`, jobs in `network_jobs.py`)
* Tabular networks on the network inputs plus feature blocks, trained on regime-N rows to predict `label - offset` (the NM baseline, or `te_unimpeded`), clipped and standardised.
* ResNet (3 x 256) with piecewise-linear, periodic or PBLD embeddings, TabM (k = 16), RealMLP, attention over the 16 or 32 neighbouring take-offs, regression or two-hot heads, AdamW or Muon.
* **Why so many:** single networks are noisy (two seeds differ by 60-70 s rms), so the gain comes from averaging diverse configurations: 160 jobs give 240 runs (four week folds and the full fit per configuration).

#### Ensembles and Second Stages (`network_ensembles.py`)
* **Ensemble:** a mix is a list of members (configurations) with weights; the ensemble is their weighted mean, leaving out the members trained on the unimpeded offset where the reference network predicts more than 6000 s.
* **Second stage:** an XGBoost model trained on the held-out week-fold rows to predict `label - ensemble` from each member's deviation from the ensemble plus row context (airport, NM estimate, push delay, ADS-B observation, time after schedule) and, in the final file, out-of-fold predictions of the tree models; it is applied to the full-fit members on the ranking rows, except where an input lies outside the range seen in training.

### 5. How the Models Combine (`taxi/prediction/build_submission.py`)

A **lane** is a complete set of ranking predictions built from one network run and one set of family weights (`build_lane`):
* value = weighted mean of the families (networks90: 90 % network, 10 % tree families; networks100: the network alone);
* regime probabilities from the two seeds with the tuned classifiers mixed in; a schedule copy is impossible outside the LOBT window; `pred` from the mixture;
* blended with the prior (prior weight 0 on NM-linked rows, 0.1 on rows without a link, 0.7 on the Rome next-day no-link rows);
* finally clipped to the window implied by the organisers' NM matching filter (|block - LOBT| <= 3606 s), with values below the window reflected into it (lower + 250 + 1.25 x the shortfall).

Rows of the submission:
* **Without an NM link:** the pass-2 hand-weighted second stage (networks90), or the plain ensembles where the departure has a second- / third-pass track (networks100). No-link departures whose LOBT can be recovered from the same flight's NM-linked arrival get rule-based raises or caps.
* **NM-linked with a third-pass track:** the pass-3 greedy-selected ensemble (networks100, reflected).
* **Other NM-linked rows:** 0.4 x the combined pass-2 / pass-3 greedy-selected networks + 0.6 x the pass-3 clip-15000 second stage, both with deeper second stages (depth 8) that also take the out-of-fold tree predictions as inputs (three on tracked and Rome rows, five on untracked rows), the push-delay Rome classifier, networks100 lanes on tracked non-Rome rows and rows with a second-pass track.
* Finally the late-day cap on the Rome next-day no-link rows (at most max(seconds after schedule, 86400 + network taxi time)).

### 6. Corrections Fitted After the Models (`out_of_fold.py`, `rome_regime_stack.py`, `aggregate_calibration.py`)

* **Rome P / S / N model** (Rome, no NM link, more than 30 min late; `--next-day-regime-model`): a multinomial P / S / N regression on the pipeline's P and S, the delay, the Rome-local date change, the date window (14.05 h), the flight's recording habit and the turnaround, fitted on every 2025 Rome no-link row more than 30 min late. In the final recipe it is applied to every such 2026 row, moving the probabilities halfway (strength 0.5) toward its own, and the late-Rome corrections (transfer factor 0.5) are applied on top. Both values were chosen on the 2025 checks below.
* **Airport-month calibration:** in 2025 the gap between the mean label and EUROCONTROL's monthly mean taxi-out is stable per airport (1-8 s between months; on arrivals it moved 5.4 s rms from 2025 to 2026), so the aggregate of a ranking month implies its mean label. The bias of the file is shrunk by the estimate's uncertainty and applied to the NM-linked rows of that airport-month, only where the linked-row and all-row estimates agree in sign (the smaller one) and not in winter months with de-icing exclusions. It reads EUROCONTROL's published monthly aggregates of the scored months themselves, information that exists only after each month.

### 7. Validation

* **Week folds:** every fourth ISO week of 2025 is held out by one fold (wk0 .. wk3); fold runs predict their held-out week and also train the second stages. Earlier rounds held out whole months (January / July 2025).
* **Corrections after the models** Some submissions were judged on the 2025 out-of-fold pipeline three ways: month-group cross-fitting, trained on January-June and tested on July-December, and the reverse. Turnaround stack, applied where the turnaround is known: -0.29 / -0.36 / -0.42 RMSE cross-fitted, -0.24 / -0.22 / -0.09 forward, -0.17 / -0.23 / -0.27 backward. Next-day regime model: -0.33 cross-fitted, +0.12 forward, -0.84 backward. Airport-month calibration: leave-one-month-out on 2025 with the month-held-out prior (it never trained on the month, though its early stopping and blend weights used it), implied vs actual monthly bias 6.4 s rms, 65 % of the predicted gain realised, 90 of 120 airport-months better.

## Results

My final RMSE on the public leaderboard was: 214.7479. Note that this was a bit worse than the best submission on the leaderboard because I was overfitting to the test data. and that went against the spirit of the competition. Also note that if you run my pipeline, you might get a very slightly different result due to random noise (for example, during GPU training), but it will be around 214.7/214.8

* Most if the loss in RMSE came from more data and more capture of that data. For example, incorporating neural networks dropped the RMSE by 7 points.
* The remaining error is in places where there is no data. For example, Istanbul's airport has no public surface data. And imputing/estimating data for this would be far worse than just leaving it blank.

## What Did Not Work

Tested on held-out data and rejected:

* Log target; capping predictions during model training; Huber loss in the networks.
* METAR features in the value model.
* Queue, arrival-interference.
* An explicit EOBT-substitution regime.
* A separate Rome value model; a dedicated no-ADS-B Rome model; training with ADS-B inputs dropped at random.
* Neural regime classifiers.
* Bigger / deeper neural networks.

## Reproducibility

You'll need to run `taxi/main.py` as detailed above. It will produce the final file (`taxi/prediction/submission.parquet`). The file submitted to the public leaderboard is just the 2 relevant months taken from this final set of predictions. However, due to inherent random processes, even with seedings, there will still be some small amount noise so don't expect a perfect match.

## How to use/modify this repository

Reading through this repository is a really big challenge. I tried to write helpful docstrings to explain each function simply. Here is how I would find my way around it.

**Where to start reading**
* `taxi/main.py` first: it lists every step in running order, holds the shared paths and helpers, and does all the downloads.
* Then `taxi/features/` in step order: `prior_tables.py` and `matrix_tables.py` (the base tables), `adsb_tracks.py` (matching ADS-B tracks to flights) and `feature_blocks.py` (the traffic and weather around each take-off).
* Then `taxi/models/` (each file trains one family of models), and last `taxi/prediction/build_submission.py`, which mixes everything into the final file.
* Fix quite a few things - most of this pipeline has extremely poor error handling. For example, if there is no METAR weather data, nothing happens. It works now, but this is just a warning to you in case the website/archive changes.

**Changing the models or the recipe**
* The hand-picked numbers are named constants at the top of each module. Most of these were recommended by AI, chosen based on cross-validation results, quick google searches (e.g. what is the average visibility distance for a pilot during clear conditions) or some of my knowledge (e.g. a decision tree shouldn't be too deep at the risk of overfitting so keep depth < 10). I would recommend toying around with most of these constants you see, because I believe there is a lot more information that can still be extracted from the data. 
* The neural network configurations are the `NETWORK_JOBS` list in `taxi/models/network_jobs.py`. The final recipe is the list of flags in `submission_step` in `taxi/main.py`.
* Run the pipeline once (the trained runs are kept by default). After that you can try other recipes by running `python -m taxi.prediction.build_submission` with different flags, without retraining anything.
* You might also be asking why it's such a large ensemble of models. Well, that one is easy to answer. It's because I needed different models to extract different bits of information from the data! That is another thing I would suggest you should toy around with - trying to simplify this pretty large ensemble because I'm sure that some models only offer a very small gain.

**Looking at other airports or months**
* You need the same kind of data the organizers of this challenge gave us: one row per departure and arrival in the challenge's format (the `_mvt` movement columns and the `_flt` NM flight-plan columns of the training files), with the label `TAXITIME_SEC_mvt` filled in for the months you train on. Arrivals matter too: they give the stand map and the arrival features.
* A new airport needs its reference point in `AIRPORT_REFERENCE_POINTS`, its weather location in `WEATHER_COORDINATES` and its X-Plane scenery id in `XPLANE_SCENERY_IDS` (all in `taxi/main.py`), and its code in `STUDY_AIRPORTS` in `taxi/features/prior_tables.py`. METAR reports are fetched by the airport's ICAO code, and the ADS-B and OPDI downloads follow the days in your files on their own.
* New months need a few fixed things changed first: the weather and METAR download windows in `taxi/main.py`, the split of training and prediction rows by year (2025 trains, 2026 is predicted, in several modules) and the prior's held-out months (`MONTH_SPLIT` in `taxi/models/prior_model.py`). EUROCONTROL's monthly averages only appear after a month ends, so `--aggregate-calibration` only works for past months.
* The late-Rome corrections (`late_rome_departures.py`, `rome_regime_stack.py`) are built around how Rome records its times and only touch Rome departures, so if another airport behaves similarly, you'll have to build similar modules.

-> From Team Charming Eel, good luck and hope you find this useful :)