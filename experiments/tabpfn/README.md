# TabPFN trial for the moneyline model

**Question:** can TabPFN, a pretrained "foundation model for tables" from Prior Labs (published on Hugging Face), predict NHL home wins better than our LightGBM model (`lgbm_market v2`) and better than the betting market?

**Answer (2026-10-04):** No. No variant passes the pre-registered rule, so `lgbm_market v2` stays the production model and nothing else changes. TabPFN used as a plain classifier (A) comes within 0.002 log loss of lgbm v2 with no tuning, but it is not better; the average of the two (C) ties lgbm v2; learning the gap to the market (B) is clearly worse. None of them, and not lgbm v2 either, beats the no-vig market price.

The pre-registration (written and committed before any run), every variant's numbers and the decision are in the `run.py` docstring. The full numbers are in `results.json`.

## Plain-English terms

- **TabPFN** → a neural network that was pretrained on millions of made-up tables. It does not train on our games in the usual way: it reads the past games as examples and predicts new games in one pass (in-context learning → learning from examples shown at prediction time, with no retraining). It is built for tables up to about 10,000 rows. Our largest training window is about 6,550 games, so it fits.
- **Moneyline** → a bet on who wins the game, overtime and shootout included.
- **Log loss** → a score for how wrong the probabilities were. Lower is better.
- **Paired SE** → the size of the random wobble in a difference between two models, measured on the same games. A gap of 2 SE or more is unlikely to be luck.
- **No-vig market price** → the sportsbook's probability with its built-in fee (the vig) taken out.
- **Walk-forward** → train only on earlier seasons and test on the next season, never the other way round.
- **Temperature scaling** → one number that squeezes or stretches a model's confidence so that "60%" happens about 60% of the time.

## What was tested

Same data, folds and features as `models/lgbm.py` v2: the 109 features in `features.game_vector`, labeled games 2020-21 to 2025-26, five walk-forward folds (validation seasons 2021-22 to 2025-26). Like lgbm v2, there are two models per fold: one for games with a betting line (trained on lined games only) and a market-blind fallback for games without one (2024-25 has no line in the stored features).

| Variant | What it is |
|---|---|
| A, feature | TabPFN classifier predicts the winner from all 109 features, with the market probability as one of them |
| B, residual | TabPFN regressor predicts how far the result lands from the market (result minus no-vig market probability); prediction = market + that correction |
| C, ensemble | Plain average of lgbm v2's and variant A's probabilities |
| A seed 1, seed 2 | Variant A with a different random seed. Report-only, to show how much the seed alone moves things |

**Pass rule (fixed before the run):** a variant replaces lgbm v2 only if it beats lgbm v2 by at least 2 paired SE in at least 4 of the 5 seasons, and its log loss on priced games is no worse than the no-vig market's.

## Results

Run once, 2026-10-04, on the RTX 5080 (about 8 minutes for all TabPFN fits). 6,993 scored games over 5 seasons, 5,061 of them with a betting line ("priced"). Log loss: lower is better. "vs lgbm v2" is the mean per-game difference (variant minus lgbm v2) with its paired SE; negative would mean the variant is better.

| Model | Pooled log loss | vs lgbm v2 | vs no-vig market (priced games) | ECE |
|---|---|---|---|---|
| lgbm v2 (production) | 0.6616 | - | +0.0041 ± 0.0012 | 0.0168 |
| No-vig market (priced games only) | 0.6523 | - | - | 0.0172 |
| A, feature | 0.6635 | +0.0019 ± 0.0010 | +0.0087 ± 0.0017 | 0.0168 |
| B, residual | 0.6777 | +0.0160 ± 0.0023 | +0.0282 ± 0.0033 | 0.0255 |
| C, ensemble | 0.6617 | +0.0001 ± 0.0005 | +0.0057 ± 0.0014 | 0.0169 |
| A seed 1 (report-only) | 0.6642 | +0.0026 ± 0.0010 | +0.0095 ± 0.0017 | 0.0172 |
| A seed 2 (report-only) | 0.6639 | +0.0023 ± 0.0010 | +0.0092 ± 0.0017 | 0.0156 |

Per season, variant minus lgbm v2 (± paired SE). To count toward the rule a season needs a value at or below −2 SE; none gets there.

| Season | A | B | C |
|---|---|---|---|
| 2021-22 | +0.0108 ± 0.0020 | +0.0260 ± 0.0037 | +0.0047 ± 0.0010 |
| 2022-23 | −0.0010 ± 0.0020 | +0.0167 ± 0.0071 | −0.0012 ± 0.0010 |
| 2023-24 | +0.0021 ± 0.0021 | +0.0237 ± 0.0058 | +0.0003 ± 0.0012 |
| 2024-25 | −0.0027 ± 0.0030 | −0.0027 ± 0.0030 | −0.0029 ± 0.0015 |
| 2025-26 | +0.0003 ± 0.0022 | +0.0165 ± 0.0052 | −0.0007 ± 0.0011 |

2024-25 has no line in the stored features, so A and B are the same market-blind model that season.

**Decision:** A, B and C each beat lgbm v2 by 2 SE in 0 of 5 seasons (the rule needs 4), and each is worse than the no-vig market on priced games (A by 5.0 SE, B by 8.5 SE, C by 4.0 SE). Both parts of the rule fail for all three.

**Seed check (report-only):** changing only the random seed moves A's pooled log loss by up to 0.0007, about a third of A's gap to lgbm v2, so A's deficit is not just seed noise.

**Outside bar for 2024-25 (report-only):** the Pinnacle no-vig closing price (Pinnacle → a low-margin "sharp" sportsbook whose closing line is the usual benchmark; from `raw.odds_history`, 1,398 games) scores 0.6574. lgbm v2 is +0.0063 ± 0.0040 worse, A +0.0036 ± 0.0034, C +0.0034 ± 0.0034. No model beats the sharp close.

**What it means:** TabPFN-2 is a capable general learner but adds nothing that the market and lgbm v2 do not already have. The gap between the result and the market price is mostly noise, and a model that learns it directly (B) overreacts to it. The market stays the bar to beat.

## Re-running

```powershell
# one-time, GPU machine (RTX 5080 needs a CUDA 12.8+ build of torch)
.venv\Scripts\python -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install tabpfn==9.1.0

# the trial (reads the database, writes nothing to it, registers no model)
.venv\Scripts\python -m experiments.tabpfn.run            # about 10 minutes on the RTX 5080
.venv\Scripts\python -m experiments.tabpfn.run --no-seeds # skip the report-only seed re-runs
```

The first run downloads the TabPFN-2 weights from Hugging Face into `%APPDATA%\tabpfn`: `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (29 MB, the package's V2 default) and `tabpfn-v2-regressor.ckpt` (44 MB). No account or token is needed for these weights. Out-of-fold predictions go to `data/experiments/tabpfn_oof.csv` (git-ignored); the summary goes to `experiments/tabpfn/results.json`.

Installed for this trial on 2026-10-04: `tabpfn 9.1.0` and `torch 2.14.1+cu130`, plus their new dependencies (`einops`, `huggingface_hub`, `pydantic`, `safetensors`, `skrub` and a few small ones). The install only added packages: no existing package changed version, and the full test suite still passes. The CPU works too, but slowly. Machines without TabPFN skip its one smoke test.

## Licence

- **The `tabpfn` Python package:** Apache License 2.0.
- **The weights this trial uses (TabPFN-2, `Prior-Labs/TabPFN-v2-clf` and `-reg`):** "Prior Labs License" version 1.1 (May 2025). This is Apache 2.0 with one added paragraph (10, Additional attribution). It allows commercial use. If we *distribute or make available* the weights or a product or service containing them, we must ship a copy of the licence and prominently show "Built with PriorLabs-TabPFN" on the related website, user interface or documentation. A model trained or improved using TabPFN's outputs and made available to others must have a name starting with "TabPFN". The licence says internal benchmarking and testing without external communication need no attribution, and this trial is that.
- **If TabPFN ever went into production:** the dashboard would need the "Built with PriorLabs-TabPFN" notice and a copy of the licence, and the model's name would have to start with "TabPFN".
- **The newer weights (TabPFN-2.5, 2.6, 3 and 3.5, the package default since v6):** these are under *non-commercial* licences and gated. Downloading them needs a Prior Labs login, plus a `TABPFN_TOKEN` on a machine with no browser. A betting engine run to make money is not clearly non-commercial use, and a headless run cannot accept the licence, so they were left out on purpose. Trying them would need the owner to read and accept those terms first, and it would be a new, separately pre-registered experiment.
