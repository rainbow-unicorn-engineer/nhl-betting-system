"""
betting/montecarlo.py
Monte Carlo bankroll simulation → play out thousands of possible seasons,
each with its own random wins and losses, to see the whole range of
bankrolls a staking rule can end with, not just one lucky or unlucky
history.

What one simulated season is:
- The bets come from the backtest (betting/backtest.py's universe: the
  2025-26 games with true DraftKings two-way prices, scored by the
  walk-forward model that never trained on them). Every bet the engine
  would make there (edge of 2.5 points or more, +EV at the price) is a
  candidate, grouped by the day it was on.
- A season is that many betting days drawn at random, with replacement,
  from those days (each keeps its own bets → a "day bootstrap"), scaled to
  a full season: SEASON_GAMES games (1,312 regular season plus about 85
  playoff) instead of the ~1,000 priced ones.
- Stakes are sized from the model's CLAIMED chance (that is all the live
  system knows) with the staking rule under test, against the bankroll
  at the start of each day (profits and losses compound day to day). A
  day's bets are placed together, strongest edge first, and the daily
  cap skips what doesn't fit, as betting/recommend.py does. A bet above
  the per-bet cap is trimmed to it (betting/engine.py); one above the
  per-game cap is skipped. No day can stake more than the whole bankroll.
- Each bet then wins with its TRUE chance, which the scenario sets:
    true = market's fair chance + shrink × (model's chance − market's)
  shrink 1: the model is right (claimed edges are real);
  shrink 0.5: claimed edges are overstated by half;
  shrink 0: no real edge (the market was right; you pay its margin);
  "backtest": shrink estimated from the backtest's own wins and losses
  (maximum likelihood), with a fresh draw from its uncertainty each
  season, so the model-error doubt is part of the result.

Staking rules compared (STAKINGS): quarter-Kelly with the default caps
(2% a bet, 10% a day, 4% a game), quarter-Kelly with 25% / 100% / 50%
caps, and full Kelly with no caps.

Every staking rule and scenario uses the same random numbers (same seed,
same draw order), so differences between rows come from the rule, not
from luck.

Read-only: the walk-forward run registers nothing and writes no plot
(models.lgbm.run_lgbm(register=False, plot=False)); --oof-csv skips it.

CLI: python -m betting.montecarlo [--sims 20000] [--seed 2026]
     [--out report.md] [--oof-csv oof.csv] [--save-oof oof.csv]
"""
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from betting.engine import EDGE_MIN_ML, decimal_odds, evaluate_moneyline

logger = logging.getLogger("nhl.betting.montecarlo")

SEASON_GAMES = 1400          # 1,312 regular-season games + ~85 playoff games
DEFAULT_SIMS = 20_000
DEFAULT_SEED = 2026
RUIN_LEVEL = 0.10            # "ruin": the bankroll falls below 10% of the start
HALF_LOSS = 0.50             # "lost half": ends the season at 50% or less
EPS = 1e-12


@dataclass(frozen=True)
class Staking:
    key: str
    label: str
    kelly_mult: float              # 0.25 = quarter-Kelly, 1.0 = full Kelly
    max_bet: Optional[float]       # per-bet cap, share of bankroll; None = none
    max_day: Optional[float]       # per-day cap
    max_game: Optional[float]      # per-game cap


STAKINGS = (
    Staking("defaults", "Quarter-Kelly, caps 2% a bet / 10% a day / 4% a game "
            "(the defaults)", 0.25, 0.02, 0.10, 0.04),
    Staking("requested", "Quarter-Kelly, caps 25% a bet / 100% a day / 50% a "
            "game (requested)", 0.25, 0.25, 1.00, 0.50),
    Staking("full_kelly", "Full Kelly, no caps", 1.0, None, None, None),
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    shrink: float                  # true edge = shrink × claimed edge
    shrink_sd: float = 0.0         # season-to-season doubt about shrink


def fixed_scenarios() -> List[Scenario]:
    return [
        Scenario("claimed", "Model right: claimed edges are real", 1.0),
        Scenario("half", "Edges overstated by half", 0.5),
        Scenario("zero", "No real edge (the market is right)", 0.0),
    ]


# ── The bets ───────────────────────────────────────────────────────

def candidate_bets(oof: pd.DataFrame = None,
                   edge_min: float = EDGE_MIN_ML) -> pd.DataFrame:
    """Every bet the engine would make on the backtest's priced games, with
    its outcome: date, game_id, side, price, decimal, model_prob (pm),
    market_prob (pf, no-vig), edge, kelly (full), won. Also returns the
    number of priced games in .attrs["n_games"]. Reads the database; oof
    (game_id, prob_home) defaults to a fresh walk-forward run."""
    from betting.backtest import load_priced_games
    if oof is None:
        from models.lgbm import run_lgbm
        oof = run_lgbm(register=False, plot=False)["oof"]
    games = load_priced_games().merge(oof[["game_id", "prob_home"]],
                                      on="game_id", how="inner")
    rows = []
    for g in games.itertuples():
        d = evaluate_moneyline(g.prob_home, g.home_ml, g.away_ml, edge_min)
        if d is None:
            continue
        won = bool(g.home_won) if d.side == "HOME" else not bool(g.home_won)
        rows.append({"date": g.date, "game_id": int(g.game_id),
                     "side": d.side, "price": int(d.price),
                     "decimal": decimal_odds(d.price),
                     "pm": float(d.model_prob), "pf": float(d.market_prob),
                     "edge": float(d.edge), "kelly": float(d.kelly),
                     "won": won})
    bets = pd.DataFrame(rows)
    bets.attrs["n_games"] = len(games)
    return bets


def estimate_shrink(bets: pd.DataFrame) -> tuple:
    """(shrink, standard error): the maximum-likelihood s in
    P(win) = pf + s·(pm − pf) over the bets' real outcomes, with its
    standard error from the curvature of the likelihood. s = 1 means the
    claimed edges were real on average, 0 means no edge at all."""
    y = bets["won"].to_numpy(float)
    pf = bets["pf"].to_numpy(float)
    e = (bets["pm"] - bets["pf"]).to_numpy(float)

    def nll(s):
        p = np.clip(pf + s * e, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    grid = np.linspace(-4, 4, 8001)
    s = float(grid[np.argmin([nll(v) for v in grid])])
    h = 1e-3
    curv = (nll(s + h) - 2 * nll(s) + nll(s - h)) / h ** 2
    se = float(1 / np.sqrt(curv)) if curv > 0 else float("nan")
    return s, se


def pack_days(bets: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Bets grouped by day into padded (days, K) arrays, strongest edge
    first within each day: pm, pf, decimal, kelly, edge, won, mask."""
    days = [g.sort_values("edge", ascending=False)
            for _, g in bets.groupby("date", sort=True)]
    k = max(len(g) for g in days)
    out = {c: np.zeros((len(days), k)) for c in
           ("pm", "pf", "decimal", "kelly", "edge", "won")}
    out["mask"] = np.zeros((len(days), k), dtype=bool)
    for i, g in enumerate(days):
        n = len(g)
        for c in ("pm", "pf", "decimal", "kelly", "edge"):
            out[c][i, :n] = g[c].to_numpy(float)
        out["won"][i, :n] = g["won"].to_numpy(float)
        out["mask"][i, :n] = True
    out["decimal"][~out["mask"]] = 1.0
    return out


# ── Staking and one season ─────────────────────────────────────────

def day_stakes(kelly: np.ndarray, mask: np.ndarray, rule: Staking) -> np.ndarray:
    """Stakes as shares of the day's starting bankroll, (n, K), bets in
    each row strongest edge first. Per-bet cap trims; per-game cap skips
    (one moneyline bet per game); the daily cap skips any bet that would
    pass it, and a smaller one later can still fit; a day never stakes
    more than the whole bankroll (scaled down if it would)."""
    frac = np.where(mask, kelly * rule.kelly_mult, 0.0)
    if rule.max_bet is not None:
        frac = np.minimum(frac, rule.max_bet)
    if rule.max_game is not None:
        frac = np.where(frac > rule.max_game + EPS, 0.0, frac)
    if rule.max_day is not None:
        spent = np.zeros(frac.shape[0])
        for j in range(frac.shape[1]):
            fits = spent + frac[:, j] <= rule.max_day + EPS
            frac[:, j] = np.where(fits, frac[:, j], 0.0)
            spent += frac[:, j]
    total = frac.sum(axis=1)
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, EPS), 1.0)
    return frac * scale[:, None]


def simulate(packed: Dict[str, np.ndarray], rule: Staking, scenario: Scenario,
             n_sims: int, n_days: int, seed: int) -> Dict[str, np.ndarray]:
    """n_sims seasons of n_days bootstrapped betting days. Returns per-season
    arrays: end (bankroll, start = 1), low (lowest end-of-day bankroll),
    max_dd (largest fall from a peak, share of that peak), n_bets,
    staked (total staked, in starting bankrolls), shrink (the draw)."""
    rng = np.random.default_rng(seed)
    shrink = scenario.shrink + scenario.shrink_sd * rng.standard_normal(n_sims)
    day_idx = rng.integers(0, packed["mask"].shape[0], size=(n_sims, n_days))
    k = packed["mask"].shape[1]

    bank = np.ones(n_sims)
    peak = np.ones(n_sims)
    low = np.ones(n_sims)
    max_dd = np.zeros(n_sims)
    n_bets = np.zeros(n_sims)
    staked = np.zeros(n_sims)
    for t in range(n_days):
        d = day_idx[:, t]
        u = rng.random((n_sims, k))                 # drawn every day, every rule
        stakes = day_stakes(packed["kelly"][d], packed["mask"][d], rule)
        pm, pf = packed["pm"][d], packed["pf"][d]
        p_true = np.clip(pf + shrink[:, None] * (pm - pf), 0.001, 0.999)
        win = u < p_true
        ret = np.where(win, packed["decimal"][d] - 1.0, -1.0)
        n_bets += (stakes > 0).sum(axis=1)
        staked += stakes.sum(axis=1) * bank
        bank = bank * (1.0 + (stakes * ret).sum(axis=1))
        bank = np.maximum(bank, 0.0)
        peak = np.maximum(peak, bank)
        low = np.minimum(low, bank)
        max_dd = np.maximum(max_dd, 1.0 - bank / peak)
    return {"end": bank, "low": low, "max_dd": max_dd, "n_bets": n_bets,
            "staked": staked, "shrink": shrink}


def replay(packed: Dict[str, np.ndarray], rule: Staking) -> Dict[str, float]:
    """The real 2025-26 bets in date order with their real results, under
    this rule: end bankroll, max drawdown, bets."""
    bank, peak, max_dd, n = 1.0, 1.0, 0.0, 0
    for i in range(packed["mask"].shape[0]):
        stakes = day_stakes(packed["kelly"][i:i + 1], packed["mask"][i:i + 1],
                            rule)[0]
        ret = np.where(packed["won"][i] > 0.5, packed["decimal"][i] - 1.0, -1.0)
        n += int((stakes > 0).sum())
        bank = max(bank * (1.0 + float((stakes * ret).sum())), 0.0)
        peak = max(peak, bank)
        max_dd = max(max_dd, 1.0 - bank / peak)
    return {"end": bank, "max_dd": max_dd, "n_bets": n}


def summarize(sim: Dict[str, np.ndarray]) -> Dict[str, float]:
    end, dd = sim["end"], sim["max_dd"]
    staked = sim["staked"]
    roi = np.where(staked > 0, (end - 1.0) / np.maximum(staked, EPS), 0.0)
    return {
        "median_end": float(np.median(end)),
        "p5_end": float(np.percentile(end, 5)),
        "p95_end": float(np.percentile(end, 95)),
        "mean_end": float(np.mean(end)),
        "p_profit": float(np.mean(end > 1.0)),
        "p_lose_half": float(np.mean(end <= HALF_LOSS)),
        "p_ruin": float(np.mean(sim["low"] < RUIN_LEVEL)),
        "median_dd": float(np.median(dd)),
        "p95_dd": float(np.percentile(dd, 95)),
        "median_bets": float(np.median(sim["n_bets"])),
        "median_roi": float(np.median(roi)),
    }


def season_days(packed: Dict[str, np.ndarray], n_games: int,
                season_games: int = SEASON_GAMES) -> int:
    """Betting days in a full season: the backtest's betting days scaled
    from its priced games to season_games."""
    return max(1, int(round(packed["mask"].shape[0] * season_games
                            / max(n_games, 1))))


def run(bets: pd.DataFrame, n_sims: int = DEFAULT_SIMS,
        seed: int = DEFAULT_SEED, season_games: int = SEASON_GAMES) -> dict:
    """Every staking rule under every scenario. Returns {"table": DataFrame,
    "replay": DataFrame, "info": dict}."""
    packed = pack_days(bets)
    n_games = int(bets.attrs.get("n_games", len(bets)))
    n_days = season_days(packed, n_games, season_games)
    s_hat, s_se = estimate_shrink(bets)
    scenarios = fixed_scenarios() + [Scenario(
        "backtest", f"Backtest estimate: shrink {s_hat:.2f} ± {s_se:.2f}, "
                    f"redrawn each season", s_hat, s_se)]
    rows = []
    for sc in scenarios:
        for rule in STAKINGS:
            sim = simulate(packed, rule, sc, n_sims, n_days, seed)
            rows.append({"scenario": sc.key, "scenario_label": sc.label,
                         "staking": rule.key, "staking_label": rule.label,
                         **summarize(sim)})
            logger.info(f"{sc.key:>9} / {rule.key:<10} median end "
                        f"{rows[-1]['median_end']:.3f}, ruin "
                        f"{rows[-1]['p_ruin']:.1%}")
    rep = pd.DataFrame([{"staking": r.key, "staking_label": r.label,
                         **replay(packed, r)} for r in STAKINGS])
    info = {
        "n_bets": len(bets), "n_games": n_games,
        "n_days_backtest": int(packed["mask"].shape[0]),
        "n_days_season": n_days, "n_sims": n_sims, "seed": seed,
        "season_games": season_games,
        "first_date": str(pd.Timestamp(bets["date"].min()).date()),
        "last_date": str(pd.Timestamp(bets["date"].max()).date()),
        "mean_edge": float(bets["edge"].mean()),
        "median_edge": float(bets["edge"].median()),
        "p90_edge": float(bets["edge"].quantile(0.9)),
        "win_rate": float(bets["won"].mean()),
        "mean_pm": float(bets["pm"].mean()), "mean_pf": float(bets["pf"].mean()),
        "flat_roi": float(np.where(bets["won"], bets["decimal"] - 1, -1).mean()),
        "shrink": s_hat, "shrink_se": s_se,
        "max_quarter_kelly": float(bets["kelly"].max() * 0.25),
        "median_quarter_kelly": float(bets["kelly"].median() * 0.25),
        "share_over_2pct": float((bets["kelly"] * 0.25 > 0.02).mean()),
        "max_full_kelly": float(bets["kelly"].max()),
        "bets_per_day_max": int(packed["mask"].sum(axis=1).max()),
        "bets_per_day_mean": float(packed["mask"].sum(axis=1).mean()),
    }
    for rule in STAKINGS:
        st = day_stakes(packed["kelly"], packed["mask"], rule)
        info[f"mean_stake_{rule.key}"] = float(st[st > 0].mean())
        info[f"mean_day_{rule.key}"] = float(st.sum(axis=1).mean())
        info[f"max_day_{rule.key}"] = float(st.sum(axis=1).max())
    return {"table": pd.DataFrame(rows), "replay": rep, "info": info}


# ── Report ─────────────────────────────────────────────────────────

def _x(v: float) -> str:
    """A bankroll multiple as dollars from $1,000."""
    return f"${v * 1000:,.0f}"


def _p(v: float) -> str:
    if v == 0:
        return "0%"
    return "<0.1%" if v < 0.001 else f"{v:.1%}"


def results_table(result: dict) -> str:
    """The main table, one row per scenario and staking rule (markdown)."""
    lines = ["| True edge | Staking | Median end | 5th pct | 95th pct | "
             "Ends up | Loses 50%+ | Ruin (<10%) | Median max drawdown | "
             "95th pct drawdown |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    short = {"defaults": "¼-Kelly, 2/10/4% caps",
             "requested": "¼-Kelly, 25/100/50% caps",
             "full_kelly": "Full Kelly, no caps"}
    for r in result["table"].itertuples():
        lines.append(
            f"| {r.scenario_label} | {short.get(r.staking, r.staking)} | "
            f"{_x(r.median_end)} | {_x(r.p5_end)} | {_x(r.p95_end)} | "
            f"{_p(r.p_profit)} | {_p(r.p_lose_half)} | {_p(r.p_ruin)} | "
            f"{r.median_dd:.0%} | {r.p95_dd:.0%} |")
    return "\n".join(lines)


def replay_table(result: dict) -> str:
    lines = ["| Staking | Bets | End bankroll | Max drawdown |",
             "|---|---|---|---|"]
    for r in result["replay"].itertuples():
        lines.append(f"| {r.staking_label} | {r.n_bets} | {_x(r.end)} | "
                     f"{r.max_dd:.0%} |")
    return "\n".join(lines)


def _row(result: dict, scenario: str, staking: str) -> pd.Series:
    t = result["table"]
    return t[(t["scenario"] == scenario) & (t["staking"] == staking)].iloc[0]


def report(result: dict) -> str:
    """The whole plain-English markdown report."""
    i = result["info"]
    r = lambda sc, st: _row(result, sc, st)       # noqa: E731
    d_bt, r_bt, f_bt = (r("backtest", "defaults"), r("backtest", "requested"),
                        r("backtest", "full_kelly"))
    d_z, r_z, f_z = (r("zero", "defaults"), r("zero", "requested"),
                     r("zero", "full_kelly"))
    d_c, r_c, f_c = (r("claimed", "defaults"), r("claimed", "requested"),
                     r("claimed", "full_kelly"))
    d_h, r_h, f_h = (r("half", "defaults"), r("half", "requested"),
                     r("half", "full_kelly"))
    t = result["table"]
    q_ruin = float(t[t["staking"] != "full_kelly"]["p_ruin"].max())
    ruin_text = ("never ruined a bankroll here (no simulated season fell below "
                 "$100 in any scenario)" if q_ruin == 0 else
                 f"rarely ruin a bankroll (at most {_p(q_ruin)} of seasons "
                 f"fall below $100 in any scenario)")
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# Monte Carlo: what the stake caps do to a bankroll

*Generated {when} by `python -m betting.montecarlo` (seed {i['seed']}, {i['n_sims']:,} simulated seasons for each row). Read-only: nothing was written to the database.*

## The short answer

- **Under quarter-Kelly, the 25% / 100% / 50% caps are the same as no caps.** Quarter-Kelly never asks for more than {i['max_quarter_kelly']:.1%} of the bankroll on one bet here (half the bets are under {i['median_quarter_kelly']:.1%}), and a day's bets never add up to more than {i['max_day_requested']:.0%}, so those caps never bind. What changes is that the default 2% cap (which trims {i['share_over_2pct']:.0%} of bets) and 10% daily cap go away: the average bet rises from {i['mean_stake_defaults']:.2%} to {i['mean_stake_requested']:.2%} of the bankroll, and the average day's total from {i['mean_day_defaults']:.1%} to {i['mean_day_requested']:.1%}. The difference is modest: if the model is right the middle season ends {_x(r_c.median_end)} instead of {_x(d_c.median_end)}; if it has no edge, 1 season in 20 ends below {_x(r_z.p5_end)} instead of {_x(d_z.p5_end)}.
- **What the caps protect against is the model being wrong.** If the claimed edges are real, bigger stakes grow the bankroll faster. If they are overstated, or there is no edge, bigger stakes lose money faster and the bad seasons get much worse.
- **The backtest itself cannot tell those cases apart yet.** On its {i['n_bets']} bets the model claimed an average edge of {i['mean_edge'] * 100:.1f} points; the results fit an edge of about {i['shrink']:.2f} of that ({i['shrink'] - 2 * i['shrink_se']:.2f} to {i['shrink'] + 2 * i['shrink_se']:.2f} at 95%), which includes "no edge at all" and "fully real". Proof has to come from closing-line value (→ whether our price beat the final price before puck drop) on the live paper picks.
- **Full Kelly with no caps is the dangerous one.** Even if every claimed edge were real, its typical season has a {f_c.median_dd:.0%} fall from a peak somewhere along the way; with no real edge it ends at a median {_x(f_z.median_end)} of $1,000 and {_p(f_z.p_ruin)} of seasons are ruined.

## What was simulated

**Monte Carlo** → play out thousands of possible seasons, each with its own random wins and losses, to see the whole range of what can happen instead of one history.

- **The bets.** The backtest's {i['n_bets']} bets: every game in the {i['first_date']} to {i['last_date']} DraftKings-priced sample ({i['n_games']:,} games) where the model's chance beat the market's fair chance by at least 2.5 points (→ percentage points: 55% vs 52% is 3 points). Average claimed edge {i['mean_edge'] * 100:.1f} points (median {i['median_edge'] * 100:.1f}, 1 bet in 10 above {i['p90_edge'] * 100:.1f}). They fell on {i['n_days_backtest']} days, {i['bets_per_day_mean']:.1f} a day on average and up to {i['bets_per_day_max']}.
- **One season** is {i['n_days_season']} betting days, drawn at random from those {i['n_days_backtest']} days (each with its own bets), so the season has the real mix of edges, prices and busy days. {i['n_days_season']} is the sample's betting days scaled to a full season of about {i['season_games']:,} games.
- **Stakes** are sized from the model's claimed chance (that is all the live system knows), as a share of the bankroll at the start of each day, so wins and losses compound. A day's bets are placed together, strongest edge first, and the daily cap skips what doesn't fit.
- **Kelly** → the bet size that grows a bankroll fastest *if* the chances are right. **Quarter-Kelly** bets a quarter of that: it keeps a bit under half of full Kelly's growth with a quarter of its swings, and it is far more forgiving when the chances are wrong. **Full Kelly** bets all of it.
- **Whether each bet wins** is drawn at random using the bet's *true* chance, which each scenario sets:
  - **Model right**: true chance = the model's chance. The claimed edge is real.
  - **Edges overstated by half**: the true edge is half the claimed one.
  - **No real edge**: true chance = the market's fair chance. Every bet then loses the book's margin (→ the vig, the book's built-in fee) on average.
  - **Backtest estimate**: the share of the claimed edge that the backtest's real wins and losses support, {i['shrink']:.2f} ± {i['shrink_se']:.2f} (→ ± one standard error, the typical size of the estimate's own error). Each simulated season draws its own value from that range, so the doubt about whether the model works is built into the result.

The three staking rules use exactly the same random seasons, so differences between them come from the rule, not from luck.

## Results

Everything is shown for a $1,000 starting bankroll.

- **Median end** → the middle season: half end higher, half lower.
- **5th / 95th pct** → 1 season in 20 ends below the 5th percentile; 1 in 20 ends above the 95th.
- **Ends up** → the share of seasons that finish above $1,000.
- **Loses 50%+** → the share of seasons that finish at $500 or less.
- **Ruin** → the share of seasons where the bankroll drops below $100 (10%) at some point.
- **Max drawdown** → the biggest fall from a high point to a later low point during the season, as a share of that high point. A 40% drawdown means $1,500 fell to $900 at some stage.

{results_table(result)}

### The real 2025-26 bets, replayed

The same {i['n_bets']} bets in their real order with their real results (one history, not a simulation). The numbers differ a little from `betting/backtest.py`, which updates the bankroll after every bet instead of placing each day's bets together:

{replay_table(result)}

## What it means

1. **If the claimed edges are real**, quarter-Kelly with the default caps ends a median {_x(d_c.median_end)}, and with no caps {_x(r_c.median_end)}. Full Kelly's median is {_x(f_c.median_end)} but with a median drawdown of {f_c.median_dd:.0%} and {_p(f_c.p_lose_half)} of seasons ending at half or less.
2. **If the edges are overstated by half**, the default caps end a median {_x(d_h.median_end)}, the requested caps {_x(r_h.median_end)}, full Kelly {_x(f_h.median_end)} ({_p(f_h.p_ruin)} ruin). Full Kelly sized for edges twice the real ones is over-betting, which loses money even with a real edge.
3. **If there is no edge**, every rule loses: default caps median {_x(d_z.median_end)} (1 in 20 seasons below {_x(d_z.p5_end)}), requested caps {_x(r_z.median_end)} (1 in 20 below {_x(r_z.p5_end)}), full Kelly {_x(f_z.median_end)}.
4. **With the backtest's own uncertainty**, the default caps end a median {_x(d_bt.median_end)} with a 5th-95th percentile range of {_x(d_bt.p5_end)} to {_x(d_bt.p95_end)}; the requested caps {_x(r_bt.median_end)} ({_x(r_bt.p5_end)} to {_x(r_bt.p95_end)}); full Kelly {_x(f_bt.median_end)} ({_x(f_bt.p5_end)} to {_x(f_bt.p95_end)}, {_p(f_bt.p_ruin)} ruin).

**Decision: keep the defaults (2% / 10% / 4%) in the code for now.** The requested caps can be set any time in `.env` (`MAX_STAKE_PCT=0.25`, `MAX_DAILY_PCT=1`, `MAX_GAME_STAKE_PCT=0.5`), and the dashboard's Today tab shows which are in use. With quarter-Kelly they {ruin_text}, but they only pay off if the edge is real: with the backtest's own uncertainty the middle season ends {_x(r_bt.median_end)} with them against {_x(d_bt.median_end)} with the defaults, while the share of seasons that lose half or more goes from {_p(d_bt.p_lose_half)} to {_p(r_bt.p_lose_half)}. Raise them once the edge is shown to be real (500+ paper picks that beat the closing price). Never use full Kelly: it needs the win chances to be exactly right, and no model's are; here it ruins {_p(f_bt.p_ruin)} of seasons under the backtest's uncertainty and {_p(f_h.p_ruin)} even when half the claimed edge is real.

## Limits of this simulation

- The bets come from one season of near-closing DraftKings prices with no line shopping. The live system takes the best price across books, which adds a little edge.
- Bets are treated as independent. Moneyline bets on different games mostly are; with only one bet per game, the per-game cap never binds here.
- Stakes compound day to day. The live system sizes stakes from the fixed `BANKROLL` setting until it is changed by hand, which makes stakes grow and shrink more slowly than shown.
- The "backtest estimate" scenario rests on {i['n_bets']} bets, so its range is wide. The answer to "does the model have an edge" will come from the live closing-line value.
"""


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        prog="python -m betting.montecarlo",
        description="Monte Carlo bankroll simulation: quarter-Kelly with the "
                    "default caps, with 25%%/100%%/50%% caps, and full Kelly "
                    "with no caps, under four assumptions about the model's "
                    "real edge. Reads the database; writes nothing to it.")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                    help=f"seasons simulated per row (default {DEFAULT_SIMS:,})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"random seed (default {DEFAULT_SEED})")
    ap.add_argument("--season-games", type=int, default=SEASON_GAMES,
                    help=f"games in a season (default {SEASON_GAMES})")
    ap.add_argument("--out", default=None,
                    help="write the markdown report here (default: print it)")
    ap.add_argument("--oof-csv", default=None,
                    help="walk-forward probabilities (game_id, prob_home) "
                         "from an earlier --save-oof, to skip the model run")
    ap.add_argument("--save-oof", default=None,
                    help="save the walk-forward probabilities to this CSV")
    args = ap.parse_args(argv)
    if args.sims < 1:
        ap.error("--sims must be 1 or more")

    oof = pd.read_csv(args.oof_csv) if args.oof_csv else None
    if oof is None and args.save_oof:
        from models.lgbm import run_lgbm
        oof = run_lgbm(register=False, plot=False)["oof"]
        oof.to_csv(args.save_oof, index=False)
    bets = candidate_bets(oof)
    if bets.empty:
        logger.error("No backtest bets: are the DraftKings-priced games and "
                     "features loaded?")
        return 1
    result = run(bets, args.sims, args.seed, args.season_games)
    text = report(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Report written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
