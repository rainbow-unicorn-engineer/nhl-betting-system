"""
models/moneyline_v3.py
Moneyline model v3 experiment, priced backtests and the bet-timing study.

Terms (each explained once):
- Moneyline → a bet on who wins, overtime and shootout included.
- Log loss → a score for how wrong the probabilities were; lower is
  better. Pooled = averaged over every validation game of every fold.
- Walk-forward → train only on earlier seasons, score the next one, never
  the other way round (models.baseline.walk_forward_folds).
- Paired SE → the random wobble of a difference between two models scored
  on the same games: the standard deviation of the per-game differences
  divided by the square root of the number of games. A gap of 2 SE or
  more is unlikely to be luck.
- No-vig price → a book's probability with its built-in fee (the vig)
  taken out. Consensus no-vig close → the median over books of the
  closing no-vig home probability (features/market_prices.py).
- Market offset → the LightGBM booster starts from logit(market
  probability) and only learns a correction to it (models/lgbm.py).
- Edge → our probability minus the no-vig market probability for a side,
  in percentage points. Kelly → the bet size that grows a bankroll
  fastest if our probabilities are right; quarter-Kelly bets a quarter of
  it. Flat stake → one unit on every bet, whatever the edge.
- ROI → profit divided by the amount staked.
- CLV (closing-line value) → whether our price was better than the
  market's closing price. Closing EV of a bet → p_close × decimal − 1,
  where p_close is the consensus no-vig closing probability of the side
  we bet and decimal is the decimal odds we took: the profit per unit
  staked if the closing market is right. It is positive when the price
  beat the close.
- Game-clustered bootstrap → resample whole games with replacement many
  times and recompute the number; the middle 95% of those values is the
  confidence interval (CI).

=====================================================================
PRE-REGISTRATION (written 2026-10-04, committed before any variant ran)
=====================================================================

Data. features.game_vector as stored (seasons 2020-21..2025-26; its two
power-play columns are a constant 99 because the vectors were built
before raw.skater_games power-play time was filled). Folds:
models.baseline.walk_forward_folds (validation seasons 2021-22..2025-26).
Every variant is fitted with models.lgbm.fit_fold / predict_fold
unchanged: same LightGBM parameters, seed 42, the same time-ordered 15%
calibration tail, early stopping and temperature scaling; market-offset
booster M where a market exists, market-blind fallback F otherwise.

Variants.
  V0  lgbm v2 exactly as now. Market = raw.historical_odds normalised
      (Unibet three-way 2020-21..2023-24, DraftKings 2025-26 from
      December); 2024-25 has no market, so F scores it.
  V1  V0, plus the Odds API consensus no-vig CLOSE as the market for
      every game that has one (all of 2024-25): market_home_prob =
      nv_consensus, market_available = 1. Other seasons unchanged.
  V1p Diagnostic only, never eligible for adoption: V1 with Pinnacle's
      no-vig close where Pinnacle quoted (consensus otherwise).
  V2  V1 + 12 power-play features (features/power_play.py): for windows
      20 and 82 games (82 = season to date), home minus away of
        pp_toi_share  = sum(PP TOI) / sum(team TOI)
        pk_toi_share  = sum(PK TOI) / sum(team TOI)
        pp_gf_per60   = 3600 * sum(PP goals) / sum(PP TOI)
        pk_ga_per60   = 3600 * sum(PP goals against) / sum(PK TOI)
        pp_xgf_per60  = 3600 * sum(PP xG for) / sum(PP TOI)
        pk_xga_per60  = 3600 * sum(PP xG against) / sum(PK TOI)
      Team PP TOI = sum of its skaters' PP TOI / 5, PK TOI = sum of
      short-handed TOI / 4, team TOI = summed goalie TOI, PP goals from
      raw.team_games, PP xG = MoneyPuck shots at 5v4/5v3/4v3 (the
      features/team_features.py conventions). Rolled within team and
      season over games strictly before the game (shift, then roll).
      NaN when either side has no prior game or a zero denominator
      (LightGBM handles missing values natively).
  V3  V2 + 7 goalie-role features: home starter minus away starter of
      each features.goalie_role.ROLE_FEATURES (the actual starter, the
      same proxy the stored vectors use).

Primary metric. Pooled walk-forward log loss over every validation game
of every fold, compared with V0 by the paired SE of per-game differences.

Market test (the real bar). Priced games = 2024-25 games with an Odds
API close plus 2025-26 games with a DraftKings close. On those games:
d_mkt = mean(model log loss - market log loss), market = consensus
no-vig close (DraftKings no-vig in 2025-26), with its paired SE. Also
reported per season and against Pinnacle (2024-25).

Adoption rule (fixed in advance).
  A variant Vk (k = 1, 2, 3) is ELIGIBLE when both hold:
    (a) it beats V0 pooled by at least 2 paired SE (mean difference
        <= -2 SE), and
    (b) it is not worse than the market at 95%: d_mkt < 1.645 * SE_mkt
        (no one-sided 95% evidence that the market beats it).
  Among eligible variants: V1 is chosen; V2 replaces V1 only if V2 also
  beats V1 pooled by at least 2 paired SE; V3 replaces V2 (or V1, if V2
  was not chosen) only if it beats the chosen one by at least 2 paired
  SE. No eligible variant → V0 stays.
  Production: the chosen variant becomes models/lgbm.py's default only
  if the live pick job can rebuild its inputs. V1 needs nothing new (live
  picks already use a live consensus market). V2/V3 would also need the
  live slate builder to compute the same features; until that is built,
  a chosen V2/V3 is reported as chosen but the default stays at V1 (or
  V0) and the gap is listed as follow-up work.
  "Beats the market" (a separate claim, not needed for adoption):
  d_mkt + 1.96 * SE_mkt < 0.
  Secondary, reported only: log loss on the games where V0 has a market;
  per-season log loss; calibration (ECE); seeds 1 and 2 as a robustness
  check (the rule uses seed 42).

Priced backtests (descriptive; they change no default, and the edge
threshold stays 2.5%). For every variant, from its walk-forward
probabilities:
  2024-25  fair = consensus no-vig close; prices = (i) best of the 10
           books at the close, (ii) best of the 6 licensed US books,
           (iii) each bettor-relevant book alone (draftkings, fanduel,
           betmgm, williamhill_us, betrivers, espnbet, pinnacle).
  2025-26  fair and price = DraftKings close.
  Decision = betting.engine.evaluate_market (edge >= 2.5% and positive
  Kelly at the price). Quarter-Kelly with the locked caps (2% a bet, 10%
  a day, 4% a game): each day's candidates sorted by edge, largest
  first, staked from the bankroll at the start of the day and kept while
  they fit the caps; the bankroll compounds day by day. Flat-stake view:
  1 unit per bet. Reported: bets, hit rate, flat ROI and Kelly ROI on
  stake with 95% game-clustered bootstrap CIs (10,000 resamples, seed
  20261004), end bankroll, maximum drawdown, mean closing EV vs the
  consensus no-vig close; by edge bucket 2.5-4, 4-6, 6-9 and 9+ points.
  Exchanges: raw.odds_history holds no Kalshi or Polymarket prices (the
  history was bought with 10 sportsbooks), so an exchange-only backtest
  has no data; that is stated, not simulated. A clearly labelled
  hypothetical shows what the taker fees would do: buy at the consensus
  no-vig price plus 1 cent, Kalshi 0.07 * p * (1 - p) and Polymarket
  0.0695 * p * (1 - p) per contract.

Bet-timing study (2024-25 games with a 10:00 Central morning snapshot,
889 games). Model used: the adopted default (V0 if none is adopted, which
then uses the morning consensus as its market input in the same way).
Morning arm: the same fold models scored with the MORNING consensus
no-vig as the market input; fair = morning consensus; price = best of 10
books at the morning snapshot. Close arm: as the 2024-25 backtest,
restricted to the same games.
  H1  The model's morning edge predicts the line's move: ordinary least
      squares of move = (close consensus - morning consensus, home) on
      e = (model morning home prob - morning consensus home), all games.
      Supported if the slope > 0 with t >= 2.
  H2  Morning bets (the engine's decisions) beat the close: mean closing
      EV of the morning price > 0 by at least 2 SE (game-clustered),
      also reported as the no-vig move toward our side
      (p_close - p_morning for the side bet).
  H3  Which slot pays better: flat ROI per bet and mean closing EV,
      morning arm minus close arm on the same games, with a 95%
      game-clustered bootstrap CI. A slot is "better" only if its CI
      excludes 0. Expected to be underpowered (about 900 games).
  Caveat fixed in advance: both arms use the actual starting goalie,
  which is often unconfirmed at 10:00, so the morning arm is slightly
  flattered.

STATUS: pre-registered, not yet run.
"""
import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from betting.engine import (DEFAULT_MAX_DAILY_PCT, DEFAULT_MAX_GAME_STAKE_PCT,
                            DEFAULT_MAX_STAKE_PCT, EDGE_MIN_ML,
                            effective_decimal, evaluate_market, settle)

logger = logging.getLogger("nhl.models.moneyline_v3")

SEED = 42
ROBUST_SEEDS = (1, 2)
VARIANTS = ("V0", "V1", "V1p", "V2", "V3")
ADOPTABLE = ("V1", "V2", "V3")
PRICED_SEASONS = (20242025, 20252026)
TIMING_SEASON = 20242025
EDGE_BUCKETS = ((0.025, 0.04, "2.5-4"), (0.04, 0.06, "4-6"),
                (0.06, 0.09, "6-9"), (0.09, math.inf, "9+"))
BOOT_N = 10_000
BOOT_SEED = 20261004
START_BANKROLL = 100.0
BET_BOOKS = ("draftkings", "fanduel", "betmgm", "williamhill_us",
             "betrivers", "espnbet", "pinnacle")
EXCHANGE_SPREAD = 0.01          # hypothetical: buy 1 cent above fair
RESULTS_PATH = Path(__file__).parent / "artifacts" / "moneyline_v3_results.json"


# ── Pure: scoring and the adoption rule ─────────────────────────────

def per_game_log_loss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(diff) -> dict:
    """Mean of per-game differences, its SE (sd / sqrt(n)), z and n."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else float("nan"),
                "se": float("nan"), "z": float("nan"), "n": n}
    se = float(d.std(ddof=1) / math.sqrt(n))
    return {"mean": float(d.mean()), "se": se,
            "z": float(d.mean() / se) if se > 0 else float("nan"), "n": n}


def eligible(vs_v0: dict, vs_mkt: dict) -> bool:
    """Rule (a): beats V0 by at least 2 paired SE. Rule (b): not worse
    than the market at one-sided 95%, d_mkt < 1.645 SE_mkt."""
    return (vs_v0["mean"] <= -2.0 * vs_v0["se"]
            and vs_mkt["mean"] < 1.645 * vs_mkt["se"])


def adopt(vs_v0: Dict[str, dict], vs_mkt: Dict[str, dict],
          pairwise: Dict[tuple, dict]) -> dict:
    """The pre-registered choice. vs_v0[k] and vs_mkt[k]: paired results of
    variant k minus V0 and minus the market; pairwise[(k, j)]: k minus j.
    Walks V1, V2, V3 in order: an eligible variant replaces the incumbent
    when the incumbent is V0 (eligibility already means beating V0 by 2
    SE) or when it beats the incumbent by at least 2 paired SE."""
    elig = {k: eligible(vs_v0[k], vs_mkt[k]) for k in ADOPTABLE if k in vs_v0}
    chosen, steps = "V0", []
    for k in ADOPTABLE:
        if not elig.get(k):
            steps.append(f"{k}: not eligible")
            continue
        if chosen == "V0":
            chosen = k
            steps.append(f"{k}: eligible, replaces V0")
            continue
        d = pairwise[(k, chosen)]
        if d["mean"] <= -2.0 * d["se"]:
            steps.append(f"{k}: beats {chosen} by {d['mean']:+.5f} "
                         f"(SE {d['se']:.5f}), replaces it")
            chosen = k
        else:
            steps.append(f"{k}: eligible but does not beat {chosen} by 2 SE "
                         f"({d['mean']:+.5f}, SE {d['se']:.5f})")
    return {"eligible": elig, "chosen": chosen, "steps": steps}


def beats_market(vs_mkt: dict) -> bool:
    """The separate claim: d_mkt + 1.96 SE_mkt < 0."""
    return vs_mkt["mean"] + 1.96 * vs_mkt["se"] < 0


# ── Pure: assembling the variant matrices ───────────────────────────

def overlay_market(X: np.ndarray, names: list, game_ids,
                   probs: pd.Series) -> np.ndarray:
    """Copy of X whose market_home_prob / market_available are replaced by
    `probs` (game_id -> home probability) wherever it has a value."""
    X = X.copy()
    p = pd.Series(np.asarray(game_ids)).map(probs).to_numpy(float)
    has = ~np.isnan(p)
    X[has, names.index("market_home_prob")] = p[has]
    X[has, names.index("market_available")] = 1.0
    return X


def append_columns(X: np.ndarray, names: list, extra: pd.DataFrame) -> tuple:
    return (np.hstack([X, extra.to_numpy(float)]),
            list(names) + list(extra.columns))


def role_diffs(home_def: pd.DataFrame, away_def: pd.DataFrame) -> pd.DataFrame:
    """Home starter minus away starter of each goalie-role feature.
    features.goalie_role.defending_role_frame's home_def describes the
    AWAY starter (who faces the home attack) and away_def the HOME one."""
    from features.goalie_role import ROLE_FEATURES
    d = (away_def[ROLE_FEATURES].to_numpy(float)
         - home_def[ROLE_FEATURES].to_numpy(float))
    return pd.DataFrame(d, columns=[f"{c}_starter_diff" for c in ROLE_FEATURES])


# ── Pure: bets, staking and the bootstrap ───────────────────────────

def _none(v):
    if v is None:
        return None
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def american_from_prob(q: float) -> int:
    """The (rounded) American price of a contract costing q (0 < q < 1)."""
    dec = 1.0 / q
    return int(round((dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)))


BET_COLUMNS = ["game_id", "date", "side", "price", "book", "model_prob",
               "fair_prob", "edge", "stake_pct", "decimal", "won", "flat_pnl",
               "p_close", "close_ev"]


def make_bets(df: pd.DataFrame, prob_col: str, fair_col: str,
              home_price: str, away_price: str,
              home_book: Optional[str] = None, away_book: Optional[str] = None,
              close_fair_col: str = "nv_consensus",
              edge_min: float = EDGE_MIN_ML,
              max_stake_pct: float = DEFAULT_MAX_STAKE_PCT) -> pd.DataFrame:
    """One row per bet betting.engine.evaluate_market makes. home_book /
    away_book: a column of df holding the book, or a fixed book name (an
    exchange's taker fee is then charged), or None. Closing EV uses
    close_fair_col (the consensus no-vig close) and the fee-inclusive
    decimal odds taken; flat_pnl settles one unit."""
    rows = []
    for r in df.to_dict("records"):
        if _none(r.get(prob_col)) is None or _none(r.get(fair_col)) is None:
            continue
        hb = r.get(home_book, home_book) if home_book else None
        ab = r.get(away_book, away_book) if away_book else None
        d = evaluate_market(float(r[prob_col]), float(r[fair_col]),
                            _none(r[home_price]), _none(r[away_price]),
                            edge_min=edge_min, max_stake_pct=max_stake_pct,
                            home_book=_none(hb), away_book=_none(ab))
        if d is None:
            continue
        home = d.side == "HOME"
        pc = float(r[close_fair_col])
        p_close = pc if home else 1.0 - pc
        dec = effective_decimal(d.price, d.book)
        hw = bool(r["home_win"])
        rows.append({"game_id": r["game_id"], "date": r["date"],
                     "side": d.side, "price": d.price, "book": d.book,
                     "model_prob": d.model_prob, "fair_prob": d.market_prob,
                     "edge": d.edge, "stake_pct": d.stake_pct,
                     "decimal": dec, "won": hw if home else not hw,
                     "flat_pnl": settle(d, hw, 1.0),
                     "p_close": p_close, "close_ev": p_close * dec - 1.0})
    return pd.DataFrame(rows, columns=BET_COLUMNS)


def simulate_kelly(bets: pd.DataFrame, start: float = START_BANKROLL,
                   max_daily_pct: float = DEFAULT_MAX_DAILY_PCT,
                   max_game_pct: float = DEFAULT_MAX_GAME_STAKE_PCT) -> tuple:
    """Quarter-Kelly with the locked caps. Each day the candidates are
    sorted by edge (largest first) and staked as stake_pct (already capped
    at 2% a bet) of the bankroll at the start of the day; a bet is kept
    while the day's total stays within max_daily_pct and its game's within
    max_game_pct; the day settles at its end. Returns (kept bets with
    stake and kelly_pnl, end bankroll, maximum drawdown)."""
    if bets.empty:
        return bets.assign(stake=[], kelly_pnl=[]), start, 0.0
    bets = bets.reset_index(drop=True)
    bank, peak, max_dd, kept = start, start, 0.0, []
    for _, day in bets.sort_values("date", kind="stable").groupby("date", sort=True):
        day = day.sort_values("edge", ascending=False, kind="stable")
        spent, by_game, pnl_day = 0.0, {}, 0.0
        for idx, b in day.iterrows():
            stake = bank * b["stake_pct"]
            g = by_game.get(b["game_id"], 0.0)
            if spent + stake > bank * max_daily_pct + 1e-12:
                continue
            if g + stake > bank * max_game_pct + 1e-12:
                continue
            spent += stake
            by_game[b["game_id"]] = g + stake
            pnl = stake * (b["decimal"] - 1.0) if b["won"] else -stake
            pnl_day += pnl
            kept.append((idx, stake, pnl))
        bank += pnl_day
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
    k = bets.loc[[i for i, _, _ in kept]].copy()
    k["stake"] = [s for _, s, _ in kept]
    k["kelly_pnl"] = [p for _, _, p in kept]
    return k, bank, max_dd


def boot_idx(n: int, b: int = BOOT_N, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def ratio_ci(num, den, idx=None) -> list:
    """95% bootstrap CI of sum(num) / sum(den), resampling rows. There is
    at most one moneyline bet per game, so resampling bets is the
    game-clustered bootstrap."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 2:
        return [float("nan"), float("nan")]
    idx = boot_idx(len(num)) if idx is None else idx
    s = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def summarize_bets(bets: pd.DataFrame, kelly: Optional[pd.DataFrame] = None,
                   end_bank: Optional[float] = None,
                   max_dd: Optional[float] = None) -> dict:
    n = len(bets)
    out = {"bets": n}
    if n == 0:
        return out
    idx = boot_idx(n)
    ce = paired(bets["close_ev"])
    out.update({
        "hit_rate": float(bets["won"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "flat_roi": float(bets["flat_pnl"].mean()),
        "flat_roi_ci": ratio_ci(bets["flat_pnl"], np.ones(n), idx),
        "close_ev": ce["mean"], "close_ev_se": ce["se"],
        "beat_close_share": float((bets["close_ev"] > 0).mean()),
    })
    if kelly is not None:
        nk = len(kelly)
        out.update({
            "kelly_bets": nk,
            "kelly_staked": float(kelly["stake"].sum()) if nk else 0.0,
            "kelly_roi": (float(kelly["kelly_pnl"].sum() / kelly["stake"].sum())
                          if nk else float("nan")),
            "kelly_roi_ci": (ratio_ci(kelly["kelly_pnl"], kelly["stake"])
                             if nk > 1 else [float("nan")] * 2),
            "end_bankroll": end_bank, "max_drawdown": max_dd})
    buckets = {}
    for lo, hi, label in EDGE_BUCKETS:
        sub = bets[(bets["edge"] >= lo - 1e-12) & (bets["edge"] < hi - 1e-12)]
        if len(sub) == 0:
            buckets[label] = {"bets": 0}
            continue
        buckets[label] = {"bets": len(sub), "hit_rate": float(sub["won"].mean()),
                          "flat_roi": float(sub["flat_pnl"].mean()),
                          "flat_roi_ci": ratio_ci(sub["flat_pnl"], np.ones(len(sub))),
                          "close_ev": float(sub["close_ev"].mean())}
    out["buckets"] = buckets
    return out


def backtest(df: pd.DataFrame, prob_col: str, fair_col: str,
             home_price: str, away_price: str,
             home_book: Optional[str] = None,
             away_book: Optional[str] = None) -> dict:
    bets = make_bets(df, prob_col, fair_col, home_price, away_price,
                     home_book, away_book)
    kelly, end, dd = simulate_kelly(bets)
    r = summarize_bets(bets, kelly, end, dd)
    r["games"] = int(len(df))
    return r


def exchange_frame(df: pd.DataFrame, fair_col: str = "nv_consensus",
                   spread: float = EXCHANGE_SPREAD) -> pd.DataFrame:
    """Hypothetical exchange prices: each side's contract at its fair
    no-vig price plus `spread`, as American odds (ex_home / ex_away)."""
    d = df.copy()
    d["ex_home"] = [american_from_prob(min(p + spread, 0.99)) for p in d[fair_col]]
    d["ex_away"] = [american_from_prob(min(1 - p + spread, 0.99)) for p in d[fair_col]]
    return d


# ── Pure: the timing study ──────────────────────────────────────────

def ols_slope(x, y) -> dict:
    """Least-squares slope of y on x (with an intercept), its classical SE
    and t."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - y.mean() - slope * xc
    se = float(math.sqrt((resid ** 2).sum() / (n - 2) / (xc ** 2).sum()))
    return {"slope": slope, "se": se, "t": slope / se, "n": n}


def arm_difference(game_ids, morning: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Morning arm minus close arm on the same games: flat ROI per bet and
    mean closing EV per bet, with 95% game-clustered bootstrap CIs
    (resample the games, recompute both arms on the resample)."""
    g = pd.DataFrame({"game_id": list(game_ids)})

    def per_game(b, p):
        x = b.groupby("game_id").agg(n=("flat_pnl", "size"),
                                      pnl=("flat_pnl", "sum"),
                                      cev=("close_ev", "sum"))
        x.columns = [p + c for c in x.columns]
        return g.merge(x.reset_index(), on="game_id", how="left").fillna(0.0)

    m, c = per_game(morning, "m_"), per_game(close, "c_")
    idx = boot_idx(len(g))
    out = {}
    for stat, col in (("flat_roi", "pnl"), ("close_ev", "cev")):
        mv, mn = m[f"m_{col}"].to_numpy(), m["m_n"].to_numpy()
        cv, cn = c[f"c_{col}"].to_numpy(), c["c_n"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            point = mv.sum() / mn.sum() - cv.sum() / cn.sum()
            bs = (mv[idx].sum(1) / mn[idx].sum(1)
                  - cv[idx].sum(1) / cn[idx].sum(1))
        bs = bs[np.isfinite(bs)]
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[stat] = {"diff": float(point), "ci": ci,
                     "better": ("morning" if ci[0] > 0 else
                                "close" if ci[1] < 0 else "neither")}
    return out


# ── Database inputs (read-only) and the walk-forward runs ───────────

def load_inputs() -> dict:
    """Everything the experiment reads, once: the stored game vectors, the
    priced market (features/market_prices.py), the power-play differences
    and the goalie-role differences, all aligned to the vector rows."""
    from sqlalchemy import text

    from config.settings import engine
    from features.goalie_role import defending_role_frame, load_appearances
    from features.market_prices import load_market_prices
    from features.power_play import compute_pp_rolling, load_pp_base, pp_diffs
    from models.baseline import load_dataset

    X, y, meta, names = load_dataset()
    with engine.connect() as conn:
        market = load_market_prices(conn)
        games = pd.read_sql(text("""
            SELECT g.game_id, g.season, g.date, g.home_team, g.away_team,
                   m.home_starter_id, m.away_starter_id
            FROM raw.games g LEFT JOIN features.matchup m USING (game_id)
            WHERE g.game_id = ANY(:ids)
        """), conn, params={"ids": [int(i) for i in meta["game_id"]]})
        pp_base = load_pp_base(conn)
        app = load_appearances(conn)
        unibet = pd.read_sql(text("""
            SELECT h.game_id, g.season, g.date, h.home_ml, h.away_ml,
                   h.over_under
            FROM raw.historical_odds h JOIN raw.games g USING (game_id)
            WHERE h.provider = 'Unibet'
              AND h.home_ml IS NOT NULL AND h.away_ml IS NOT NULL
        """), conn)
    games = meta[["game_id"]].merge(games, on="game_id", how="left")
    games["date"] = pd.to_datetime(games["date"])
    pp = pp_diffs(games, compute_pp_rolling(pp_base)).reset_index(drop=True)
    home_def, away_def = defending_role_frame(games, app)
    roles = role_diffs(home_def, away_def)
    inplay = unibet.loc[unibet_inplay_mask(unibet), "game_id"].tolist()
    return {"X": X, "y": y, "meta": meta, "names": names, "market": market,
            "pp": pp, "roles": roles, "unibet_inplay_ids": inplay}


def variant_matrices(X, names, meta, market, pp, roles) -> dict:
    """(X, names) per variant, as pre-registered."""
    oa = market[market["source"] == "odds_api"].set_index("game_id")
    cons = oa["nv_consensus"]
    pin = oa["nv_pinnacle"].fillna(oa["nv_consensus"])
    ids = meta["game_id"].to_numpy()
    X1 = overlay_market(X, names, ids, cons)
    X1p = overlay_market(X, names, ids, pin)
    X2, n2 = append_columns(X1, names, pp)
    X3, n3 = append_columns(X2, n2, roles)
    return {"V0": (X, list(names)), "V1": (X1, list(names)),
            "V1p": (X1p, list(names)), "V2": (X2, n2), "V3": (X3, n3)}


@contextmanager
def lgbm_seed(seed: int):
    """models.lgbm unchanged, with only its LightGBM seed swapped."""
    import models.lgbm as L
    orig = L.LGBM_PARAMS
    L.LGBM_PARAMS = {**orig, "random_state": int(seed)}
    try:
        yield
    finally:
        L.LGBM_PARAMS = orig


def walk_forward(X, y, meta, names, seed: int = SEED,
                 keep_season: Optional[int] = None,
                 fold_transform=None) -> tuple:
    """Out-of-fold P(home win) for every validation game with
    models.lgbm.fit_fold / predict_fold, and the fitted fold model of
    keep_season (for re-scoring it with another market input).
    fold_transform(X, train_idx) -> X: an optional per-fold change of the
    inputs fitted on that fold's training rows only (the post-hoc
    diagnostics use it)."""
    from models.baseline import walk_forward_folds
    from models.lgbm import fit_fold, market_offset, predict_fold

    oof = np.full(len(y), np.nan)
    kept = None
    with lgbm_seed(seed):
        for f in walk_forward_folds(meta):
            Xf = X if fold_transform is None else fold_transform(X, f.train_idx)
            base = market_offset(Xf, names)
            avail = Xf[:, names.index("market_available")] == 1.0
            fm = fit_fold(Xf, y, base, f.train_idx, meta["date"], names)
            oof[f.val_idx] = predict_fold(fm, Xf[f.val_idx], base[f.val_idx],
                                          avail[f.val_idx])
            if f.val_season == keep_season:
                kept = fm
    return oof, kept


# ── Post-hoc diagnostics (added AFTER the pre-registered run; never
#    adoptable, see STATUS) ──────────────────────────────────────────

UNIBET_LAST_SEASON = 20232024
UNIBET_MAX_ABS_ML = 1000          # |moneyline| at or above: captured in play
UNIBET_MIN_IMPLIED_SUM = 0.75     # both sides long: a tied game in play
UNIBET_TOTAL_RANGE = (5.0, 7.0)   # pre-game NHL totals; NULL is kept
UNIBET_INPLAY_FROM = {20232024: pd.Timestamp("2024-04-08")}


def unibet_inplay_mask(df: pd.DataFrame) -> pd.Series:
    """True for an ESPN Unibet row captured during the game, or inside the
    late-2023-24 stretch where about half the rows were. df: season, date,
    home_ml, away_ml, over_under."""
    from features.util import american_implied_prob
    h, a = df["home_ml"].astype(float), df["away_ml"].astype(float)
    big = (h.abs() >= UNIBET_MAX_ABS_ML) | (a.abs() >= UNIBET_MAX_ABS_ML)
    s = h.map(american_implied_prob) + a.map(american_implied_prob)
    ou = df["over_under"].astype(float)
    lo, hi = UNIBET_TOTAL_RANGE
    bad_total = ou.notna() & ((ou < lo) | (ou > hi))
    window = pd.Series(False, index=df.index)
    d = pd.to_datetime(df["date"])
    for season, start in UNIBET_INPLAY_FROM.items():
        window |= (df["season"] == season) & (d >= start)
    return big | (s < UNIBET_MIN_IMPLIED_SUM) | bad_total | window


def drop_market(X: np.ndarray, names: list, game_ids, drop_ids) -> np.ndarray:
    """Copy of X with the listed games set to 'no market' (0.5, 0)."""
    X = X.copy()
    rows = pd.Series(np.asarray(game_ids)).isin(set(drop_ids)).to_numpy()
    X[rows, names.index("market_home_prob")] = 0.5
    X[rows, names.index("market_available")] = 0.0
    return X


def unibet_mapper(y: np.ndarray, names: list, unibet_rows: np.ndarray,
                  min_rows: int = 200):
    """fold_transform for walk_forward: a logistic fit
    y ~ a * logit(p_unibet) + b on the fold's TRAINING Unibet-era games
    maps every Unibet-era market probability onto the two-way scale
    (train and validation rows alike); two-way sources are untouched."""
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression

    from models.lgbm import PROB_CLIP
    i = names.index("market_home_prob")

    def transform(X, train_idx):
        tr = train_idx[unibet_rows[train_idx]]
        if len(tr) < min_rows:
            return X
        z = logit(np.clip(X[tr, i], *PROB_CLIP))
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y[tr])
        a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        X = X.copy()
        X[unibet_rows, i] = expit(a * logit(np.clip(X[unibet_rows, i], *PROB_CLIP)) + b)
        return X
    return transform


def diagnostic_runs(X1, names, y, meta, unibet_inplay_ids) -> dict:
    """V1m (V1 + the per-fold Unibet two-way mapping) and V1mc (V1m with
    the in-play Unibet rows treated as having no market): (X, names,
    fold_transform) per diagnostic."""
    season = meta["season"].to_numpy()
    ids = meta["game_id"].to_numpy()
    avail = X1[:, names.index("market_available")] == 1.0
    uni = avail & (season <= UNIBET_LAST_SEASON)
    X1c = drop_market(X1, names, ids, unibet_inplay_ids)
    uni_c = (X1c[:, names.index("market_available")] == 1.0) & (season <= UNIBET_LAST_SEASON)
    return {"V1m": (X1, names, unibet_mapper(y, names, uni)),
            "V1mc": (X1c, names, unibet_mapper(y, names, uni_c))}


def rescore(fm: dict, X, names, rows: np.ndarray, game_ids,
            probs: pd.Series) -> np.ndarray:
    """A fold model's P(home win) for `rows` with `probs` as the market."""
    from models.lgbm import market_offset, predict_fold
    Xr = overlay_market(X[rows], names, np.asarray(game_ids)[rows], probs)
    base = market_offset(Xr, names)
    avail = Xr[:, names.index("market_available")] == 1.0
    return predict_fold(fm, Xr, base, avail)


# ── The experiment ──────────────────────────────────────────────────

def score_variants(y, meta, oofs: dict, market: pd.DataFrame,
                   v0_avail: np.ndarray) -> dict:
    """Pooled and per-season log loss, ECE, the paired comparisons, the
    market test and the adoption decision."""
    from models.baseline import expected_calibration_error

    scored = ~np.isnan(oofs["V0"])
    ll = {k: per_game_log_loss(y, np.where(scored, v, 0.5)) for k, v in oofs.items()}
    for k in ll:
        ll[k][~scored] = np.nan
    mk = meta[["game_id", "season"]].merge(
        market[["game_id", "nv_consensus", "nv_pinnacle"]], on="game_id", how="left")
    priced = scored & mk["nv_consensus"].notna().to_numpy()
    mkt_ll = np.where(priced, per_game_log_loss(y, mk["nv_consensus"].fillna(0.5)), np.nan)
    has_pin = priced & mk["nv_pinnacle"].notna().to_numpy()
    pin_ll = np.where(has_pin, per_game_log_loss(y, mk["nv_pinnacle"].fillna(0.5)), np.nan)
    season = meta["season"].to_numpy()
    seasons = sorted(set(season[scored]))

    out = {"n_scored": int(scored.sum()), "n_priced": int(priced.sum()),
           "market_log_loss": float(np.nanmean(mkt_ll)),
           "market_log_loss_by_season": {
               str(s): float(np.nanmean(mkt_ll[priced & (season == s)]))
               for s in PRICED_SEASONS if (priced & (season == s)).any()},
           "pinnacle_log_loss": float(np.nanmean(pin_ll)),
           "variants": {}}
    vs_v0, vs_mkt = {}, {}
    for k in oofs:
        v = {"log_loss": float(np.nanmean(ll[k])),
             "ece": expected_calibration_error(y[scored], oofs[k][scored]),
             "by_season": {str(s): float(np.nanmean(ll[k][season == s]))
                           for s in seasons},
             "log_loss_v0_market_games": float(np.nanmean(ll[k][scored & v0_avail])),
             "vs_v0": paired((ll[k] - ll["V0"])[scored]),
             "vs_market": paired((ll[k] - mkt_ll)[priced]),
             "vs_market_by_season": {
                 str(s): paired((ll[k] - mkt_ll)[priced & (season == s)])
                 for s in PRICED_SEASONS if (priced & (season == s)).any()},
             "vs_pinnacle": paired((ll[k] - pin_ll)[has_pin]),
             "log_loss_priced": float(np.nanmean(ll[k][priced]))}
        v["beats_market"] = beats_market(v["vs_market"])
        out["variants"][k] = v
        vs_v0[k], vs_mkt[k] = v["vs_v0"], v["vs_market"]
    pairwise = {(a, b): paired((ll[a] - ll[b])[scored])
                for a in ADOPTABLE for b in ADOPTABLE if a != b}
    out["pairwise"] = {f"{a}-{b}": d for (a, b), d in pairwise.items()}
    out["decision"] = adopt(vs_v0, vs_mkt, pairwise)
    return out


def priced_frame(meta, market, prob: np.ndarray) -> pd.DataFrame:
    p = meta[["game_id"]].assign(prob_home=prob)
    df = market.merge(p, on="game_id", how="inner")
    return df[df["prob_home"].notna()].reset_index(drop=True)


def run_backtests(df: pd.DataFrame) -> dict:
    """The pre-registered price views for one variant's probabilities."""
    out = {}
    s24 = df[df["season"] == 20242025]
    s25 = df[df["season"] == 20252026]
    out["2024-25 best of 10 books"] = backtest(
        s24, "prob_home", "nv_consensus", "best_home_price", "best_away_price",
        "best_home_book", "best_away_book")
    out["2024-25 best of 6 US books"] = backtest(
        s24, "prob_home", "nv_consensus", "best_us_home_price",
        "best_us_away_price", "best_us_home_book", "best_us_away_book")
    for b in BET_BOOKS:
        out[f"2024-25 {b} only"] = backtest(
            s24, "prob_home", "nv_consensus", f"{b}_home", f"{b}_away", b, b)
    out["2025-26 DraftKings"] = backtest(
        s25, "prob_home", "nv_consensus", "best_home_price", "best_away_price",
        "best_home_book", "best_away_book")
    for label, part in (("2024-25", s24), ("2025-26", s25)):
        ex = exchange_frame(part)
        out[f"{label} HYPOTHETICAL exchange, no fee"] = backtest(
            ex, "prob_home", "nv_consensus", "ex_home", "ex_away")
        for venue in ("kalshi", "polymarket"):
            out[f"{label} HYPOTHETICAL {venue} with taker fee"] = backtest(
                ex, "prob_home", "nv_consensus", "ex_home", "ex_away", venue, venue)
    return out


def timing_study(fm: dict, X, names, meta, market, oof_close: np.ndarray) -> dict:
    """The pre-registered H1-H3 on 2024-25 games with a morning snapshot."""
    ids = meta["game_id"].to_numpy()
    m = market[(market["season"] == TIMING_SEASON)
               & market["m_nv_consensus"].notna()].set_index("game_id")
    rows = np.flatnonzero(pd.Series(ids).isin(m.index).to_numpy())
    p_morning = rescore(fm, X, names, rows, ids, m["m_nv_consensus"])
    p_close = rescore(fm, X, names, rows, ids, m["nv_consensus"])
    df = m.loc[ids[rows]].reset_index()
    df["p_morning"], df["p_close"] = p_morning, p_close
    df["oof"] = oof_close[rows]

    e = df["p_morning"] - df["m_nv_consensus"]
    move = df["nv_consensus"] - df["m_nv_consensus"]
    h1 = ols_slope(e, move)
    h1["supported"] = bool(h1["slope"] > 0 and h1["t"] >= 2)

    mb = make_bets(df, "p_morning", "m_nv_consensus", "m_best_home_price",
                   "m_best_away_price", "m_best_home_book", "m_best_away_book")
    cb = make_bets(df, "p_close", "nv_consensus", "best_home_price",
                   "best_away_price", "best_home_book", "best_away_book")
    ce = paired(mb["close_ev"])
    mv = paired(mb["p_close"] - mb["fair_prob"])
    h2 = {"close_ev": ce, "novig_move_to_side": mv,
          "supported": bool(ce["n"] > 1 and ce["mean"] >= 2 * ce["se"])}
    mk, mend, mdd = simulate_kelly(mb)
    ck, cend, cdd = simulate_kelly(cb)
    return {"games": int(len(df)),
            "close_model_matches_oof": float(np.nanmax(np.abs(df["p_close"] - df["oof"]))),
            "H1": h1, "H2": h2,
            "H3": arm_difference(df["game_id"], mb, cb),
            "morning_arm": summarize_bets(mb, mk, mend, mdd),
            "close_arm": summarize_bets(cb, ck, cend, cdd)}


def run_all(seeds: Iterable[int] = ROBUST_SEEDS, save: bool = True) -> dict:
    """The whole pre-registered run. Read-only: writes nothing to the
    database, only the results JSON in models/artifacts."""
    inp = load_inputs()
    X, y, meta, names, market = (inp["X"], inp["y"], inp["meta"],
                                 inp["names"], inp["market"])
    mats = variant_matrices(X, names, meta, market, inp["pp"], inp["roles"])
    oofs, folds = {}, {}
    for k, (Xk, nk) in mats.items():
        logger.info(f"{k}: {Xk.shape[1]} columns, seed {SEED}")
        oofs[k], folds[k] = walk_forward(Xk, y, meta, nk, SEED, TIMING_SEASON)
    v0_avail = X[:, names.index("market_available")] == 1.0
    res = {"seed": SEED, "scores": score_variants(y, meta, oofs, market, v0_avail)}

    # Post-hoc diagnostics: scored next to the variants, never adoptable
    # (score_variants' decision only looks at ADOPTABLE).
    diag = diagnostic_runs(mats["V1"][0], mats["V1"][1], y, meta,
                           inp["unibet_inplay_ids"])
    res["n_unibet_inplay_games"] = len(inp["unibet_inplay_ids"])
    doofs = {}
    for k, (Xk, nk, tf) in diag.items():
        logger.info(f"{k} (post-hoc diagnostic), seed {SEED}")
        doofs[k], _ = walk_forward(Xk, y, meta, nk, SEED, fold_transform=tf)
    dsc = score_variants(y, meta, {**oofs, **doofs}, market, v0_avail)
    res["diagnostics"] = {k: dsc["variants"][k] for k in doofs}
    res["diagnostics_pairwise_vs_V1"] = {
        k: paired((per_game_log_loss(y, np.nan_to_num(doofs[k], nan=0.5))
                   - per_game_log_loss(y, np.nan_to_num(oofs["V1"], nan=0.5)))
                  [~np.isnan(oofs["V0"])]) for k in doofs}

    res["robustness"] = {}
    for s in seeds:
        so = {}
        for k, (Xk, nk) in mats.items():
            logger.info(f"{k}: seed {s}")
            so[k], _ = walk_forward(Xk, y, meta, nk, s)
        for k, (Xk, nk, tf) in diag.items():
            so[k], _ = walk_forward(Xk, y, meta, nk, s, fold_transform=tf)
        sc = score_variants(y, meta, so, market, v0_avail)
        res["robustness"][str(s)] = {
            k: {"log_loss": v["log_loss"], "vs_v0": v["vs_v0"],
                "vs_market": v["vs_market"]} for k, v in sc["variants"].items()}
        res["robustness"][str(s)]["decision"] = sc["decision"]

    res["backtests"] = {k: run_backtests(priced_frame(meta, market, o))
                        for k, o in {**oofs, **doofs}.items()}
    chosen = res["scores"]["decision"]["chosen"]
    Xc, nc = mats[chosen]
    res["timing"] = {"model": chosen, **timing_study(
        folds[chosen], Xc, nc, meta, market, oofs[chosen])}
    if save:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(res, indent=1, default=_json))
        logger.info(f"Results saved to {RESULTS_PATH}")
    return res


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def report(res: dict) -> str:
    """Plain-text summary of a run_all result."""
    sc = res["scores"]
    lines = [f"Scored {sc['n_scored']} games; priced {sc['n_priced']}; "
             f"market log loss {sc['market_log_loss']:.4f} "
             f"{sc['market_log_loss_by_season']}"]
    for k, v in sc["variants"].items():
        a, b = v["vs_v0"], v["vs_market"]
        lines.append(
            f"{k:4s} LL {v['log_loss']:.5f} ECE {v['ece']:.4f} | vs V0 "
            f"{a['mean']:+.5f} (SE {a['se']:.5f}) | vs market {b['mean']:+.5f} "
            f"(SE {b['se']:.5f}, n {b['n']}) | priced LL {v['log_loss_priced']:.5f}")
    lines.append(f"Decision: {sc['decision']['chosen']} — "
                 + "; ".join(sc["decision"]["steps"]))
    for k, v in res.get("diagnostics", {}).items():
        a, b = v["vs_v0"], v["vs_market"]
        lines.append(
            f"{k:4s} (post-hoc) LL {v['log_loss']:.5f} ECE {v['ece']:.4f} | vs V0 "
            f"{a['mean']:+.5f} (SE {a['se']:.5f}) | vs market {b['mean']:+.5f} "
            f"(SE {b['se']:.5f}) | by season {v['by_season']} | mkt by season "
            + str({s: round(d['mean'], 4) for s, d in v['vs_market_by_season'].items()}))
    for s, r in res.get("robustness", {}).items():
        lines.append(f"seed {s}: " + ", ".join(
            f"{k} {v['vs_v0']['mean']:+.5f}/{v['vs_v0']['se']:.5f}"
            for k, v in r.items() if k != "decision")
            + f" -> {r['decision']['chosen']}")
    for k, views in res["backtests"].items():
        for view, r in views.items():
            if r.get("bets"):
                lines.append(
                    f"{k:4s} {view:45s} bets {r['bets']:4d} hit {r['hit_rate']:.3f} "
                    f"flat {r['flat_roi']:+.3%} {['%+.1f%%' % (100 * c) for c in r['flat_roi_ci']]} "
                    f"kelly {r['kelly_roi']:+.3%} end {r['end_bankroll']:.1f} "
                    f"closeEV {r['close_ev']:+.4f}")
            else:
                lines.append(f"{k:4s} {view:45s} no bets")
    t = res["timing"]
    lines.append(f"Timing ({t['model']}, {t['games']} games): H1 slope "
                 f"{t['H1']['slope']:.3f} t {t['H1']['t']:.2f} "
                 f"supported={t['H1']['supported']}; H2 closeEV "
                 f"{t['H2']['close_ev']['mean']:+.4f} (SE {t['H2']['close_ev']['se']:.4f}) "
                 f"supported={t['H2']['supported']}; H3 {t['H3']}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(report(run_all()))
