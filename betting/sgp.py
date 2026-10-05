"""
betting/sgp.py
Same-game parlay (SGP) joint pricer.

Terms (each explained once):
- Same-game parlay (SGP) → one slip with several bets ("legs") on the SAME
  game, e.g. "Toronto to win AND over 6.5 goals". Every leg must win.
- Joint probability → the chance that two things happen together. For
  legs in different games it is just the product of their chances
  (independence → one result tells you nothing about the other). For
  legs in one game it is not: if the favourite wins, the game was more
  likely high-scoring, so "favourite + over" happens more often than the
  product says.
- Score grid → a table of probabilities for every final score (0-0, 1-0,
  0-1, ...). Any bet on the game (who wins, the total, one team's goals,
  the puck line) is a set of cells in the grid, so any combination of
  bets is priced by adding up the cells where every leg wins.
- No-vig price → the sportsbook's probability with its built-in fee (vig)
  taken out.
- Log loss → a score for how surprised a set of probabilities was by what
  actually happened. Lower is better.
- Tilt → a smooth reweighting of the grid that moves one number (e.g. the
  home team's win chance) to a target while keeping another (e.g. the
  distribution of the total) exactly as it was.
- Regulation → the first 60 minutes. OT → overtime; SO → shootout. A game
  tied after regulation goes to OT/SO and the winner is credited with one
  extra goal, which counts toward the total for settlement.

How the grid is built (joint_grid):
1. Start from the totals model's per-side regulation goal distributions
   (models/totals.py: one Poisson per side, then each score cell
   reweighted by its regulation winning margin, MARGIN_WEIGHTS, because
   hockey has more regulation ties and fewer one-goal games than two
   independent Poissons say).
2. Optionally tilt the TOTAL so P(over the line) matches a target (the
   no-vig market): every cell is multiplied by exp(phi * settlement total)
   and the grid renormalized.
3. Tilt the MARGIN so P(home wins) matches the win model (or the no-vig
   market) while keeping the settlement-total distribution exactly:
   within every settlement total t, each cell is multiplied by
   exp(theta * (home goals - away goals)) and the cells of that total
   rescaled to keep P(total = t). For two Poissons this is exactly
   "shift the share of the goals toward one side, same number of goals".
4. Regulation ties go to OT/SO: home wins it with q_ot =
   0.5 + OT_BETA * (P(home wins) - 0.5), so overtime is close to a coin
   flip with a lean toward the stronger team. The OT winner gets +1 goal,
   so a 3-3 regulation tie settles as 4-3 (total 7).
Every leg is then read from the final-score grid (final_scores,
leg_masks, price_legs).

PRE-REGISTRATION (written and committed 2026-10-04, before any variant
was run on real outcomes)
------------------------------------------------------------------------
Question: does the joint pricer price "moneyline side x over/under" in
one game better than the independence product of the same two
probabilities?

Data (closing prices only; one row per game):
- 2024-25: raw.odds_history, each book's last snapshot strictly before
  puck drop (raw.games.start_time_utc). Moneyline: each book's no-vig
  home probability (proportional), median across books. Over/under: the
  half-point line quoted by the most books (tie: the one whose median
  no-vig P(over) is nearest 50%), median no-vig P(over) across the books
  quoting it. Games with no half-point line are dropped.
- 2025-26: raw.historical_odds, provider = 'DraftKings', closing
  moneyline (home_ml/away_ml) and closing over/under line + prices
  (over_under, over_price, under_price), proportional no-vig. Rows with
  |moneyline| >= 1000, missing prices or an integer line are dropped.
- Regular season and playoffs; games with a final score only.
- 2023-24 and earlier (Unibet) are NOT used: their moneyline is a 3-way
  regulation line, not a two-way price.
Base grids: the totals model v2 scored out-of-fold → (each game predicted
by a model trained only on earlier seasons) from the walk-forward folds
for 2024-25 and 2025-26 (models.baseline.walk_forward_folds, the same
fold loop as models.totals.run_totals: booster, drift correction, margin
weights fitted on the fold's training games). OT_BETA per validation
season: maximum likelihood on the fold's training games that went to
OT/SO, with the pre-game Elo home expectation (features.matchup
home_elo/away_elo, +50 home ice) as P(home wins).
Outcome per game, 4 cells: (favourite wins, over), (favourite, under),
(underdog, over), (underdog, under); favourite = the side the no-vig
market makes > 50% (home on exactly 50%); "wins" includes OT/SO; over =
final total incl. the shootout goal > line.
Score: 4-way log loss per game, joint minus independence; mean over the
pooled 2024-25 + 2025-26 games, standard error clustered by game (each
game is one cluster). Independence = P(side) x P(over) using the SAME
two marginals as the joint (the joint matches them exactly), so the
comparison isolates the dependence between the legs.
Variants (every one is reported):
  A  totals-model grid; total tilted to the market's no-vig P(over);
     margin tilted to the market's no-vig P(home win); fitted OT_BETA.
     Independence: market P(side) x market P(over).
  B  totals-model grid, its OWN total distribution kept (no total tilt);
     margin tilted to the market P(home win); fitted OT_BETA.
     Independence: market P(side) x model P(over). This is the form the
     bet checker would use (with the win model in place of the market).
  C  as A, but the base grid is the environment baseline (trailing
     league scoring rates only, same margin weights).
  D  as A, but without the margin reweighting (two independent Poissons).
  E  as A, with OT_BETA = 0 (overtime a pure coin flip).
  F  as B, but the margin tilted to the win model's out-of-fold P(home
     win) (models.lgbm.run_lgbm, register=False, plot=False);
     independence: model P(side) x model P(over).
Pass rule (fixed in advance): a variant passes when its pooled mean
log-loss difference (joint - independence) is negative by at least 2
clustered standard errors. The joint pricer is USED (GATE_PASSED = True:
the bet checker prices same-game ML x over/under slips with it instead
of withholding a verdict) only if BOTH A and B pass. C-F are reported,
never decisive. Per-season numbers are reported too, not decisive. Any
over/under leg stays subject to the totals model's own gate
(models.totals.GATE_PASSED): no BET verdict while it is False.
Production OT_BETA: refit by the same method on every completed game
2020-21 through 2025-26 and written into OT_BETA by hand.

STATUS (2026-10-04): GATE FAILED. The bet checker keeps withholding a
verdict on same-game slips (WITHHELD); GATE_PASSED stays False.
2,408 games: 1,398 in 2024-25 (10-book consensus; every 2024-25 game
had a half-point line) and 1,010 in 2025-26 (DraftKings). Out-of-fold
OT_BETA 0.345 (2024-25 fold) and 0.366 (2025-26 fold). Joint minus
independence, 4-way log loss (negative = joint better; z = diff / SE):
  A  +0.00005  SE 0.00046  z +0.11   fail   (2024-25 +0.00027, 2025-26 -0.00025)
  B  +0.00008  SE 0.00046  z +0.18   fail   (+0.00032, -0.00024)
  C  +0.00005  SE 0.00046  z +0.10   fail   (+0.00027, -0.00026)
  D  +0.00011  SE 0.00052  z +0.20   fail   (+0.00036, -0.00025)
  E  +0.00011  SE 0.00049  z +0.22   fail   (+0.00033, -0.00021)
  F  +0.00031  SE 0.00052  z +0.61   fail   (+0.00071, -0.00024)
(4-way log loss itself: A 1.35807 joint vs 1.35802 independence.)
No variant comes close; each is slightly behind independence pooled,
ahead in 2025-26 and behind in 2024-25.
What the numbers say:
- The joint model says "favourite wins AND over" happens 0.44 points
  more often than the product (A: 30.14% vs 29.70% on average). In the
  data it happened 29.07% of the time; the realised link between the
  two legs (covariance of the favourite-win and over surprises) was
  +0.0011 ± 0.0050 (prediction +0.0044): the right sign, about a
  quarter of the size, and well inside the noise.
- The test could not have passed. If the joint model were exactly
  right, its expected gain would be 0.00027 per game against an SE of
  0.00047 (z -0.57): the chance of clearing 2 SE with 2,408 games was
  about 8%, and an expected z of -2 needs about 30,000 games (more than
  twenty NHL seasons). So this is "not proven", not "proven wrong": the
  dependence is too small to measure on two seasons of one game market.
- The margin reweighting (D vs A) and the overtime lean (E vs A) change
  the result by less than 0.0001; so does the base grid (C vs A).
- The production overtime lean refit on every completed game 2020-21
  through 2025-26 is 0.2931 (OT_BETA): overtime is close to a coin flip,
  the stronger side wins it slightly more often.
What it means for betting: a book's same-game parlay price builds in
some correlation, so treating the legs as independent (or using this
joint model) gives an EV that cannot be trusted either way. The checker
keeps saying so. The pricer stays available for research (price_legs).
Re-run when far more priced games exist (live 2026-27 snapshots add
about 1,300 games a season) or when a sharper market tests it (books'
own same-game parlay prices, if they are ever collected).

Run: python -m betting.sgp --evaluate   (read-only; writes nothing)
"""
import argparse
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from models.totals import MARGIN_WEIGHTS, MAX_GOALS, joint_pmf

logger = logging.getLogger("nhl.betting.sgp")

# Set by hand from the pre-registered rule above (both A and B must pass)
GATE_PASSED = False
PASS_SE = 2.0
VALIDATION_SEASONS = (20242025, 20252026)
# Production overtime lean (see joint_grid step 4); set by hand from the
# evaluation's fit on every completed game 2020-21..2025-26 (2026-10-04)
OT_BETA = 0.2931
OT_PROB_CLIP = (0.02, 0.98)
TARGET_CLIP = (0.005, 0.995)
ELO_HOME_ADV = 50.0
MAX_ABS_ML = 1000
K1 = MAX_GOALS + 1


# ── Grid geometry (pure) ───────────────────────────────────────────

def _cells(k1: int = K1):
    h, a = np.meshgrid(np.arange(k1), np.arange(k1), indexing="ij")
    return h, a


def settlement_totals(k1: int = K1) -> np.ndarray:
    """(k1, k1) settlement total of each regulation score cell: h + a,
    plus the OT/SO winner's goal on a regulation tie."""
    h, a = _cells(k1)
    return h + a + (h == a)


def ot_home_prob(p_home, beta: float = None) -> np.ndarray:
    """P(home wins OT/SO) from the full-game P(home wins): 0.5 +
    beta * (p - 0.5), clipped."""
    beta = OT_BETA if beta is None else beta
    return np.clip(0.5 + beta * (np.asarray(p_home, dtype=float) - 0.5),
                   *OT_PROB_CLIP)


def home_win_prob(grid: np.ndarray, q_ot: float) -> float:
    """P(home wins, OT/SO included) from a regulation grid."""
    h, a = _cells(grid.shape[0])
    return float(grid[h > a].sum() + q_ot * np.trace(grid))


def _over_given_no_push(grid: np.ndarray, line: float) -> float:
    t = settlement_totals(grid.shape[0])
    over = grid[t > line].sum()
    push = grid[t == line].sum()
    return float(over / max(1.0 - push, 1e-12))


def _solve(f, lo: float, hi: float) -> float:
    """Root of an increasing f on [lo, hi]; the nearer bound when the
    target is out of reach."""
    from scipy.optimize import brentq
    flo, fhi = f(lo), f(hi)
    if flo >= 0:
        return lo
    if fhi <= 0:
        return hi
    return brentq(f, lo, hi, xtol=1e-10)


def tilt_total(grid: np.ndarray, p_over: float, line: float) -> np.ndarray:
    """Grid with its settlement total exponentially tilted so P(over the
    line | no push) equals p_over. Regulation margins within a total keep
    their relative weights."""
    t = settlement_totals(grid.shape[0])
    target = float(np.clip(p_over, *TARGET_CLIP))

    def tilted(phi):
        w = grid * np.exp(phi * (t - 6.0))
        return w / w.sum()

    phi = _solve(lambda x: _over_given_no_push(tilted(x), line) - target,
                 -5.0, 5.0)
    return tilted(phi)


def tilt_margin(grid: np.ndarray, p_home: float, q_ot: float) -> np.ndarray:
    """Grid with each settlement total's cells tilted by
    exp(theta * (home - away)) and rescaled to keep P(total = t), theta
    solved so P(home wins, OT/SO incl.) equals p_home. The total
    distribution is unchanged."""
    k1 = grid.shape[0]
    h, a = _cells(k1)
    d = (h - a).astype(float)
    t = settlement_totals(k1)
    p_t = np.bincount(t.ravel(), weights=grid.ravel(), minlength=2 * k1)
    target = float(np.clip(p_home, *TARGET_CLIP))

    def tilted(theta):
        w = grid * np.exp(theta * d)
        z = np.bincount(t.ravel(), weights=w.ravel(), minlength=2 * k1)
        scale = np.divide(p_t, z, out=np.zeros_like(p_t), where=z > 0)
        return w * scale[t]

    theta = _solve(lambda x: home_win_prob(tilted(x), q_ot) - target,
                   -8.0, 8.0)
    return tilted(theta)


def base_grid(pmf_h: np.ndarray, pmf_a: np.ndarray,
              margin_weights=MARGIN_WEIGHTS) -> np.ndarray:
    """(K+1, K+1) regulation grid of one game from per-side PMFs, margin-
    reweighted like the totals model (None: independent product)."""
    return joint_pmf(np.asarray(pmf_h, float)[None],
                     np.asarray(pmf_a, float)[None], margin_weights)[0]


def joint_grid(pmf_h, pmf_a, p_home: Optional[float] = None,
               p_over: Optional[float] = None, line: Optional[float] = None,
               margin_weights=MARGIN_WEIGHTS,
               beta: Optional[float] = None) -> tuple:
    """(regulation grid, q_ot) for one game: base grid, then the total
    tilt (when p_over and line are given), then the margin tilt (when
    p_home is given). q_ot is the home team's OT/SO win chance."""
    g = base_grid(pmf_h, pmf_a, margin_weights)
    if p_over is not None:
        if line is None:
            raise ValueError("a total tilt needs the line")
        g = tilt_total(g, p_over, line)
    if p_home is None:
        # no win-probability target: the OT lean follows the grid's own
        # strength (its win chance with a coin-flip overtime)
        return g, float(ot_home_prob(home_win_prob(g, 0.5), beta))
    q = float(ot_home_prob(p_home, beta))
    return tilt_margin(g, p_home, q), q


def final_scores(grid: np.ndarray, q_ot: float) -> np.ndarray:
    """(K+2, K+2) final-score distribution (OT/SO goal credited):
    F[home_final, away_final]."""
    k1 = grid.shape[0]
    f = np.zeros((k1 + 1, k1 + 1))
    h, a = _cells(k1)
    off = h != a
    f[h[off], a[off]] += grid[off]
    i = np.arange(k1)
    tie = np.diag(grid)
    f[i + 1, i] += q_ot * tie
    f[i, i + 1] += (1.0 - q_ot) * tie
    return f


# ── Legs ───────────────────────────────────────────────────────────

MARKETS = ("ml", "total", "team_total", "spread")


@dataclass
class SgpLeg:
    """One leg of a same-game slip.
    market: 'ml' (side HOME/AWAY), 'total' (side OVER/UNDER, line),
    'team_total' (team HOME/AWAY, side OVER/UNDER, line) or 'spread'
    (puck line: side HOME/AWAY, line = that side's handicap, e.g. -1.5)."""
    market: str
    side: str
    line: Optional[float] = None
    team: Optional[str] = None


def leg_masks(leg: SgpLeg, n: int) -> tuple:
    """(win, push) boolean (n, n) masks over final scores F[h, a]."""
    hf, af = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    side = leg.side.upper()
    if leg.market not in MARKETS:
        raise ValueError(f"unknown market {leg.market!r}")
    if leg.market == "ml":
        if side not in ("HOME", "AWAY"):
            raise ValueError(f"moneyline side must be HOME or AWAY: {side!r}")
        win = hf > af if side == "HOME" else af > hf
        return win, np.zeros_like(win)
    if leg.line is None:
        raise ValueError(f"a {leg.market} leg needs a line")
    line = float(leg.line)
    if leg.market == "spread":
        if side not in ("HOME", "AWAY"):
            raise ValueError(f"puck-line side must be HOME or AWAY: {side!r}")
        m = (hf - af if side == "HOME" else af - hf) + line
        return m > 0, m == 0
    if leg.market == "total":
        x = hf + af
    else:
        team = (leg.team or "").upper()
        if team not in ("HOME", "AWAY"):
            raise ValueError("a team_total leg needs team HOME or AWAY")
        x = hf if team == "HOME" else af
    if side not in ("OVER", "UNDER"):
        raise ValueError(f"over/under side must be OVER or UNDER: {side!r}")
    return (x > line if side == "OVER" else x < line), x == line


def price_legs(f: np.ndarray, legs: Sequence[SgpLeg],
               decimals: Optional[Sequence[float]] = None) -> dict:
    """Joint pricing of same-game legs on a final-score grid F.
    p_all_win: P(every leg wins). p_win/p_push: each leg's own chances
    (its marginals in F). p_independent: the product of the legs' win
    chances. exp_multiplier (when decimals are given): expected payout
    per unit staked under leg-by-leg settlement (a pushed leg pays 1x,
    a lost leg 0), i.e. sum over scores of F * prod(dec_i if win, 1 if
    push, 0 if lose)."""
    n = f.shape[0]
    masks = [leg_masks(l, n) for l in legs]
    all_win = np.ones_like(f, dtype=bool)
    for w, _ in masks:
        all_win &= w
    out = {
        "p_all_win": float(f[all_win].sum()),
        "p_win": [float(f[w].sum()) for w, _ in masks],
        "p_push": [float(f[p].sum()) for _, p in masks],
    }
    out["p_independent"] = float(np.prod(out["p_win"]))
    if decimals is not None:
        mult = np.ones_like(f)
        for (w, p), d in zip(masks, decimals):
            mult *= np.where(w, float(d), np.where(p, 1.0, 0.0))
        out["exp_multiplier"] = float((f * mult).sum())
    return out


# ── Validation (DB, read-only) ─────────────────────────────────────

def _novig(p_a, p_b):
    from features.util import american_implied_prob
    pa = np.array([american_implied_prob(x) for x in p_a], dtype=float)
    pb = np.array([american_implied_prob(x) for x in p_b], dtype=float)
    return pa / (pa + pb)


def consensus_from_quotes(q: pd.DataFrame) -> pd.DataFrame:
    """Per game, the consensus no-vig market from closing book quotes.
    q columns: game_id, book, market ('h2h'|'totals'), side
    ('home'|'away'|'over'|'under'), price, point. Returns game_id,
    p_home, line, p_over, n_ml_books, n_ou_books (pre-registered rule:
    median no-vig across books; the half-point line most books quote,
    tie -> median P(over) nearest 50%; no half-point line -> dropped)."""
    cols = ["game_id", "p_home", "line", "p_over", "n_ml_books", "n_ou_books"]
    if q.empty:
        return pd.DataFrame(columns=cols)
    h2h = q[q["market"] == "h2h"].pivot_table(
        index=["game_id", "book"], columns="side", values="price",
        aggfunc="first").dropna(subset=["home", "away"]).reset_index()
    h2h["nv"] = _novig(h2h["home"], h2h["away"])
    ml = h2h.groupby("game_id")["nv"].agg(["median", "count"]).rename(
        columns={"median": "p_home", "count": "n_ml_books"})

    tot = q[q["market"] == "totals"].copy()
    tot["point"] = tot["point"].astype(float)
    tot = tot[(tot["point"] * 2) % 2 == 1]                # half-point lines
    tot = tot.pivot_table(index=["game_id", "book", "point"], columns="side",
                          values="price", aggfunc="first")
    if tot.empty:
        return pd.DataFrame(columns=cols)
    tot = tot.dropna(subset=["over", "under"]).reset_index()
    tot["nv"] = _novig(tot["over"], tot["under"])
    by_line = (tot.groupby(["game_id", "point"])["nv"]
               .agg(["median", "count"]).reset_index())
    by_line["balance"] = (by_line["median"] - 0.5).abs()
    best = (by_line.sort_values(["game_id", "count", "balance"],
                                ascending=[True, False, True])
            .drop_duplicates("game_id")
            .rename(columns={"point": "line", "median": "p_over",
                             "count": "n_ou_books"})
            .set_index("game_id")[["line", "p_over", "n_ou_books"]])
    out = ml.join(best, how="inner").reset_index()
    return out[cols]


_ODDS_HISTORY_CLOSE_SQL = """
    WITH last AS (
        SELECT o.game_id, o.book, o.market, max(o.snapshot_ts) AS ts
        FROM raw.odds_history o
        JOIN raw.games g USING (game_id)
        WHERE g.season = :season AND g.start_time_utc IS NOT NULL
          AND g.game_state IN ('FINAL', 'OFF')
          AND o.snapshot_ts < (g.start_time_utc AT TIME ZONE 'UTC')
        GROUP BY o.game_id, o.book, o.market
    )
    SELECT o.game_id, o.book, o.market, o.side, o.price, o.point
    FROM raw.odds_history o
    JOIN last l ON o.game_id = l.game_id AND o.book = l.book
               AND o.market = l.market AND o.snapshot_ts = l.ts
"""

_DK_CLOSE_SQL = """
    SELECT h.game_id, h.home_ml, h.away_ml, h.over_under AS line,
           h.over_price, h.under_price
    FROM raw.historical_odds h
    JOIN raw.games g USING (game_id)
    WHERE h.provider = 'DraftKings' AND g.season = :season
      AND g.game_state IN ('FINAL', 'OFF')
"""


def dk_market(rows: pd.DataFrame) -> pd.DataFrame:
    """2025-26 DraftKings closing rows -> game_id, p_home, line, p_over
    (pre-registered filters: both moneylines and both O/U prices present,
    |moneyline| < MAX_ABS_ML, half-point line)."""
    r = rows.dropna(subset=["home_ml", "away_ml", "line", "over_price",
                            "under_price"]).copy()
    r["line"] = r["line"].astype(float)
    r = r[(r["home_ml"].abs() < MAX_ABS_ML) & (r["away_ml"].abs() < MAX_ABS_ML)
          & ((r["line"] * 2) % 2 == 1)]
    r["p_home"] = _novig(r["home_ml"], r["away_ml"])
    r["p_over"] = _novig(r["over_price"], r["under_price"])
    r["n_ml_books"] = 1
    r["n_ou_books"] = 1
    return r[["game_id", "p_home", "line", "p_over", "n_ml_books",
              "n_ou_books"]].reset_index(drop=True)


def load_validation_market(conn) -> pd.DataFrame:
    """Both validation seasons' closing no-vig markets with outcomes:
    game_id, season, source, p_home, line, p_over, home_score,
    away_score."""
    from sqlalchemy import text
    q24 = pd.read_sql(text(_ODDS_HISTORY_CLOSE_SQL), conn,
                      params={"season": VALIDATION_SEASONS[0]})
    m24 = consensus_from_quotes(q24).assign(source="odds_history consensus")
    dk = dk_market(pd.read_sql(text(_DK_CLOSE_SQL), conn,
                               params={"season": VALIDATION_SEASONS[1]}))
    dk = dk.assign(source="DraftKings close")
    mk = pd.concat([m24, dk], ignore_index=True)
    res = pd.read_sql(text("""
        SELECT game_id, season, home_score, away_score FROM raw.games
        WHERE game_state IN ('FINAL', 'OFF') AND home_score IS NOT NULL
    """), conn)
    return mk.merge(res, on="game_id", how="inner")


def fit_ot_beta(p_home, home_won) -> float:
    """Maximum-likelihood OT_BETA from games that went to OT/SO:
    P(home wins OT) = 0.5 + beta * (p_home - 0.5)."""
    from scipy.optimize import minimize_scalar
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(home_won, dtype=bool)

    def nll(b):
        q = ot_home_prob(p, b)
        return -np.sum(np.where(y, np.log(q), np.log(1.0 - q)))

    return float(minimize_scalar(nll, bounds=(-2.0, 4.0),
                                 method="bounded").x)


def elo_home_prob(home_elo, away_elo) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-((np.asarray(home_elo, float) + ELO_HOME_ADV)
                                   - np.asarray(away_elo, float)) / 400.0))


def _ot_frame(conn) -> pd.DataFrame:
    from sqlalchemy import text
    return pd.read_sql(text("""
        SELECT g.game_id, g.season, g.date, g.home_score, g.away_score,
               g.is_ot, m.home_elo, m.away_elo
        FROM raw.games g JOIN features.matchup m USING (game_id)
        WHERE g.game_state IN ('FINAL', 'OFF')
    """), conn)


def ot_beta_before(ot: pd.DataFrame, cutoff) -> float:
    """OT_BETA fitted on OT/SO games dated strictly before cutoff (a
    fold's training window) with the pre-game Elo home expectation."""
    d = ot[(pd.to_datetime(ot["date"]) < pd.Timestamp(cutoff))
           & ot["is_ot"].fillna(False).astype(bool)].dropna(
        subset=["home_elo", "away_elo"])
    return fit_ot_beta(elo_home_prob(d["home_elo"], d["away_elo"]),
                       d["home_score"] > d["away_score"])


def totals_oof(val_seasons: Iterable[int] = VALIDATION_SEASONS) -> pd.DataFrame:
    """Out-of-fold totals-model rates for the validation seasons, the
    same fold loop as models.totals.run_totals: game_id, season,
    fold_cutoff, lam_h, lam_a (drift-corrected), env_h, env_a, w0..w4
    (the fold's margin weights)."""
    from models.baseline import PURGE_DAYS, walk_forward_folds
    from models.totals import (ENV_PRIOR_RATE, apply_drift,
                               booster_adjustment, drift_shift, env_rates,
                               fit_margin_weights, fit_totals_fold,
                               load_totals_dataset, poisson_pmf,
                               predict_lambdas)
    Xh, Xa, y_h, y_a, meta, _ = load_totals_dataset()
    env_h = env_rates(meta["date"], y_h, ENV_PRIOR_RATE["home"])
    env_a = env_rates(meta["date"], y_a, ENV_PRIOR_RATE["away"])
    seasons, dates = meta["season"].to_numpy(), meta["date"].to_numpy()
    parts = []
    for fold in walk_forward_folds(meta):
        if fold.val_season not in set(val_seasons):
            continue
        tr, val = fold.train_idx, fold.val_idx
        fm = fit_totals_fold(Xh, Xa, y_h, y_a, env_h, env_a, tr,
                             meta["date"],
                             is_playoff=meta["is_playoff"].to_numpy())
        lam_h, lam_a = predict_lambdas(fm, Xh[val], Xa[val], env_h[val],
                                       env_a[val])
        adj = booster_adjustment(lam_h, lam_a, env_h[val], env_a[val])
        lam_h, lam_a = apply_drift(lam_h, lam_a,
                                   drift_shift(seasons[val], dates[val], adj))
        w = fit_margin_weights(poisson_pmf(env_h[tr]), poisson_pmf(env_a[tr]),
                               y_h[tr], y_a[tr])
        start = meta["date"].iloc[val].min()
        part = pd.DataFrame({
            "game_id": meta["game_id"].to_numpy()[val],
            "season": fold.val_season,
            "fold_cutoff": start - pd.Timedelta(days=PURGE_DAYS),
            "lam_h": lam_h, "lam_a": lam_a,
            "env_h": env_h[val], "env_a": env_a[val]})
        for i, wi in enumerate(w):
            part[f"w{i}"] = wi
        parts.append(part)
        logger.info(f"totals fold {fold.val_season}: {len(val)} games, "
                    f"margin weights {np.round(w, 3).tolist()}")
    return pd.concat(parts, ignore_index=True)


def four_way(f: np.ndarray, home_fav: bool, line: float) -> np.ndarray:
    """[fav&over, fav&under, dog&over, dog&under] probabilities from a
    final-score grid (half-point line)."""
    fav, dog = ("HOME", "AWAY") if home_fav else ("AWAY", "HOME")
    out = []
    for side in (fav, dog):
        for ou in ("OVER", "UNDER"):
            out.append(price_legs(f, [SgpLeg("ml", side),
                                      SgpLeg("total", ou, line)])["p_all_win"])
    return np.array(out)


def four_way_independent(f: np.ndarray, home_fav: bool,
                         line: float) -> np.ndarray:
    """The independence product of the SAME grid's two marginals."""
    n = f.shape[0]
    p_home = float(f[leg_masks(SgpLeg("ml", "HOME"), n)[0]].sum())
    p_over = float(f[leg_masks(SgpLeg("total", "OVER", line), n)[0]].sum())
    p_fav = p_home if home_fav else 1.0 - p_home
    return np.array([p_fav * p_over, p_fav * (1 - p_over),
                     (1 - p_fav) * p_over, (1 - p_fav) * (1 - p_over)])


def outcome_cell(home_score, away_score, home_fav: bool, line: float) -> int:
    fav_won = (home_score > away_score) == home_fav
    over = home_score + away_score > line
    return (0 if fav_won else 2) + (0 if over else 1)


def clustered_se(diff, clusters) -> float:
    """Standard error of mean(diff) with observations clustered (here:
    by game)."""
    d = pd.Series(np.asarray(diff, float)).groupby(np.asarray(clusters)).sum()
    n_obs = len(diff)
    g = len(d)
    if g < 2:
        return float("nan")
    sizes = pd.Series(1, index=range(n_obs)).groupby(np.asarray(clusters)).sum()
    m = float(np.mean(diff))
    resid = d.to_numpy() - m * sizes.reindex(d.index).to_numpy()
    return float(np.sqrt(g / (g - 1) * np.sum(resid ** 2)) / n_obs)


VARIANTS = {
    "A": dict(base="model", tilt_total=True, p_home="market", beta="fit",
              weights="fold"),
    "B": dict(base="model", tilt_total=False, p_home="market", beta="fit",
              weights="fold"),
    "C": dict(base="env", tilt_total=True, p_home="market", beta="fit",
              weights="fold"),
    "D": dict(base="model", tilt_total=True, p_home="market", beta="fit",
              weights=None),
    "E": dict(base="model", tilt_total=True, p_home="market", beta="zero",
              weights="fold"),
    "F": dict(base="model", tilt_total=False, p_home="lgbm", beta="fit",
              weights="fold"),
}
DECISIVE = ("A", "B")


def score_games(data: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Per game 4-way log loss of the joint pricer and of independence
    for one variant. data: validation rows joined to totals_oof (and
    p_lgbm for F, beta for the fitted-OT variants)."""
    from models.totals import poisson_pmf
    spec = VARIANTS[variant]
    rows = []
    for r in data.itertuples(index=False):
        lh, la = ((r.lam_h, r.lam_a) if spec["base"] == "model"
                  else (r.env_h, r.env_a))
        ph, pa = poisson_pmf(np.array([lh]))[0], poisson_pmf(np.array([la]))[0]
        w = (np.array([r.w0, r.w1, r.w2, r.w3, r.w4])
             if spec["weights"] == "fold" else None)
        p_home = r.p_home if spec["p_home"] == "market" else r.p_lgbm
        beta = r.beta if spec["beta"] == "fit" else 0.0
        g, q = joint_grid(ph, pa, p_home=p_home,
                          p_over=r.p_over if spec["tilt_total"] else None,
                          line=r.line, margin_weights=w, beta=beta)
        f = final_scores(g, q)
        home_fav = r.p_home >= 0.5
        pj = four_way(f, home_fav, r.line)
        pi = four_way_independent(f, home_fav, r.line)
        c = outcome_cell(r.home_score, r.away_score, home_fav, r.line)
        rows.append({"game_id": r.game_id, "season": r.season, "cell": c,
                     "ll_joint": -np.log(max(pj[c], 1e-12)),
                     "ll_indep": -np.log(max(pi[c], 1e-12)),
                     "p_fav_over_joint": pj[0], "p_fav_over_indep": pi[0],
                     "p_home_grid": home_win_prob(g, q),
                     "p_over_grid": float(pj[0] + pj[2])})
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> dict:
    """Pooled and per-season joint-minus-independence log loss."""
    def block(s):
        d = s["ll_joint"] - s["ll_indep"]
        se = clustered_se(d, s["game_id"])
        return {"n": int(len(s)), "ll_joint": float(s["ll_joint"].mean()),
                "ll_indep": float(s["ll_indep"].mean()),
                "diff": float(d.mean()), "se": se,
                "z": float(d.mean() / se) if se and se > 0 else float("nan"),
                "fav_over_rate": float((s["cell"] == 0).mean()),
                "fav_over_joint": float(s["p_fav_over_joint"].mean()),
                "fav_over_indep": float(s["p_fav_over_indep"].mean())}
    out = block(scored)
    out["passes"] = bool(out["diff"] <= -PASS_SE * out["se"])
    out["by_season"] = {int(k): block(v) for k, v in scored.groupby("season")}
    return out


def power_if_true(data: pd.DataFrame, variant: str = "A") -> dict:
    """How well the test could detect the joint model if it were exactly
    right (reported, not decisive): per game, the expected joint-minus-
    independence log loss under the joint's own probabilities (minus the
    KL divergence → how far apart two sets of probabilities are) and its
    variance. Returns the expected diff, its SE, z, the chance of
    clearing PASS_SE with these games, and the games an expected z of
    -PASS_SE would need."""
    from scipy.stats import norm

    from models.totals import poisson_pmf
    spec = VARIANTS[variant]
    kl, var = [], []
    for r in data.itertuples(index=False):
        lh, la = ((r.lam_h, r.lam_a) if spec["base"] == "model"
                  else (r.env_h, r.env_a))
        w = (np.array([r.w0, r.w1, r.w2, r.w3, r.w4])
             if spec["weights"] == "fold" else None)
        p_home = r.p_home if spec["p_home"] == "market" else r.p_lgbm
        g, q = joint_grid(poisson_pmf(np.array([lh]))[0],
                          poisson_pmf(np.array([la]))[0], p_home=p_home,
                          p_over=r.p_over if spec["tilt_total"] else None,
                          line=r.line, margin_weights=w,
                          beta=r.beta if spec["beta"] == "fit" else 0.0)
        f = final_scores(g, q)
        pj = np.clip(four_way(f, r.p_home >= 0.5, r.line), 1e-12, None)
        pi = np.clip(four_way_independent(f, r.p_home >= 0.5, r.line),
                     1e-12, None)
        dd = np.log(pi) - np.log(pj)        # joint minus independence
        kl.append(float(-(pj @ dd)))
        var.append(float(pj @ dd ** 2 - (pj @ dd) ** 2))
    kl, var = np.array(kl), np.array(var)
    n = len(kl)
    exp_diff, se = -float(kl.mean()), float(np.sqrt(var.sum()) / n)
    return {"n": n, "expected_diff": exp_diff, "se": se,
            "z": exp_diff / se if se > 0 else float("nan"),
            "power": float(norm.cdf((-PASS_SE * se - exp_diff) / se))
            if se > 0 else float("nan"),
            "games_needed": int(np.ceil((PASS_SE * np.sqrt(var.mean())
                                         / kl.mean()) ** 2))
            if kl.mean() > 0 else None}


def build_validation_data(with_lgbm: bool = True) -> pd.DataFrame:
    """Validation games with markets, outcomes, out-of-fold totals rates,
    fold OT_BETA and (with_lgbm) the win model's out-of-fold P(home)."""
    from config.settings import engine
    with engine.connect() as conn:
        mk = load_validation_market(conn)
        ot = _ot_frame(conn)
    oof = totals_oof()
    data = mk.merge(oof.drop(columns="season"), on="game_id", how="inner")
    betas = {c: ot_beta_before(ot, c) for c in data["fold_cutoff"].unique()}
    data["beta"] = data["fold_cutoff"].map(betas)
    for s, c in data.groupby("season")["fold_cutoff"].first().items():
        logger.info(f"OT_BETA for {s} (fit before {c.date()}): {betas[c]:.3f}")
    if with_lgbm:
        from models.lgbm import run_lgbm
        lg = run_lgbm(register=False, plot=False)["oof"]
        data = data.merge(lg[["game_id", "prob_home"]].rename(
            columns={"prob_home": "p_lgbm"}), on="game_id", how="left")
    return data


def production_ot_beta() -> float:
    """OT_BETA refit on every completed game 2020-21 through 2025-26."""
    from config.settings import engine
    with engine.connect() as conn:
        ot = _ot_frame(conn)
    ot = ot[ot["season"] <= VALIDATION_SEASONS[-1]]
    return ot_beta_before(ot, pd.Timestamp.max)


def evaluate(variants: Sequence[str] = tuple(VARIANTS)) -> dict:
    """Run the pre-registered validation (read-only) and report every
    variant."""
    data = build_validation_data(with_lgbm="F" in variants)
    logger.info(f"validation games: {len(data)} "
                + str(data.groupby("season").size().to_dict()))
    report = {"n_games": int(len(data)), "variants": {}}
    for v in variants:
        d = data if VARIANTS[v]["p_home"] != "lgbm" \
            else data.dropna(subset=["p_lgbm"])
        report["variants"][v] = summarize(score_games(d, v))
    report["gate_passed"] = all(report["variants"][v]["passes"]
                                for v in DECISIVE if v in report["variants"])
    if "A" in variants:
        report["power_A"] = power_if_true(data, "A")
    report["production_ot_beta"] = production_ot_beta()
    return report


def format_report(report: dict) -> str:
    lines = [f"SGP joint pricer validation: {report['n_games']} games"]
    for v, s in report["variants"].items():
        lines.append(
            f"  {v}: n={s['n']} joint {s['ll_joint']:.5f} vs independence "
            f"{s['ll_indep']:.5f} diff {s['diff']:+.5f} (SE {s['se']:.5f}, "
            f"z {s['z']:+.2f}) {'PASS' if s['passes'] else 'fail'} | "
            f"fav&over actual {s['fav_over_rate']:.4f}, joint "
            f"{s['fav_over_joint']:.4f}, indep {s['fav_over_indep']:.4f}")
        for season, b in s["by_season"].items():
            lines.append(f"      {season}: n={b['n']} diff {b['diff']:+.5f} "
                         f"(SE {b['se']:.5f}, z {b['z']:+.2f}) fav&over "
                         f"{b['fav_over_rate']:.4f} / {b['fav_over_joint']:.4f}"
                         f" / {b['fav_over_indep']:.4f}")
    lines.append(f"  decisive variants {DECISIVE}: GATE "
                 f"{'PASSED' if report['gate_passed'] else 'FAILED'}")
    if "power_A" in report:
        p = report["power_A"]
        lines.append(f"  if A were exactly right: expected diff "
                     f"{p['expected_diff']:+.5f} (SE {p['se']:.5f}, z "
                     f"{p['z']:+.2f}), chance of passing {p['power']:.0%}, "
                     f"games needed for z -{PASS_SE:g}: {p['games_needed']}")
    lines.append(f"  production OT_BETA (2020-21..2025-26): "
                 f"{report['production_ot_beta']:.4f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Same-game parlay joint pricer: run the pre-registered "
                    "validation (read-only)")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the validation against the closing markets")
    parser.add_argument("--variants", default="".join(VARIANTS),
                        help="which variants to run, e.g. AB")
    args = parser.parse_args(argv)
    if not args.evaluate:
        parser.print_help()
        return None
    report = evaluate(tuple(args.variants.upper()))
    print(format_report(report))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
