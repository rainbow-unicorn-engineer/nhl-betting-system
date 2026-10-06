"""
models/props_market_check.py
The pre-registered market check of the skater shots-on-goal (SOG) props
model (models/props_sog.py; v2, and since v3 the adopted variant) against
stored prop PRICES. Read only: it
SELECTs from the database, writes nothing, registers nothing.

Terms:
- Prop: a bet on one player's own number, here "over/under N.5 shots on
  goal" in one game.
- Decimal odds d: what 1 unit staked returns in total on a win (American
  -120 -> 1 + 100/120 = 1.833; +110 -> 2.10).
- Implied probability 1/d: the break-even win rate at that price. The two
  sides' implied probabilities add up to more than 1; the excess is the
  bookmaker's margin (the "vig").
- No-vig probability: the over's implied probability rescaled so the two
  sides add up to 1 — the market's own fair estimate:
      p_mkt(over) = (1/d_over) / (1/d_over + 1/d_under)
- Log loss: -log(probability given to what happened); lower is better.
- Brier score: mean squared error of a probability; lower is better.
- ECE (expected calibration error): average gap between predicted and
  actual frequency over 10 equal-width probability bins.
- Push: on an integer line (say 2), exactly 2 shots refunds the bet.

PRE-REGISTERED (fixed before any result was seen; every run reports all of
it and nothing else is a variant):
- Model probabilities: the 2025-26 walk-forward VALIDATION fold of
  props_sog v2, i.e. out-of-fold predictions from run_props(register=False)
  (booster trained only on earlier seasons, with the purge gap). For each
  player-game, P(over) = P(SOG > line), P(under) = P(SOG < line) under that
  row's negative binomial; on an integer line the push mass P(SOG = line)
  is removed and both are renormalised.
- Price rows: raw.prop_odds_hist, market player_shots_on_goal, two-sided
  pairs (over AND under at the same line, book and player-game). Pre-game
  per the loader (ingestion/espn_props.py): the stored current pair
  (over_price, under_price) when the row's last_updated is strictly before
  event_start; otherwise the opening pair (over_price_open,
  under_price_open), which the loader keeps only at the line the row is
  stored at. A current pair on a row whose last_updated is at or after
  puck drop is never used (the opening pair is, if complete). Rows with
  no complete pre-game pair are excluded and counted.
- Matching to the model by (game_id, player_id); unmatched props are
  counted by reason: no player id, did not play (no skater row with
  TOI > 0), not eligible (< 5 prior appearances), other.
- Outcome: actual SOG (raw.skater_games.shots); pushes dropped.
- PRIMARY (per book and pooled): per prop, log loss of the model's
  over-probability minus log loss of the no-vig market over-probability.
  Mean difference, SE clustered by game:
      SE = sqrt( G/(G-1) x sum_g (sum_{i in g} (d_i - mean d))^2 ) / n
  The model BEATS THE MARKET for a book only if mean + 1.96 x SE < 0 AND
  n >= 300 props. Brier and 10-bin ECE of both are reported.
- SECONDARY (information only, not a gate): flat 1-unit bets at the
  ACTUAL quoted prices (vig included): the over when P_model(over) -
  1/d_over >= T, else the under when P_model(under) - 1/d_under >= T (at
  most one side; with a margin both can never qualify), T = 0.04 and 0.06.
  Bets, hit rate, ROI with a game-clustered bootstrap 95% interval (2,000
  resamples of games, seed 7). For T = 0.04 also split by line (0.5, 1.5,
  2.5, 3.5+) and by position (F, D).
- player_shots_on_goal_alternate (DraftKings "N+" milestones, over only)
  cannot be de-vigged: reported separately, information only.

v3 ADDITIONS (2026-10-04; written and committed before any v3 result,
with the props_sog v3 pre-registration):
- Model: the ADOPTED props_sog v3 variant (props_sog.DEFAULT_VARIANT
  after its adoption rule), using its 2025-26 out-of-fold fold exactly as
  above. Rows: every 2025-26 player_shots_on_goal row in
  raw.prop_odds_hist now loaded, including the late playoffs (DraftKings
  through the 2026 Final). The pairing, matching, primary test,
  thresholds and bootstrap are unchanged.
- The market check PASSES (and registration may be switched on) only if
  the POOLED primary test beats the market (mean + 1.96 SE < 0, n >= 300)
  AND every book with n >= 300 has a negative mean difference.
- MARKET-MEAN DIAGNOSTIC (information only, not a gate). It asks whether
  the model's misses against the market are in the LEVEL (the expected
  number of shots) or in the SHAPE (how spread out the count is around
  it). Per book, pooled, and per line group (0.5, 1.5, 2.5, 3.5+):
  * mu_mkt, the market's implied mean: the mean m with
    P_NB(SOG > line; m, alpha) = the no-vig over probability, where alpha
    is the model's own fold dispersion (bisection on [0.02, 15]).
  * mean SOG, mean mu_model and mean mu_mkt. The bias of each, mean(mu -
    SOG), with its game-clustered SE.
  * Paired count log loss: -log P_NB(actual SOG; mu_model, alpha) minus
    the same at mu_mkt. This scores the two MEANS on the whole count, not
    just one side of the line. Game-clustered SE.
  * Information slope beta: OLS of (SOG - mu_mkt) on (mu_model - mu_mkt),
    game-clustered SE. beta ~ 0: the model's disagreements with the
    market carry no information. beta ~ 1: they are right on average.
  * Level-only fix: mu_model times one constant per book (mean SOG / mean
    mu_model on those props; fitted in-sample, so an upper bound), and
    the primary log-loss gap re-computed.
  * Shape-only fix: alpha* = the maximum-likelihood dispersion of the
    actual SOG around mu_mkt on those props (in-sample), and the primary
    gap re-computed from the model's own means with alpha*.
  Reading rule: "level" if the level-only fix closes at least half of
  the primary gap. "Shape" if the shape-only fix closes at least half.
  Otherwise the gap is in the per-player means (which player-games the
  model rates above or below the market), and beta says whether those
  disagreements carry information (beta > 0 by 2 SE) or not.

RESULTS: v2 in props_sog.py STATUS (v2, "Market check"); v3 (the
adopted P3, every price row now loaded, the pass rule and the
market-mean diagnostic) in props_sog.py STATUS v3. Neither passes.

Usage: python -m models.props_market_check   (prints the report)
"""
import argparse
import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger("nhl.models.props_market_check")

SEASON = 20252026
MARKET = "player_shots_on_goal"
ALT_MARKET = "player_shots_on_goal_alternate"
GATE_Z = 1.96
GATE_MIN_N = 300
THRESHOLDS = (0.04, 0.06)
SPLIT_T = 0.04
N_BOOT = 2000
BOOT_SEED = 7
EPS = 1e-15
POOLED = "pooled"


# ── Prices (pure) ──────────────────────────────────────────────────

def american_to_decimal(price) -> np.ndarray:
    """American odds -> decimal odds: +a -> 1 + a/100, -a -> 1 + 100/a."""
    a = np.asarray(price, float)
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def no_vig_over(d_over, d_under) -> np.ndarray:
    """(1/d_over) / (1/d_over + 1/d_under)."""
    io, iu = 1.0 / np.asarray(d_over, float), 1.0 / np.asarray(d_under, float)
    return io / (io + iu)


def _has(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _complete(a, b) -> bool:
    return _has(a) and _has(b)


def _pregame(row) -> bool:
    """last_updated strictly before event_start (both known)."""
    lu, es = row.get("last_updated"), row.get("event_start")
    return (lu is not None and es is not None and not pd.isna(lu)
            and not pd.isna(es) and pd.Timestamp(lu) < pd.Timestamp(es))


def pregame_pair(row) -> tuple:
    """(over, under, source) of the pre-game two-sided pair a stored row
    gives, source 'current' or 'open'; (None, None, reason) when none.
    The current pair counts only when last_updated < event_start (the
    loader's rule): a current price on a row updated at or after puck drop
    is never used."""
    if _pregame(row) and _complete(row.get("over_price"), row.get("under_price")):
        return int(row["over_price"]), int(row["under_price"]), "current"
    if _complete(row.get("over_price_open"), row.get("under_price_open")):
        return int(row["over_price_open"]), int(row["under_price_open"]), "open"
    return None, None, "no_two_sided_pregame_pair"


def pair_prices(rows: pd.DataFrame) -> tuple:
    """One priced prop per stored row (book, game, player, line) with a
    complete pre-game pair: adds over_am, under_am, d_over, d_under,
    p_mkt (no-vig over) and price_source. Returns (props, counts)."""
    counts = {"rows": int(len(rows)), "current": 0, "open": 0,
              "no_two_sided_pregame_pair": 0,
              "current_ignored_after_puck_drop": 0}
    out = []
    for r in rows.to_dict("records"):
        o, u, src = pregame_pair(r)
        counts[src] += 1
        if src == "open" and _complete(r.get("over_price"), r.get("under_price")):
            counts["current_ignored_after_puck_drop"] += 1
        if o is None:
            continue
        out.append({**r, "over_am": o, "under_am": u, "price_source": src})
    props = pd.DataFrame(out, columns=list(rows.columns)
                         + ["over_am", "under_am", "price_source"])
    props["d_over"] = american_to_decimal(props["over_am"]) if len(props) else []
    props["d_under"] = american_to_decimal(props["under_am"]) if len(props) else []
    props["p_mkt"] = no_vig_over(props["d_over"], props["d_under"]) if len(props) else []
    return props, counts


# ── Model probabilities (pure) ─────────────────────────────────────

def model_side_probs(mu, alpha, line) -> tuple:
    """(P(over), P(under), P(push)) per row under NB2(mu, alpha), with
    the push mass removed and both sides renormalised on an integer line.
    alpha and line may be per-row arrays."""
    from scipy.stats import nbinom, poisson
    mu = np.asarray(mu, float)
    alpha = np.broadcast_to(np.asarray(alpha, float), mu.shape)
    line = np.broadcast_to(np.asarray(line, float), mu.shape)
    k_over = np.floor(line)                    # over: SOG >= floor(line) + 1
    k_under = np.ceil(line) - 1.0              # under: SOG <= ceil(line) - 1
    integer = np.isclose(line, np.round(line))
    pois = alpha <= 1e-10
    r = np.where(pois, 1.0, 1.0 / np.where(pois, 1.0, alpha))
    q = r / (r + mu)
    p_over = np.where(pois, poisson.sf(k_over, mu), nbinom.sf(k_over, r, q))
    p_under = np.where(pois, poisson.cdf(k_under, mu), nbinom.cdf(k_under, r, q))
    p_push = np.where(integer, np.where(pois, poisson.pmf(np.round(line), mu),
                                        nbinom.pmf(np.round(line), r, q)), 0.0)
    keep = 1.0 - p_push
    return p_over / keep, p_under / keep, p_push


def validation_predictions(res: dict, season: int = SEASON) -> pd.DataFrame:
    """The out-of-fold rows of `season` from a run_props result, checked:
    `season` must be one of the walk-forward VALIDATION folds (a season
    that only trains has no out-of-fold prediction and raises), the row
    count must equal that fold's n, every prediction must be finite, and
    the rows must carry that fold's own NB alpha. Never refits anything."""
    folds = {int(f["val_season"]): f for f in res.get("folds", [])}
    if int(season) not in folds:
        raise ValueError(f"season {season} is not a walk-forward validation "
                         f"fold (folds: {sorted(folds)}): no out-of-fold "
                         f"predictions exist for it")
    fold = folds[int(season)]
    if fold.get("n_train", 0) <= 0:
        raise ValueError(f"fold {season} has no training rows")
    oof = res["oof"]
    v = oof[oof["season"] == season].copy()
    if len(v) != fold["n"]:
        raise ValueError(f"{len(v)} out-of-fold rows for {season}, fold "
                         f"scored {fold['n']}")
    if not (np.isfinite(v["mu_M"]).all() and np.isfinite(v["alpha_M"]).all()):
        raise ValueError(f"non-finite out-of-fold predictions in {season}")
    if not np.allclose(v["alpha_M"], fold["alpha_M"], atol=1e-4):
        raise ValueError(f"alpha_M of {season} rows is not the fold's own")
    if v.duplicated(["game_id", "player_id"]).any():
        raise ValueError("duplicate player-games in the out-of-fold rows")
    return v


# ── Matching (pure) ────────────────────────────────────────────────

UNMATCHED_REASONS = ("no_player_id", "did_not_play", "not_eligible", "other")


def match_props(props: pd.DataFrame, oof: pd.DataFrame,
                frame: pd.DataFrame) -> tuple:
    """Join priced props to out-of-fold predictions on (game_id,
    player_id). frame: the full played-skater feature frame (every
    player-game with TOI > 0 and its n_prior), used only to say WHY a prop
    is unmatched. Returns (matched, unmatched counts by book and reason)."""
    keys = ["game_id", "player_id"]
    p = props.copy()
    p["_pid"] = pd.to_numeric(p["player_id"], errors="coerce")
    has_pid = p["_pid"].notna()
    p = p.drop(columns=["player_id"]).rename(columns={"_pid": "player_id"})
    o = oof[keys + ["pos_group", "sog", "mu_M", "alpha_M"]].copy()
    o["player_id"] = o["player_id"].astype(float)
    o["game_id"] = o["game_id"].astype("int64")
    p["game_id"] = p["game_id"].astype("int64")
    m = p.merge(o, on=keys, how="left", indicator=True)
    m.index = p.index
    played = frame[keys + ["n_prior"]].copy()
    played["player_id"] = played["player_id"].astype(float)
    played["game_id"] = played["game_id"].astype("int64")
    played = played.drop_duplicates(keys)
    um = m[m["_merge"] != "both"]
    reason = pd.Series("other", index=um.index)
    reason[~has_pid.loc[um.index]] = "no_player_id"
    chk = um[has_pid.loc[um.index]][keys].reset_index().merge(
        played, on=keys, how="left").set_index("index")
    reason.loc[chk.index[chk["n_prior"].isna()]] = "did_not_play"
    reason.loc[chk.index[chk["n_prior"] < 5]] = "not_eligible"
    counts = {}
    for book in sorted(p["book"].unique()):
        sel = um["book"] == book
        counts[book] = {r: int((reason[sel] == r).sum()) for r in UNMATCHED_REASONS}
        counts[book]["matched"] = int(((m["book"] == book) & (m["_merge"] == "both")).sum())
    matched = m[m["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    return matched, counts


def attach_outcomes(matched: pd.DataFrame, outcomes: pd.DataFrame) -> tuple:
    """Add the actual SOG from raw.skater_games (columns game_id,
    player_id, shots), the model's side probabilities and the realised
    over flag; drop pushes. Returns (props, n_push)."""
    o = outcomes[["game_id", "player_id", "shots"]].copy()
    o["player_id"] = o["player_id"].astype(float)
    o["game_id"] = o["game_id"].astype("int64")
    df = matched.merge(o.drop_duplicates(["game_id", "player_id"]),
                       on=["game_id", "player_id"], how="left")
    if df["shots"].isna().any():
        raise ValueError("matched props without a skater_games outcome")
    if not np.array_equal(df["shots"].to_numpy(float), df["sog"].to_numpy(float)):
        raise ValueError("raw.skater_games shots disagree with the model's sog")
    line = df["line"].to_numpy(float)
    p_over, p_under, p_push = model_side_probs(df["mu_M"], df["alpha_M"], line)
    df["p_model"], df["p_model_under"], df["p_push"] = p_over, p_under, p_push
    push = df["shots"].to_numpy(float) == line
    df = df[~push].reset_index(drop=True)
    df["over_hit"] = (df["shots"].to_numpy(float) > df["line"].to_numpy(float)).astype(float)
    return df, int(push.sum())


# ── Scoring (pure) ─────────────────────────────────────────────────

def log_loss(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_mean_se(d, clusters) -> tuple:
    """(mean, SE) of d with the SE clustered by `clusters`:
    sqrt(G/(G-1) x sum_g S_g^2) / n, S_g = sum of (d_i - mean) in g."""
    d = np.asarray(d, float)
    n = len(d)
    mean = float(d.mean())
    s = pd.Series(d - mean).groupby(np.asarray(clusters)).sum().to_numpy()
    g = len(s)
    if g < 2:
        return mean, float("nan")
    return mean, float(math.sqrt(g / (g - 1) * float(np.sum(s ** 2))) / n)


def primary_block(df: pd.DataFrame) -> dict:
    """The pre-registered primary test on one set of props."""
    from models.baseline import expected_calibration_error
    y = df["over_hit"].to_numpy(float)
    pm, pk = df["p_model"].to_numpy(float), df["p_mkt"].to_numpy(float)
    ll_m, ll_k = log_loss(pm, y), log_loss(pk, y)
    mean, se = clustered_mean_se(ll_m - ll_k, df["game_id"].to_numpy())
    n = int(len(df))
    beats = bool(n >= GATE_MIN_N and np.isfinite(se) and mean + GATE_Z * se < 0)
    return {"n": n, "games": int(df["game_id"].nunique()),
            "over_rate": float(y.mean()) if n else float("nan"),
            "mean_p_model": float(pm.mean()), "mean_p_mkt": float(pk.mean()),
            "logloss_model": float(ll_m.mean()), "logloss_mkt": float(ll_k.mean()),
            "diff": mean, "se_game": se, "upper": mean + GATE_Z * se,
            "brier_model": float(np.mean((pm - y) ** 2)),
            "brier_mkt": float(np.mean((pk - y) ** 2)),
            "ece_model": expected_calibration_error(y, pm),
            "ece_mkt": expected_calibration_error(y, pk),
            "beats_market": beats}


def bet_side(p_over, p_under, d_over, d_under, t: float) -> np.ndarray:
    """+1 bet the over, -1 the under, 0 no bet. The over when P(over) -
    1/d_over >= t, else the under when P(under) - 1/d_under >= t; if both
    ever qualified, the larger edge wins (ties: the over)."""
    eo = np.asarray(p_over, float) - 1.0 / np.asarray(d_over, float)
    eu = np.asarray(p_under, float) - 1.0 / np.asarray(d_under, float)
    over = eo >= t - 1e-12
    under = eu >= t - 1e-12
    return np.where(over & (~under | (eo >= eu)), 1, np.where(under, -1, 0))


def bet_profits(df: pd.DataFrame, t: float) -> pd.DataFrame:
    """The bets placed at threshold t, with profit per 1-unit stake at
    the actual quoted (vig-included) price."""
    side = bet_side(df["p_model"], df["p_model_under"], df["d_over"],
                    df["d_under"], t)
    b = df[side != 0].copy()
    s = side[side != 0]
    b["side"] = np.where(s > 0, "over", "under")
    won = np.where(s > 0, b["over_hit"] == 1, b["over_hit"] == 0)
    price = np.where(s > 0, b["d_over"], b["d_under"])
    b["won"] = won.astype(float)
    b["profit"] = np.where(won, price - 1.0, -1.0)
    return b


def bootstrap_roi(profit, games, n_boot: int = N_BOOT,
                  seed: int = BOOT_SEED) -> tuple:
    """Game-clustered bootstrap 95% interval of ROI = sum profit / bets:
    games (with at least one bet) are resampled with replacement."""
    if len(profit) == 0:
        return float("nan"), float("nan")
    g = pd.DataFrame({"p": np.asarray(profit, float), "g": np.asarray(games)}
                     ).groupby("g")["p"].agg(["sum", "count"])
    ps, ns = g["sum"].to_numpy(), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    roi = ps[idx].sum(axis=1) / ns[idx].sum(axis=1)
    lo, hi = np.percentile(roi, [2.5, 97.5])
    return float(lo), float(hi)


def betting_block(df: pd.DataFrame, t: float) -> dict:
    b = bet_profits(df, t)
    n = int(len(b))
    lo, hi = bootstrap_roi(b["profit"], b["game_id"])
    return {"t": t, "bets": n, "overs": int((b["side"] == "over").sum()),
            "unders": int((b["side"] == "under").sum()),
            "hit_rate": float(b["won"].mean()) if n else float("nan"),
            "profit": float(b["profit"].sum()),
            "roi": float(b["profit"].sum() / n) if n else float("nan"),
            "roi_lo": lo, "roi_hi": hi}


def line_group(line) -> np.ndarray:
    line = np.asarray(line, float)
    return np.where(line >= 3.5, "3.5+", np.char.mod("%.1f", line))


def book_frames(df: pd.DataFrame) -> dict:
    out = {b: g for b, g in df.groupby("book", sort=True)}
    out[POOLED] = df
    return out


def evaluate(df: pd.DataFrame) -> dict:
    """Primary and secondary results per book and pooled."""
    res = {}
    for book, g in book_frames(df).items():
        r = {"primary": primary_block(g),
             "betting": [betting_block(g, t) for t in THRESHOLDS],
             "by_line": {}, "by_position": {},
             "price_source": g["price_source"].value_counts().to_dict()}
        for lg in sorted(set(line_group(g["line"]))):
            r["by_line"][lg] = betting_block(g[line_group(g["line"]) == lg], SPLIT_T)
        for pos in ("F", "D"):
            r["by_position"][pos] = betting_block(g[g["pos_group"] == pos], SPLIT_T)
        res[book] = r
    return res


# ── v3: market-mean diagnostic (information only; pure) ───────────

MEAN_BOUNDS = (0.02, 15.0)
LEVEL_SHAPE_CLOSE = 0.5          # "closes at least half of the gap"


def implied_mean(p_over, alpha, line, bounds=MEAN_BOUNDS,
                 iters: int = 60) -> np.ndarray:
    """mu_mkt: per row, the mean m with P_NB(SOG > line; m, alpha) equal
    to the no-vig over probability (push mass removed on an integer line,
    as for the model), by bisection on `bounds`. P(over) rises with m, so
    the bisection is exact to (hi - lo) / 2^iters; a probability outside
    what `bounds` can give returns the nearer bound."""
    p_over = np.asarray(p_over, float)
    lo = np.full(p_over.shape, float(bounds[0]))
    hi = np.full(p_over.shape, float(bounds[1]))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = model_side_probs(mid, alpha, line)[0] < p_over
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def clustered_ols(y, x, clusters) -> dict:
    """OLS of y on [1, x] with game-clustered (sandwich) SEs and the same
    G/(G-1) small-sample factor as clustered_mean_se."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    sc = pd.DataFrame(X * e[:, None]).groupby(np.asarray(clusters)).sum().to_numpy()
    g = len(sc)
    meat = sc.T @ sc * (g / (g - 1) if g > 1 else float("nan"))
    v = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(v), 0, None))
    return {"intercept": float(b[0]), "slope": float(b[1]),
            "se_intercept": float(se[0]), "se_slope": float(se[1])}


def _gap(p_model, df) -> float:
    """Mean log loss(p_model) - log loss(no-vig market) over df's props."""
    y = df["over_hit"].to_numpy(float)
    return float(np.mean(log_loss(p_model, y) - log_loss(df["p_mkt"], y)))


def level_constants(df: pd.DataFrame) -> dict:
    """Per book: mean actual SOG / mean model mean on its props (fitted
    in-sample, so an upper bound on what a level fix could do)."""
    return {b: float(g["shots"].mean() / g["mu_M"].mean())
            for b, g in df.groupby("book", sort=True)}


def mean_block(df: pd.DataFrame, level_c: dict) -> dict:
    """The pre-registered market-mean diagnostic on one set of props
    (df: scored props with mu_mkt). Level fix: each prop's model mean
    times its BOOK's constant (level_c). Shape fix: alpha* = the ML
    dispersion of the actual SOG around mu_mkt on these props."""
    from models.props_sog import fit_nb_alpha, nb_nll_rows
    y = df["shots"].to_numpy(float)
    mu_m, mu_k = df["mu_M"].to_numpy(float), df["mu_mkt"].to_numpy(float)
    a = df["alpha_M"].to_numpy(float)
    line = df["line"].to_numpy(float)
    games = df["game_id"].to_numpy()
    out = {"n": int(len(df)), "games": int(df["game_id"].nunique()),
           "mean_sog": float(y.mean()), "mean_mu_model": float(mu_m.mean()),
           "mean_mu_mkt": float(mu_k.mean())}
    out["bias_model"], out["se_bias_model"] = clustered_mean_se(mu_m - y, games)
    out["bias_mkt"], out["se_bias_mkt"] = clustered_mean_se(mu_k - y, games)
    d = nb_nll_rows(y, mu_m, a) - nb_nll_rows(y, mu_k, a)
    out["count_ll_diff"], out["se_count_ll_diff"] = clustered_mean_se(d, games)
    ols = clustered_ols(y - mu_k, mu_m - mu_k, games)
    out["beta"], out["se_beta"] = ols["slope"], ols["se_slope"]
    out["beta_intercept"] = ols["intercept"]
    out["se_beta_intercept"] = ols["se_intercept"]
    gap = _gap(df["p_model"].to_numpy(float), df)
    c = df["book"].map(level_c).to_numpy(float)
    gap_level = _gap(model_side_probs(mu_m * c, a, line)[0], df)
    a_star = fit_nb_alpha(y, mu_k)
    gap_shape = _gap(model_side_probs(mu_m, a_star, line)[0], df)
    out.update({"gap": gap, "gap_level_fix": gap_level,
                "gap_shape_fix": gap_shape, "alpha_star": a_star,
                "mean_alpha_model": float(a.mean()),
                "level_c": {b: level_c[b] for b in sorted(set(df["book"]))}})
    if gap > 0:
        out["closed_level"] = (gap - gap_level) / gap
        out["closed_shape"] = (gap - gap_shape) / gap
    else:
        out["closed_level"] = out["closed_shape"] = float("nan")
    out["reading"] = mean_reading(out)
    return out


def mean_reading(b: dict) -> str:
    """The pre-registered reading rule."""
    if not b["gap"] > 0:
        return "no gap (the model is not behind the market here)"
    parts = [w for w, k in (("level", "closed_level"), ("shape", "closed_shape"))
             if b[k] >= LEVEL_SHAPE_CLOSE]
    if parts:
        return " and ".join(parts)
    if b["beta"] - 2.0 * b["se_beta"] > 0:
        return ("per-player means; the model's disagreements carry "
                "information (beta > 0 by 2 SE)")
    return ("per-player means; the model's disagreements carry no "
            "information (beta not > 0 by 2 SE)")


def mean_diagnostic(df: pd.DataFrame) -> dict:
    """Per book, pooled and per line group (0.5, 1.5, 2.5, 3.5+)."""
    df = df.copy()
    df["mu_mkt"] = implied_mean(df["p_mkt"], df["alpha_M"], df["line"])
    lc = level_constants(df)
    out = {b: mean_block(g, lc) for b, g in book_frames(df).items()}
    lg = line_group(df["line"])
    out["by_line"] = {k: mean_block(df[lg == k], lc) for k in sorted(set(lg))}
    return out


def market_check_passed(results: dict) -> dict:
    """The v3 pass rule: the POOLED primary test beats the market AND
    every book with n >= GATE_MIN_N has a negative mean difference."""
    pooled = results.get(POOLED, {}).get("primary", {})
    books = {b: r["primary"] for b, r in results.items() if b != POOLED}
    big = {b: bool(p["diff"] < 0) for b, p in books.items()
           if p["n"] >= GATE_MIN_N}
    beats = bool(pooled.get("beats_market", False))
    return {"pooled_beats": beats, "books_negative": big,
            "passed": bool(beats and all(big.values()))}


# ── Alternate milestones (information only) ────────────────────────

def alternate_block(alt: pd.DataFrame, oof: pd.DataFrame,
                    outcomes: pd.DataFrame) -> dict:
    """One-sided 'N+' overs: the pre-game over price (current if updated
    before puck drop, else the opening one), matched like the main props;
    the model's P(over) against the VIG-INCLUDED implied probability, and
    over bets at each T. No no-vig probability is possible."""
    rows = []
    for r in alt.to_dict("records"):
        price = r.get("over_price") if _pregame(r) and _has(r.get("over_price")) else None
        if price is None and _has(r.get("over_price_open")):
            price = r["over_price_open"]
        if price is not None:
            rows.append({**r, "over_am": int(price)})
    if not rows:
        return {"n_rows": int(len(alt)), "n": 0}
    a = pd.DataFrame(rows)
    a["player_id"] = pd.to_numeric(a["player_id"], errors="coerce")
    a = a[a["player_id"].notna()]
    o = oof[["game_id", "player_id", "mu_M", "alpha_M", "sog"]].copy()
    o["player_id"] = o["player_id"].astype(float)
    a["game_id"] = a["game_id"].astype("int64")
    o["game_id"] = o["game_id"].astype("int64")
    a = a.merge(o, on=["game_id", "player_id"], how="inner")
    if a.empty:
        return {"n_rows": int(len(alt)), "n": 0}
    a["d_over"] = american_to_decimal(a["over_am"])
    a["p_model"] = model_side_probs(a["mu_M"], a["alpha_M"], a["line"])[0]
    y = (a["sog"].to_numpy(float) > a["line"].to_numpy(float)).astype(float)
    out = {"n_rows": int(len(alt)), "n": int(len(a)),
           "games": int(a["game_id"].nunique()), "over_rate": float(y.mean()),
           "mean_p_model": float(a["p_model"].mean()),
           "mean_implied_with_vig": float((1 / a["d_over"]).mean()),
           "brier_model": float(np.mean((a["p_model"] - y) ** 2)),
           "betting": []}
    for t in THRESHOLDS:
        sel = (a["p_model"] - 1 / a["d_over"]) >= t - 1e-12
        prof = np.where(y[sel] == 1, a["d_over"][sel] - 1.0, -1.0)
        lo, hi = bootstrap_roi(prof, a["game_id"][sel])
        n = int(sel.sum())
        out["betting"].append({"t": t, "bets": n,
                               "hit_rate": float(y[sel].mean()) if n else float("nan"),
                               "roi": float(prof.sum() / n) if n else float("nan"),
                               "roi_lo": lo, "roi_hi": hi})
    return out


# ── Database (SELECT only) ─────────────────────────────────────────

def load_prop_rows(season: int = SEASON) -> pd.DataFrame:
    from sqlalchemy import text

    from config.settings import engine
    with engine.connect() as conn:
        return pd.read_sql(text("""
            SELECT h.game_id, h.book, h.market, h.player_name, h.espn_athlete_id,
                   h.player_id, h.line::float AS line, h.over_price, h.under_price,
                   h.over_price_open, h.under_price_open, h.last_updated,
                   h.event_start, g.date, g.game_type
            FROM raw.prop_odds_hist h JOIN raw.games g USING (game_id)
            WHERE g.season = :s AND h.market IN (:m, :a)
            ORDER BY g.date, h.game_id, h.book, h.espn_athlete_id, h.line
        """), conn, params={"s": season, "m": MARKET, "a": ALT_MARKET})


def load_outcomes(game_ids) -> pd.DataFrame:
    from sqlalchemy import text

    from config.settings import engine
    with engine.connect() as conn:
        return pd.read_sql(text("""
            SELECT game_id, player_id, shots, toi_seconds
            FROM raw.skater_games WHERE game_id = ANY(:g) AND toi_seconds > 0
        """), conn, params={"g": [int(g) for g in game_ids]})


# ── Run ────────────────────────────────────────────────────────────

def _object_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable integer price columns as Python objects (None, not NaN)."""
    df = df.copy()
    for c in ("over_price", "under_price", "over_price_open", "under_price_open"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def run_market_check(season: int = SEASON, frame: pd.DataFrame = None,
                     res: dict = None, rows: pd.DataFrame = None,
                     outcomes: pd.DataFrame = None, params=None,
                     variant: str = None) -> dict:
    """The whole pre-registered check. Every input can be given (tests);
    None reads it from the database (SELECT only). res: a
    props_sog.run_props result; it is only ever produced with
    register=False. variant: the props_sog v3 variant (default: its
    DEFAULT_VARIANT, the adopted one); not used when res is given."""
    from models import props_sog as P
    if frame is None:
        from features.player_shots import load_player_features
        frame = load_player_features()
    if res is None:
        res = P.run_props(register=False, frame=frame, params=params,
                          variant=variant)
    oof = validation_predictions(res, season)
    if rows is None:
        rows = load_prop_rows(season)
    rows = _object_rows(rows)
    main = rows[rows["market"] == MARKET].reset_index(drop=True)
    alt = rows[rows["market"] == ALT_MARKET].reset_index(drop=True)

    props, price_counts = pair_prices(main)
    price_counts_by_book = {}
    for book, g in main.groupby("book", sort=True):
        price_counts_by_book[book] = pair_prices(g)[1]
    matched, unmatched = match_props(props, oof, frame)
    if outcomes is None:
        outcomes = load_outcomes(sorted(set(matched["game_id"])))
    scored, n_push = attach_outcomes(matched, outcomes)
    fold = next(f for f in res["folds"] if int(f["val_season"]) == int(season))
    results = evaluate(scored) if len(scored) else {}
    return {
        "season": season, "model": res["pooled"].get("model"),
        "variant": res.get("variant", res["pooled"].get("variant")),
        "model_version": P.MODEL_VERSION,
        "fold": {"n": fold["n"], "n_train": fold["n_train"],
                 "nll_M": fold["nll_M"], "alpha_M": fold["alpha_M"]},
        "rows": {b: {"rows": int(len(g)), "games": int(g["game_id"].nunique()),
                     "first": str(g["date"].min()), "last": str(g["date"].max())}
                 for b, g in main.groupby("book", sort=True)},
        "prices": price_counts_by_book, "prices_total": price_counts,
        "unmatched": unmatched, "pushes_dropped": n_push,
        "results": results,
        "market_check": market_check_passed(results),
        "mean_diagnostic": mean_diagnostic(scored) if len(scored) else {},
        "alternate": {b: alternate_block(g, oof, outcomes)
                      for b, g in alt.groupby("book", sort=True)},
        "scored": scored,
    }


# ── Report ─────────────────────────────────────────────────────────

def _f(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _bet_line(b: dict, label: str = "") -> str:
    return (f"    {label}T={b['t']:.2f}: bets {b['bets']:5d}"
            + (f" (over {b['overs']}, under {b['unders']})" if "overs" in b else "")
            + f"  hit {_f(b['hit_rate'], 3)}  ROI {_f(b['roi'], 3)}"
            f"  95% [{_f(b['roi_lo'], 3)}, {_f(b['roi_hi'], 3)}]")


def format_report(out: dict) -> str:
    L = [f"PROPS MARKET CHECK - props_sog {out['model_version']} "
         f"(variant {out.get('variant')}, M = {out['model']}), season {out['season']} out-of-fold "
         f"validation fold (n={out['fold']['n']}, trained on "
         f"{out['fold']['n_train']} earlier rows, NLL {_f(out['fold']['nll_M'], 5)}, "
         f"alpha {_f(out['fold']['alpha_M'])})", "",
         "Price rows (player_shots_on_goal):"]
    for b, r in out["rows"].items():
        pc = out["prices"][b]
        um = out["unmatched"].get(b, {})
        L.append(f"  {b}: {r['rows']} rows, {r['games']} games, {r['first']} .. "
                 f"{r['last']}; pre-game pairs: current {pc['current']}, opening "
                 f"{pc['open']} (current ignored, updated after puck drop: "
                 f"{pc['current_ignored_after_puck_drop']}); no two-sided "
                 f"pre-game pair {pc['no_two_sided_pregame_pair']}")
        L.append(f"     matched {um.get('matched', 0)}; unmatched: no player id "
                 f"{um.get('no_player_id', 0)}, did not play "
                 f"{um.get('did_not_play', 0)}, not eligible (<5 prior) "
                 f"{um.get('not_eligible', 0)}, other {um.get('other', 0)}")
    L.append(f"  pushes dropped: {out['pushes_dropped']}")
    L += ["", "PRIMARY (log loss model - log loss no-vig market, over side; "
          "SE clustered by game; beats market iff mean + 1.96 SE < 0 and n >= 300):"]
    for b, r in out["results"].items():
        p = r["primary"]
        L.append(f"  {b}: n {p['n']} props / {p['games']} games; over rate "
                 f"{_f(p['over_rate'], 3)}, mean P model {_f(p['mean_p_model'], 3)}, "
                 f"market {_f(p['mean_p_mkt'], 3)}")
        L.append(f"    log loss model {_f(p['logloss_model'], 5)} market "
                 f"{_f(p['logloss_mkt'], 5)}; diff {p['diff']:+.5f} (SE "
                 f"{_f(p['se_game'], 5)}; mean + 1.96 SE {p['upper']:+.5f}) -> "
                 f"{'BEATS THE MARKET' if p['beats_market'] else 'does NOT beat the market'}")
        L.append(f"    Brier model {_f(p['brier_model'], 5)} market "
                 f"{_f(p['brier_mkt'], 5)}; ECE (10 bins) model "
                 f"{_f(p['ece_model'])} market {_f(p['ece_mkt'])}")
    mc = out.get("market_check")
    if mc:
        L.append(f"  v3 PASS RULE (pooled beats AND every book with n >= "
                 f"{GATE_MIN_N} negative): pooled beats {mc['pooled_beats']}, "
                 f"books negative {mc['books_negative']} -> "
                 f"{'PASSED' if mc['passed'] else 'NOT PASSED'}")
    L += ["", "SECONDARY (information only): flat 1-unit bets at the quoted "
          "prices, game-clustered bootstrap 95% (2,000, seed 7):"]
    for b, r in out["results"].items():
        L.append(f"  {b}:")
        L += [_bet_line(x) for x in r["betting"]]
        for lg, x in r["by_line"].items():
            L.append(_bet_line(x, f"line {lg:>4} "))
        for pos, x in r["by_position"].items():
            L.append(_bet_line(x, f"pos {pos}      "))
    md = out.get("mean_diagnostic") or {}
    if md:
        L += ["", "MARKET-MEAN DIAGNOSTIC (information only; level vs shape):"]
        blocks = [(k, v) for k, v in md.items() if k != "by_line"]
        blocks += [(f"line {k}", v) for k, v in md.get("by_line", {}).items()]
        for k, b in blocks:
            L.append(f"  {k}: n {b['n']}; mean SOG {_f(b['mean_sog'], 3)}, "
                     f"mu model {_f(b['mean_mu_model'], 3)} (bias "
                     f"{b['bias_model']:+.3f} SE {_f(b['se_bias_model'], 3)}), "
                     f"mu mkt {_f(b['mean_mu_mkt'], 3)} (bias "
                     f"{b['bias_mkt']:+.3f} SE {_f(b['se_bias_mkt'], 3)})")
            L.append(f"    count log loss model - mkt {b['count_ll_diff']:+.5f} "
                     f"(SE {_f(b['se_count_ll_diff'], 5)}); beta "
                     f"{_f(b['beta'], 3)} (SE {_f(b['se_beta'], 3)}), "
                     f"intercept {b['beta_intercept']:+.3f}")
            L.append(f"    gap {b['gap']:+.5f}; level fix {b['gap_level_fix']:+.5f} "
                     f"(closes {_f(b['closed_level'], 2)}); shape fix "
                     f"{b['gap_shape_fix']:+.5f} (alpha* {_f(b['alpha_star'], 3)} "
                     f"vs model {_f(b['mean_alpha_model'], 3)}; closes "
                     f"{_f(b['closed_shape'], 2)}) -> {b['reading']}")
    L += ["", "ALTERNATE 'N+' milestones (over only, no no-vig possible; "
          "information only):"]
    for b, a in out["alternate"].items():
        if not a.get("n"):
            L.append(f"  {b}: {a['n_rows']} rows, none matched")
            continue
        L.append(f"  {b}: {a['n_rows']} rows, {a['n']} matched over {a['games']} "
                 f"games; over rate {_f(a['over_rate'], 3)}, mean P model "
                 f"{_f(a['mean_p_model'], 3)}, mean implied (with vig) "
                 f"{_f(a['mean_implied_with_vig'], 3)}, Brier model "
                 f"{_f(a['brier_model'], 5)}")
        L += [_bet_line(x) for x in a["betting"]]
    return "\n".join(L)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m models.props_market_check",
        description="Pre-registered check of the shots-on-goal props model "
                    "against stored prop prices (read only; never "
                    "registers).")
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument("--variant", default=None,
                        help="props_sog v3 variant (default: the adopted one)")
    args = parser.parse_args(argv)
    out = run_market_check(season=args.season, variant=args.variant)
    print(format_report(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
