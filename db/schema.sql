-- ============================================================
-- NHL Sports Betting Predictive System — Database Schema
-- Version: 1.0.0
-- Generated from File 3 Design Proposal
-- ============================================================

-- ── Schemas ──
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS betting;

-- ============================================================
-- RAW SCHEMA (ingested data, append-only)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw.games (
    game_id         BIGINT PRIMARY KEY,
    season          INTEGER NOT NULL,          -- e.g. 20252026
    game_type       SMALLINT NOT NULL,         -- 1=preseason, 2=regular, 3=playoff
    date            DATE NOT NULL,
    home_team       VARCHAR(3) NOT NULL,
    away_team       VARCHAR(3) NOT NULL,
    home_score      SMALLINT,
    away_score      SMALLINT,
    period_scores   JSONB,                     -- {"1": [1,0], "2": [2,1], ...}
    is_ot           BOOLEAN DEFAULT FALSE,
    is_so           BOOLEAN DEFAULT FALSE,
    venue           VARCHAR(100),
    game_state      VARCHAR(20) DEFAULT 'SCHEDULED',  -- SCHEDULED, LIVE, FINAL
    home_coach      VARCHAR(80),
    away_coach      VARCHAR(80),
    attendance      INTEGER,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_games_date ON raw.games(date);
CREATE INDEX IF NOT EXISTS idx_games_season ON raw.games(season);
CREATE INDEX IF NOT EXISTS idx_games_teams ON raw.games(home_team, away_team);

CREATE TABLE IF NOT EXISTS raw.teams (
    team_abbrev     VARCHAR(3) PRIMARY KEY,
    team_name       VARCHAR(60) NOT NULL,
    conference      VARCHAR(10),
    division        VARCHAR(20),
    venue_name      VARCHAR(80),
    venue_city      VARCHAR(40),
    venue_timezone  VARCHAR(40),
    latitude        NUMERIC(8,5),
    longitude       NUMERIC(8,5),
    active          BOOLEAN DEFAULT TRUE,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.players (
    player_id       INTEGER PRIMARY KEY,
    full_name       VARCHAR(80) NOT NULL,
    position        VARCHAR(2),                -- C, L, R, D, G
    shoots_catches  VARCHAR(1),                -- L, R
    birth_date      DATE,
    height_cm       SMALLINT,
    weight_kg       SMALLINT,
    nationality     VARCHAR(3),
    active          BOOLEAN DEFAULT TRUE,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.rosters (
    player_id       INTEGER NOT NULL REFERENCES raw.players(player_id),
    season          INTEGER NOT NULL,
    team            VARCHAR(3) NOT NULL,
    position        VARCHAR(2),
    jersey_number   SMALLINT,
    PRIMARY KEY (player_id, season, team)
);

CREATE TABLE IF NOT EXISTS raw.shots (
    shot_id         BIGSERIAL PRIMARY KEY,
    game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
    season          INTEGER NOT NULL,
    period          SMALLINT NOT NULL,
    time_elapsed    INTEGER NOT NULL,          -- seconds elapsed in game (MoneyPuck `time`)
    team            VARCHAR(3) NOT NULL,
    shooter_id      INTEGER,
    goalie_id       INTEGER,
    x               NUMERIC(6,2),
    y               NUMERIC(6,2),
    shot_type       VARCHAR(20),               -- Wrist, Slap, Snap, Backhand, Tip, Deflection, Wrap
    event_type      VARCHAR(10) NOT NULL,       -- SHOT, GOAL, MISS, BLOCK
    is_goal         BOOLEAN NOT NULL DEFAULT FALSE,
    xg_moneypuck    NUMERIC(6,4),              -- MoneyPuck pre-computed xG
    strength        VARCHAR(5),                -- 5v5, 5v4, 4v5, etc.
    score_state     SMALLINT,                  -- home_score - away_score at time of shot
    is_rebound      BOOLEAN DEFAULT FALSE,
    is_rush         BOOLEAN DEFAULT FALSE,
    shot_distance   NUMERIC(6,2),
    shot_angle      NUMERIC(6,2),
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_shots_game ON raw.shots(game_id);
CREATE INDEX IF NOT EXISTS idx_shots_season ON raw.shots(season);
CREATE INDEX IF NOT EXISTS idx_shots_goalie ON raw.shots(goalie_id);
CREATE INDEX IF NOT EXISTS idx_shots_shooter ON raw.shots(shooter_id);

-- Team-level game stats from the gamecenter right-rail endpoint
-- (PP conversions, faceoffs, hits, blocks — not available in boxscore
-- playerByGameStats; needed for pp_pct/pk_pct/fow_pct rolling features)
CREATE TABLE IF NOT EXISTS raw.team_games (
    game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
    team            VARCHAR(3) NOT NULL,
    is_home         BOOLEAN NOT NULL,
    sog             SMALLINT,
    faceoff_wins    SMALLINT,
    faceoff_total   SMALLINT,
    pp_goals        SMALLINT,
    pp_opps         SMALLINT,
    pim             SMALLINT,
    hits            SMALLINT,
    blocked_shots   SMALLINT,
    giveaways       SMALLINT,
    takeaways       SMALLINT,
    PRIMARY KEY (game_id, team)
);
CREATE INDEX IF NOT EXISTS idx_team_games_team ON raw.team_games(team);

CREATE TABLE IF NOT EXISTS raw.skater_games (
    player_id       INTEGER NOT NULL,
    game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
    team            VARCHAR(3) NOT NULL,
    position        VARCHAR(2),
    toi_seconds     INTEGER,
    goals           SMALLINT DEFAULT 0,
    assists         SMALLINT DEFAULT 0,
    points          SMALLINT DEFAULT 0,
    shots           SMALLINT DEFAULT 0,
    hits            SMALLINT DEFAULT 0,
    blocks          SMALLINT DEFAULT 0,
    pim             SMALLINT DEFAULT 0,
    plus_minus      SMALLINT DEFAULT 0,
    fow             SMALLINT DEFAULT 0,        -- faceoffs won
    fol             SMALLINT DEFAULT 0,        -- faceoffs lost
    cf              SMALLINT,                  -- corsi for (5v5)
    ca              SMALLINT,                  -- corsi against (5v5)
    xgf             NUMERIC(5,2),
    xga             NUMERIC(5,2),
    pp_toi_seconds  INTEGER DEFAULT 0,
    sh_toi_seconds  INTEGER DEFAULT 0,
    pp_goals        SMALLINT DEFAULT 0,
    pp_assists      SMALLINT DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_skater_games_game ON raw.skater_games(game_id);
CREATE INDEX IF NOT EXISTS idx_skater_games_team ON raw.skater_games(team);

CREATE TABLE IF NOT EXISTS raw.goalie_games (
    player_id       INTEGER NOT NULL,
    game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
    team            VARCHAR(3) NOT NULL,
    decision        VARCHAR(3),                -- W, L, OTL, NULL (relief)
    is_starter      BOOLEAN DEFAULT FALSE,
    shots_against   SMALLINT DEFAULT 0,
    saves           SMALLINT DEFAULT 0,
    goals_against   SMALLINT DEFAULT 0,
    sv_pct          NUMERIC(5,4),
    toi_seconds     INTEGER DEFAULT 0,
    xga             NUMERIC(5,2),              -- expected goals against
    gsax            NUMERIC(5,2),              -- goals saved above expected
    even_shots      SMALLINT DEFAULT 0,
    even_saves      SMALLINT DEFAULT 0,
    pp_shots        SMALLINT DEFAULT 0,
    pp_saves        SMALLINT DEFAULT 0,
    sh_shots        SMALLINT DEFAULT 0,
    sh_saves        SMALLINT DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_goalie_games_game ON raw.goalie_games(game_id);

CREATE TABLE IF NOT EXISTS raw.odds_snapshots (
    snapshot_id     BIGSERIAL PRIMARY KEY,
    game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
    captured_at     TIMESTAMP NOT NULL,
    book_name       VARCHAR(40) NOT NULL,
    market_type     VARCHAR(10) NOT NULL,      -- ml (moneyline), pl (puckline), total
    home_price      INTEGER,                   -- American odds e.g. -150, +130
    away_price      INTEGER,
    over_price      INTEGER,
    under_price     INTEGER,
    line            NUMERIC(4,1),              -- spread (1.5) or total line (6.0)
    is_closing      BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_odds_game ON raw.odds_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_odds_time ON raw.odds_snapshots(captured_at);

-- Historical reference odds (one row per game) backfilled from ESPN's public
-- summary API. Single book, near-closing line — good enough for a market
-- feature and strategy backtests; NOT a substitute for our own multi-book
-- time series in raw.odds_snapshots.
CREATE TABLE IF NOT EXISTS raw.historical_odds (
    game_id         BIGINT PRIMARY KEY REFERENCES raw.games(game_id),
    provider        VARCHAR(40),               -- e.g. DraftKings (varies by era)
    home_ml         INTEGER,                   -- American odds
    away_ml         INTEGER,
    spread          NUMERIC(4,1),              -- home puck line
    over_under      NUMERIC(4,1),
    details         VARCHAR(40),               -- ESPN display string, e.g. "MTL -125"
    fetched_at      TIMESTAMP NOT NULL DEFAULT now()
);

-- Confirmed starting goalies scraped from Daily Faceoff (Phase 4).
-- One row per (game_date, team), upserted as confirmations roll in.
CREATE TABLE IF NOT EXISTS raw.starting_goalies (
    game_date       DATE NOT NULL,
    team            VARCHAR(3) NOT NULL,
    goalie_name     VARCHAR(80) NOT NULL,
    goalie_id       INTEGER,                   -- resolved raw.players id (NULL if unmatched)
    confirmation    VARCHAR(20),               -- Confirmed / Likely / ... / NULL
    source          VARCHAR(20) NOT NULL DEFAULT 'dailyfaceoff',
    fetched_at      TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (game_date, team)
);

CREATE TABLE IF NOT EXISTS raw.shifts (
    shift_id        BIGSERIAL PRIMARY KEY,
    game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
    player_id       INTEGER NOT NULL,
    period          SMALLINT NOT NULL,
    start_time      INTEGER NOT NULL,          -- seconds into period
    end_time        INTEGER,
    duration        INTEGER,
    team            VARCHAR(3)
);
CREATE INDEX IF NOT EXISTS idx_shifts_game ON raw.shifts(game_id);

-- ============================================================
-- FEATURES SCHEMA (computed, point-in-time correct)
-- ============================================================

CREATE TABLE IF NOT EXISTS features.team_rolling (
    game_id         BIGINT NOT NULL,
    team            VARCHAR(3) NOT NULL,
    window_size     SMALLINT NOT NULL,         -- 5, 10, 20, 40, 82
    gf_per60        NUMERIC(6,3),
    ga_per60        NUMERIC(6,3),
    xgf_pct         NUMERIC(5,3),
    cf_pct          NUMERIC(5,3),
    ff_pct          NUMERIC(5,3),
    sh_pct          NUMERIC(5,3),
    sv_pct          NUMERIC(5,4),
    pdo             NUMERIC(6,3),
    pp_pct          NUMERIC(5,3),
    pk_pct          NUMERIC(5,3),
    pp_xgf_per60    NUMERIC(6,3),
    pk_xga_per60    NUMERIC(6,3),
    fow_pct         NUMERIC(5,3),
    pim_per60       NUMERIC(5,2),
    games_played    SMALLINT NOT NULL,
    computed_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (game_id, team, window_size)
);

CREATE TABLE IF NOT EXISTS features.goalie_rolling (
    game_id         BIGINT NOT NULL,
    goalie_id       INTEGER NOT NULL,
    window_size     SMALLINT NOT NULL,         -- last N starts
    sv_pct          NUMERIC(5,4),
    gsax_per60      NUMERIC(6,3),
    hd_sv_pct       NUMERIC(5,4),
    fenwick_sv_pct  NUMERIC(5,4),
    xga_per60       NUMERIC(6,3),
    starts_in_window SMALLINT NOT NULL,
    credibility_z   NUMERIC(4,3),              -- Buhlmann Z = n/(n+k)
    shrunk_sv_pct   NUMERIC(5,4),              -- Z*goalie + (1-Z)*league
    shrunk_gsax     NUMERIC(6,3),
    computed_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (game_id, goalie_id, window_size)
);

CREATE TABLE IF NOT EXISTS features.matchup (
    game_id             BIGINT PRIMARY KEY,
    home_team           VARCHAR(3) NOT NULL,
    away_team           VARCHAR(3) NOT NULL,
    home_rest_days      SMALLINT,
    away_rest_days      SMALLINT,
    home_b2b            BOOLEAN DEFAULT FALSE,
    away_b2b            BOOLEAN DEFAULT FALSE,
    home_travel_km      NUMERIC(7,1),
    away_travel_km      NUMERIC(7,1),
    home_tz_shift       SMALLINT DEFAULT 0,     -- hours of timezone change
    away_tz_shift       SMALLINT DEFAULT 0,
    home_elo            NUMERIC(7,1),
    away_elo            NUMERIC(7,1),
    home_starter_id     INTEGER,
    away_starter_id     INTEGER,
    season_stage        VARCHAR(5),             -- EARLY, MID, LATE
    home_game_num       SMALLINT,               -- game # within season
    away_game_num       SMALLINT,
    computed_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS features.game_vector (
    game_id             BIGINT PRIMARY KEY,
    season              INTEGER NOT NULL,
    date                DATE NOT NULL,
    feature_names       TEXT[],                 -- ordered list of feature names
    feature_vector      DOUBLE PRECISION[],     -- ordered values (home - away differentials)
    home_win            BOOLEAN,                -- label (NULL if game not yet played)
    home_goals          SMALLINT,
    away_goals          SMALLINT,
    computed_at         TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gv_season ON features.game_vector(season);

-- ============================================================
-- MODELS SCHEMA (predictions)
-- ============================================================

CREATE TABLE IF NOT EXISTS models.model_registry (
    model_id            SERIAL PRIMARY KEY,
    model_name          VARCHAR(60) NOT NULL,
    version             VARCHAR(20) NOT NULL,
    model_type          VARCHAR(30),            -- lgbm_moneyline, xgb_xg, pmf_totals, etc.
    trained_through     DATE,
    feature_set_hash    VARCHAR(64),
    cv_log_loss         NUMERIC(6,4),
    cv_brier            NUMERIC(6,4),
    cv_auc              NUMERIC(6,4),
    cv_accuracy         NUMERIC(5,3),
    artifact_path       VARCHAR(200),
    is_active           BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_name, version)
);

CREATE TABLE IF NOT EXISTS models.predictions (
    prediction_id       BIGSERIAL PRIMARY KEY,
    game_id             BIGINT NOT NULL REFERENCES raw.games(game_id),
    model_id            INTEGER NOT NULL REFERENCES models.model_registry(model_id),
    market_type         VARCHAR(10) NOT NULL,   -- ml, pl, total, goalie_prop
    home_win_prob       NUMERIC(5,4),
    away_win_prob       NUMERIC(5,4),
    home_spread_prob    NUMERIC(5,4),           -- P(home wins by > spread)
    total_over_prob     NUMERIC(5,4),
    total_line          NUMERIC(4,1),
    home_goals_pmf      DOUBLE PRECISION[],     -- P(home_goals=k) for k=0..10
    away_goals_pmf      DOUBLE PRECISION[],
    goalie_props        JSONB,                  -- {"saves_ou": 28.5, "over_prob": 0.55, ...}
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, model_id, market_type)
);

-- ============================================================
-- BETTING SCHEMA (strategy + tracking)
-- ============================================================

CREATE TABLE IF NOT EXISTS betting.recommendations (
    rec_id              BIGSERIAL PRIMARY KEY,
    game_id             BIGINT NOT NULL REFERENCES raw.games(game_id),
    prediction_id       BIGINT REFERENCES models.predictions(prediction_id),
    market_type         VARCHAR(10) NOT NULL,
    side                VARCHAR(20) NOT NULL,   -- HOME, AWAY, OVER, UNDER, prop_name
    model_prob          NUMERIC(5,4) NOT NULL,
    best_book           VARCHAR(40),
    best_price          INTEGER,                -- American odds
    implied_prob_novig  NUMERIC(5,4),
    edge_pct            NUMERIC(5,3),
    kelly_fraction      NUMERIC(5,4),
    recommended_stake   NUMERIC(8,2),
    status              VARCHAR(10) DEFAULT 'PENDING',  -- PENDING, APPROVED, PLACED, SKIPPED
    created_at          TIMESTAMP DEFAULT NOW(),
    decided_at          TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rec_game ON betting.recommendations(game_id);
CREATE INDEX IF NOT EXISTS idx_rec_status ON betting.recommendations(status);

CREATE TABLE IF NOT EXISTS betting.placed_bets (
    bet_id              BIGSERIAL PRIMARY KEY,
    rec_id              BIGINT NOT NULL REFERENCES betting.recommendations(rec_id),
    book_name           VARCHAR(40) NOT NULL,
    placed_price        INTEGER NOT NULL,
    stake_amount        NUMERIC(8,2) NOT NULL,
    placed_at           TIMESTAMP NOT NULL,
    result              VARCHAR(5),             -- WIN, LOSS, PUSH, VOID
    pnl                 NUMERIC(10,2),
    closing_line        INTEGER,
    clv                 NUMERIC(5,3),           -- implied(closing) - implied(placed)
    settled_at          TIMESTAMP,
    is_paper            BOOLEAN NOT NULL DEFAULT TRUE   -- paper trail until a human records a real bet
);

CREATE TABLE IF NOT EXISTS betting.bankroll_log (
    log_id              SERIAL PRIMARY KEY,
    date                DATE NOT NULL UNIQUE,
    opening_balance     NUMERIC(10,2) NOT NULL,
    deposits            NUMERIC(10,2) DEFAULT 0,
    withdrawals         NUMERIC(10,2) DEFAULT 0,
    gross_pnl           NUMERIC(10,2) DEFAULT 0,
    closing_balance     NUMERIC(10,2) NOT NULL,
    total_bets          INTEGER DEFAULT 0,
    wins                INTEGER DEFAULT 0,
    losses              INTEGER DEFAULT 0,
    pushes              INTEGER DEFAULT 0,
    roi_pct             NUMERIC(6,3),
    clv_avg             NUMERIC(5,3)
);

-- ============================================================
-- Utility: Updated-at trigger
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_games_updated BEFORE UPDATE ON raw.games
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_teams_updated BEFORE UPDATE ON raw.teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_players_updated BEFORE UPDATE ON raw.players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
