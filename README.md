# NHL Sports Betting Predictive System

A professional-grade NHL sports betting system built on calibrated probability models, fractional Kelly staking, and closing-line value tracking.

## Quick Start

### Prerequisites
- Docker & Docker Compose (for PostgreSQL)
- Python 3.11+
- (Optional) The Odds API key from [the-odds-api.com](https://the-odds-api.com)

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[ml,dashboard,dev]"
cp .env.example .env
```

### 2. Start Database

```bash
docker compose up -d
```

### 3. Verify & Backfill

```bash
python pipeline.py setup
python pipeline.py backfill
python pipeline.py status
```

### 4. Daily Operations

```bash
python pipeline.py daily
```

## Project Structure

```
nhl-betting-system/
├── config/          # Settings, database connection
├── db/              # SQL schema
├── ingestion/       # nhl_api.py, moneypuck.py, odds_api.py
├── features/        # Feature store (Phase 2)
├── models/          # ML models (Phase 3)
├── betting/         # Betting engine (Phase 3)
├── dashboard/       # Streamlit UI (Phase 3)
├── tests/           # Test suite
├── pipeline.py       # Master orchestrator
├── PROJECT_CONTEXT.md
├── PHASE2_PLAN.md
└── docs/             # Design documents (File1-3, status report)
```

See `PROJECT_CONTEXT.md` for locked architecture decisions and `PHASE2_PLAN.md` for the next build phase.
