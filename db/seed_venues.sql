-- db/seed_venues.sql
-- One-time seed: home-venue coordinates + IANA timezones for travel/schedule
-- features (Phase 2, Task 4). Idempotent — safe to re-run any time.
--
-- Simplifications (documented):
--   * One venue per franchise, the current arena. Recent moves are within
--     the same metro (NYI Nassau Coliseum -> UBS Arena ~25 km; ARI Gila
--     River -> Mullett ~20 km) — negligible against inter-city distances.
--   * ARI (Arizona Coyotes, folded into UTA in 2024) appears in historical
--     games but not in the current-standings API, so it is inserted here as
--     an inactive franchise.
--   * ingestion/nhl_api.py:ingest_teams() upserts only name/conference/
--     division, so these venue columns survive re-ingestion.

INSERT INTO raw.teams (team_abbrev, team_name, venue_name, venue_city, venue_timezone, latitude, longitude, active) VALUES
    ('ANA', 'Anaheim Ducks',         'Honda Center',             'Anaheim',        'America/Los_Angeles', 33.80780, -117.87660, TRUE),
    ('ARI', 'Arizona Coyotes',       'Mullett Arena',            'Tempe',          'America/Phoenix',     33.42550, -111.93250, FALSE),
    ('BOS', 'Boston Bruins',         'TD Garden',                'Boston',         'America/New_York',    42.36620,  -71.06210, TRUE),
    ('BUF', 'Buffalo Sabres',        'KeyBank Center',           'Buffalo',        'America/New_York',    42.87500,  -78.87650, TRUE),
    ('CAR', 'Carolina Hurricanes',   'Lenovo Center',            'Raleigh',        'America/New_York',    35.80330,  -78.72190, TRUE),
    ('CBJ', 'Columbus Blue Jackets', 'Nationwide Arena',         'Columbus',       'America/New_York',    39.96950,  -83.00600, TRUE),
    ('CGY', 'Calgary Flames',        'Scotiabank Saddledome',    'Calgary',        'America/Edmonton',    51.03740, -114.05190, TRUE),
    ('CHI', 'Chicago Blackhawks',    'United Center',            'Chicago',        'America/Chicago',     41.88070,  -87.67420, TRUE),
    ('COL', 'Colorado Avalanche',    'Ball Arena',               'Denver',         'America/Denver',      39.74870, -105.00760, TRUE),
    ('DAL', 'Dallas Stars',          'American Airlines Center', 'Dallas',         'America/Chicago',     32.79050,  -96.81030, TRUE),
    ('DET', 'Detroit Red Wings',     'Little Caesars Arena',     'Detroit',        'America/Detroit',     42.34110,  -83.05530, TRUE),
    ('EDM', 'Edmonton Oilers',       'Rogers Place',             'Edmonton',       'America/Edmonton',    53.54690, -113.49730, TRUE),
    ('FLA', 'Florida Panthers',      'Amerant Bank Arena',       'Sunrise',        'America/New_York',    26.15850,  -80.32550, TRUE),
    ('LAK', 'Los Angeles Kings',     'Crypto.com Arena',         'Los Angeles',    'America/Los_Angeles', 34.04300, -118.26730, TRUE),
    ('MIN', 'Minnesota Wild',        'Grand Casino Arena',       'Saint Paul',     'America/Chicago',     44.94480,  -93.10110, TRUE),
    ('MTL', 'Montréal Canadiens',    'Bell Centre',              'Montréal',       'America/Toronto',     45.49610,  -73.56930, TRUE),
    ('NJD', 'New Jersey Devils',     'Prudential Center',        'Newark',         'America/New_York',    40.73360,  -74.17110, TRUE),
    ('NSH', 'Nashville Predators',   'Bridgestone Arena',        'Nashville',      'America/Chicago',     36.15930,  -86.77850, TRUE),
    ('NYI', 'New York Islanders',    'UBS Arena',                'Elmont',         'America/New_York',    40.71200,  -73.72690, TRUE),
    ('NYR', 'New York Rangers',      'Madison Square Garden',    'New York',       'America/New_York',    40.75050,  -73.99340, TRUE),
    ('OTT', 'Ottawa Senators',       'Canadian Tire Centre',     'Ottawa',         'America/Toronto',     45.29690,  -75.92730, TRUE),
    ('PHI', 'Philadelphia Flyers',   'Wells Fargo Center',       'Philadelphia',   'America/New_York',    39.90120,  -75.17200, TRUE),
    ('PIT', 'Pittsburgh Penguins',   'PPG Paints Arena',         'Pittsburgh',     'America/New_York',    40.43950,  -79.98960, TRUE),
    ('SEA', 'Seattle Kraken',        'Climate Pledge Arena',     'Seattle',        'America/Los_Angeles', 47.62210, -122.35400, TRUE),
    ('SJS', 'San Jose Sharks',       'SAP Center',               'San Jose',       'America/Los_Angeles', 37.33280, -121.90120, TRUE),
    ('STL', 'St. Louis Blues',       'Enterprise Center',        'St. Louis',      'America/Chicago',     38.62680,  -90.20270, TRUE),
    ('TBL', 'Tampa Bay Lightning',   'Amalie Arena',             'Tampa',          'America/New_York',    27.94270,  -82.45180, TRUE),
    ('TOR', 'Toronto Maple Leafs',   'Scotiabank Arena',         'Toronto',        'America/Toronto',     43.64350,  -79.37910, TRUE),
    ('UTA', 'Utah Mammoth',          'Delta Center',             'Salt Lake City', 'America/Denver',      40.76830, -111.90110, TRUE),
    ('VAN', 'Vancouver Canucks',     'Rogers Arena',             'Vancouver',      'America/Vancouver',   49.27780, -123.10890, TRUE),
    ('VGK', 'Vegas Golden Knights',  'T-Mobile Arena',           'Las Vegas',      'America/Los_Angeles', 36.10290, -115.17840, TRUE),
    ('WPG', 'Winnipeg Jets',         'Canada Life Centre',       'Winnipeg',       'America/Winnipeg',    49.89270,  -97.14360, TRUE),
    ('WSH', 'Washington Capitals',   'Capital One Arena',        'Washington',     'America/New_York',    38.89810,  -77.02090, TRUE)
ON CONFLICT (team_abbrev) DO UPDATE SET
    venue_name     = EXCLUDED.venue_name,
    venue_city     = EXCLUDED.venue_city,
    venue_timezone = EXCLUDED.venue_timezone,
    latitude       = EXCLUDED.latitude,
    longitude      = EXCLUDED.longitude,
    active         = EXCLUDED.active;
