import pandas as pd

df = pd.read_csv('results.csv')
print(df.shape)
print(df.head())

# How many unique teams?
print(df['home_team'].nunique())

# What tournaments are in here?
print(df['tournament'].value_counts().head(10))

# Any missing scores?
print(df[['home_score', 'away_score']].isnull().sum())

# Drop rows with missing scores
df = df.dropna(subset=['home_score', 'away_score'])

# Convert date to datetime so we can sort chronologically
df['date'] = pd.to_datetime(df['date'])

# Sort by date — critical for Elo, which updates match by match
df = df.sort_values('date').reset_index(drop=True)

print(f"Matches remaining: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

from collections import defaultdict

# All teams start at 1500
elo_ratings = defaultdict(lambda: 1000)

K_BASE = 30 # How many points can shift per match

def get_k(tournament):
    """Weight matches by importance"""
    if 'FIFA World Cup' in tournament and 'qualification' not in tournament:
        return K_BASE * 2
    elif tournament in ['UEFA Euro', 'Copa América', 'African Cup of Nations']:
        return K_BASE * 1.5
    elif 'qualification' in tournament:
        return K_BASE * 1.2
    else:
        return K_BASE * 0.8

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(home_team, away_team, home_score, away_score, tournament, neutral):
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]

    # Adjust for home advantage (only tiny bit)
    home_advantage = 0 if neutral else 20

    expected_home = expected_score(home_elo + home_advantage, away_elo)
    expected_away = 1 - expected_home

    # Actual result from team perspective
    if home_score > away_score:
        actual_home = 1
        actual_away = 0
    elif home_score < away_score:
        actual_home = 0
        actual_away = 1
    else:
        actual_home = 0.5
        actual_away = 0.5

    k = get_k(tournament)
    elo_ratings[home_team] += k * (actual_home - expected_home)
    elo_ratings[away_team] += k * (actual_away - expected_away)

# Run through every match in chronological order
for _, row in df.iterrows():
    update_elo(row['home_team'], row['away_team'],
               row['home_score'], row['away_score'],
               row['tournament'], row['neutral'])

# See current top 20 teams
top_teams = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:20]
for team, rating in top_teams:
    print(f"{team:<25} {rating:.0f}")

# Recompute Elo but starting from 2000 only
elo_2022 = defaultdict(lambda: 1000)

for _, row in df[df['date'].dt.year >= 2022].iterrows():
    home, away = row['home_team'], row['away_team']
    # same update logic, just on a subset
    home_elo = elo_2022[home]
    away_elo = elo_2022[away]
    home_advantage = 0 if row['neutral'] else 50
    exp_home = expected_score(home_elo + home_advantage, away_elo)
    exp_away = 1 - exp_home
    if row['home_score'] > row['away_score']:
        act_home, act_away = 1, 0
    elif row['home_score'] < row['away_score']:
        act_home, act_away = 0, 1
    else:
        act_home, act_away = 0.5, 0.5
    k = get_k(row['tournament'])
    elo_2022[home] += k * (act_home - exp_home)
    elo_2022[away] += k * (act_away - exp_away)

# Compare top teams between the two systems
teams = ['Brazil', 'Argentina', 'France', 'England', 'Germany', 'Spain']
print(f"{'Team':<20} {'Elo (1872)':<15} {'Elo (2022)':<15} {'Difference'}")
print("-" * 60)
for team in teams:
    r1 = elo_ratings[team]
    r2 = elo_2022[team]
    print(f"{team:<20} {r1:<15.0f} {r2:<15.0f} {r1-r2:+.0f}")

from collections import defaultdict, deque

# Reset everything so we start clean
elo_ratings = defaultdict(lambda: 1000)

# For each team, keep a rolling window of their last 10 matches.
# We store (points, goal_difference) per match.
#   points: 3 win / 1 draw / 0 loss   (from that team's perspective)
recent = defaultdict(lambda: deque(maxlen=10))

def form_stats(team):
    """Average points-per-game and goal-difference over last 10 matches."""
    games = recent[team]
    if not games:
        return 1.0, 0.0           # neutral defaults for teams with no history
    pts = sum(g[0] for g in games) / len(games)
    gd  = sum(g[1] for g in games) / len(games)
    return pts, gd

# ---- Squad strength (FIFA ratings): point-in-time player signal ----
import numpy as np

_ss = pd.read_csv('squad_strength.csv')
# Rename FIFA country names to match our results.csv naming
_FIFA_NAME_MAP = {
    'Korea Republic': 'South Korea',
    'Curacao': 'Curaçao',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cape Verde Islands': 'Cape Verde',
    'Congo DR': 'DR Congo',
}
_ss['nationality_name'] = _ss['nationality_name'].replace(_FIFA_NAME_MAP)

# Lookup: (fifa_version, team) -> squad_strength
squad_lookup = {(r.fifa_version, r.nationality_name): r.squad_strength
                for r in _ss.itertuples()}

# Edition release dates, sorted, for point-in-time mapping
_editions = (_ss[['fifa_version', 'release_date']].drop_duplicates()
             .assign(release_date=lambda d: pd.to_datetime(d['release_date']))
             .sort_values('release_date'))
_edition_list = list(zip(_editions['release_date'], _editions['fifa_version']))
_latest_version = _edition_list[-1][1]

def squad_strength(team, date=None):
    """Top-23 avg FIFA rating for `team` from the edition current as of `date`.
       date=None uses the latest edition (for forward 2026 predictions).
       Returns NaN if no edition covers that date / team is unknown —
       XGBoost handles NaN natively, so older matches simply lack this signal."""
    if date is None:
        version = _latest_version
    else:
        version = None
        for rel_date, v in _edition_list:
            if rel_date <= date:
                version = v
            else:
                break
        if version is None:
            return np.nan
    return squad_lookup.get((version, team), np.nan)

rows = []  # we'll collect one dict per match here

for _, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    hs, as_ = row['home_score'], row['away_score']
    neutral = row['neutral']

    # ---- 1. SNAPSHOT (before any update) ----
    home_elo = elo_ratings[home]
    away_elo = elo_ratings[away]
    home_pts, home_gd = form_stats(home)
    away_pts, away_gd = form_stats(away)
    home_squad = squad_strength(home, row['date'])
    away_squad = squad_strength(away, row['date'])

    # ---- 2. LABEL ----
    if hs > as_:
        result = 2          # home win
    elif hs < as_:
        result = 0          # away win
    else:
        result = 1          # draw

    # Only keep modern matches for training (Elo still built from full history)
    if row['date'].year >= 2018:
        rows.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'home_form': home_pts,
            'away_form': away_pts,
            'home_gd_form': home_gd,
            'away_gd_form': away_gd,
            'home_squad': home_squad,
            'away_squad': away_squad,
            'squad_diff': home_squad - away_squad,
            'neutral': int(neutral),
            'result': result,
        })

    # ---- 3. UPDATE Elo (same logic as Stage 2) ----
    home_advantage = 0 if neutral else 50
    exp_home = expected_score(home_elo + home_advantage, away_elo)
    exp_away = 1 - exp_home
    if result == 2:
        act_home, act_away = 1, 0
    elif result == 0:
        act_home, act_away = 0, 1
    else:
        act_home, act_away = 0.5, 0.5
    k = get_k(row['tournament'])
    elo_ratings[home] += k * (act_home - exp_home)
    elo_ratings[away] += k * (act_away - exp_away)

    # ---- 4. UPDATE form windows (after the match) ----
    gd = hs - as_
    home_match_pts = 3 if result == 2 else (1 if result == 1 else 0)
    away_match_pts = 3 if result == 0 else (1 if result == 1 else 0)
    recent[home].append((home_match_pts,  gd))
    recent[away].append((away_match_pts, -gd))

# Turn the collected rows into a DataFrame
features = pd.DataFrame(rows)
print(f"Training rows: {len(features)}")
print(features.head())
print(features['result'].value_counts())

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

# The columns the model learns FROM (features) ...
feature_cols = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_form', 'away_form',
    'home_gd_form', 'away_gd_form',
    'home_squad', 'away_squad', 'squad_diff',
    'neutral',
]

# ... and the column it learns to PREDICT (label)
target_col = 'result'

# ---- Time-based split ----
train = features[features['date'] <  '2022-01-01']
test  = features[features['date'] >= '2022-01-01']

X_train, y_train = train[feature_cols], train[target_col]
X_test,  y_test  = test[feature_cols],  test[target_col]

print(f"Training matches: {len(X_train)}")
print(f"Testing matches:  {len(X_test)}")

# ---- Build the model: train ONCE, then reuse the saved copy ----
# Same model every run (deterministic). Re-training is non-deterministic
# (XGBoost multi-threading), which added meaningless run-to-run noise to idle
# games. Train once, persist, load thereafter. Delete model.pkl to retrain.
import os
import joblib

MODEL_PATH = 'model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"\nLoaded frozen model from {MODEL_PATH}")
else:
    model = XGBClassifier(
        n_estimators=300,      # number of trees
        max_depth=4,           # how deep each tree can go (small = less overfitting)
        learning_rate=0.05,    # how much each tree contributes (smaller = more careful)
        subsample=0.9,         # use 90% of rows per tree (adds robustness)
        objective='multi:softprob',  # output probabilities for 3 classes
        eval_metric='mlogloss',
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained and saved model to {MODEL_PATH}")

# ---- Evaluate ----
preds  = model.predict(X_test)               # most likely outcome per match
probs  = model.predict_proba(X_test)         # probabilities for [away, draw, home]

accuracy = accuracy_score(y_test, preds)
ll       = log_loss(y_test, probs)

print(f"\nAccuracy:  {accuracy:.1%}")
print(f"Log loss:  {ll:.3f}")

# Compare to baseline (always predict home win)
baseline = (y_test == 2).mean()
print(f"Baseline (always home): {baseline:.1%}")

import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print(importance)

HOST_BOOST = 88   # calibrated from 1998+ World Cup host nations (was a guessed 75)

def predict_match(home, away, host_team=None):
    """
    Predict a single match.
    host_team: pass the name of a host nation (USA/Canada/Mexico) if THIS
               match is a genuine home game for them; otherwise leave None
               (World Cup matches are neutral by default).
    """
    # Current strength + form (these are live after the loop ran through 2026)
    home_elo = elo_ratings[home]
    away_elo = elo_ratings[away]
    home_pts, home_gd = form_stats(home)
    away_pts, away_gd = form_stats(away)
    home_squad = squad_strength(home)   # latest FIFA edition
    away_squad = squad_strength(away)

    # Host-nation home advantage: nudge the host's Elo up, mark non-neutral
    neutral = 1
    if host_team == home:
        home_elo += HOST_BOOST   # calibrated from 1998+ World Cup hosts
        neutral = 0
    elif host_team == away:
        away_elo += HOST_BOOST
        neutral = 0

    # Build the single-row feature table in the SAME column order as training
    import pandas as pd
    row = pd.DataFrame([{
        'home_elo': home_elo,
        'away_elo': away_elo,
        'elo_diff': home_elo - away_elo,
        'home_form': home_pts,
        'away_form': away_pts,
        'home_gd_form': home_gd,
        'away_gd_form': away_gd,
        'home_squad': home_squad,
        'away_squad': away_squad,
        'squad_diff': home_squad - away_squad,
        'neutral': neutral,
    }])[feature_cols]

    probs = model.predict_proba(row)[0]   # [away_win, draw, home_win]
    return {
        f'{home} win': probs[2],
        'draw':        probs[1],
        f'{away} win': probs[0],
    }

result = predict_match('Brazil', 'Germany')
for outcome, p in result.items():
    print(f"{outcome:<20} {p:.1%}")

print(f"Brazil Elo:  {elo_ratings['Brazil']:.0f}, form: {form_stats('Brazil')}")
print(f"Germany Elo: {elo_ratings['Germany']:.0f}, form: {form_stats('Germany')}")

# Grab the most recent matches (e.g. last 14 days of data)
recent_matches = features[features['date'] >= '2026-06-01'].copy()

# Predict them all at once
X_recent = recent_matches[feature_cols]
recent_matches['pred'] = model.predict(X_recent)

# Map numeric codes back to readable labels
label = {0: 'away win', 1: 'draw', 2: 'home win'}

correct = 0
for _, r in recent_matches.iterrows():
    got_it = '✅' if r['pred'] == r['result'] else '❌'
    if r['pred'] == r['result']:
        correct += 1
    print(f"{got_it} {r['home_team']:<18} vs {r['away_team']:<18} "
          f"| predicted: {label[r['pred']]:<9} | actual: {label[r['result']]}")

n = len(recent_matches)
print(f"\nGot {correct}/{n} correct ({correct/n:.0%})")

groups = {
    'A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],        # Mexico hosts
    'B': ['Canada', 'Switzerland', 'Bosnia and Herzegovina', 'Qatar'],        # Canada hosts
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],           # USA hosts
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

HOSTS = {'Mexico', 'Canada', 'United States'}

from itertools import combinations

def expected_points(home, away):
    """Return (home_pts, away_pts) as EXPECTED points, weighting by probability.
       3*P(win) + 1*P(draw) — gives smoother standings than forcing a single outcome."""
    host = None
    if home in HOSTS: host = home
    elif away in HOSTS: host = away

    p = predict_match(home, away, host_team=host)
    p_home = p[f'{home} win']
    p_draw = p['draw']
    p_away = p[f'{away} win']

    home_pts = 3 * p_home + 1 * p_draw
    away_pts = 3 * p_away + 1 * p_draw
    return home_pts, away_pts, p_home, p_draw, p_away

# Actual points from 2026 WC group games already played (for a LIVE table:
# played games count their real 3/1/0; unplayed games use expected points).
# Group-stage stats ONLY: restrict to the 72 group-round pairings (12 groups x 6),
# first occurrence per pairing. This keeps Points/GP/GD/xPts frozen once a team has
# played its 3 group games — knockout results never leak into the group standings.
_group_pairs = {frozenset(p) for ts in groups.values() for p in combinations(ts, 2)}
actual_wc_pts = {}   # frozenset(home, away) -> {team: points}  (group games only)
team_pts = defaultdict(int)  # actual points (PRIMARY standings sort)
team_gd = defaultdict(int)   # actual goal difference (1st tiebreaker)
team_gf = defaultdict(int)   # actual goals for (2nd tiebreaker)
for r in df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2026)].itertuples():
    pair = frozenset((r.home_team, r.away_team))
    if pair not in _group_pairs or pair in actual_wc_pts:
        continue             # knockout game (or same-group rematch) — not group stats
    h, a = r.home_team, r.away_team
    hs, as_ = int(r.home_score), int(r.away_score)
    if hs > as_: hp, ap = 3, 0
    elif hs < as_: hp, ap = 0, 3
    else: hp, ap = 1, 1
    actual_wc_pts[pair] = {h: hp, a: ap}
    team_pts[h] += hp;      team_pts[a] += ap
    team_gd[h] += hs - as_; team_gd[a] += as_ - hs
    team_gf[h] += hs;       team_gf[a] += as_

group_standings = {}   # group_name -> ordered list of (team, xPts)
group_matches = {}     # group_name -> list of (home, away, p_home, p_draw, p_away)

for group_name, teams in groups.items():
    print(f"\n===== GROUP {group_name} =====")
    table = {t: 0.0 for t in teams}
    matches = []

    for home, away in combinations(teams, 2):
        h_pts, a_pts, ph, pd_, pa = expected_points(home, away)
        pair = frozenset((home, away))
        if pair in actual_wc_pts:                 # played -> ACTUAL points
            table[home] += actual_wc_pts[pair][home]
            table[away] += actual_wc_pts[pair][away]
        else:                                     # not played -> expected points
            table[home] += h_pts
            table[away] += a_pts
        matches.append((home, away, ph, pd_, pa))
        print(f"  {home:<16} vs {away:<16}  "
              f"{ph:.0%}/{pd_:.0%}/{pa:.0%}")

    group_matches[group_name] = matches

    # Bracket/odds rank by the model's PROJECTED final standings (xPts), with
    # actual GD then goals-for breaking ties (matters once a group is complete).
    # The Groups TABLE is re-sorted by actual points in the dashboard.
    standings = sorted(table.items(),
                       key=lambda kv: (kv[1], team_gd.get(kv[0], 0), team_gf.get(kv[0], 0)),
                       reverse=True)
    group_standings[group_name] = standings
    print(f"  --- Predicted standings ---")
    for rank, (team, pts) in enumerate(standings, 1):
        mark = '  ⬆️ advances' if rank <= 2 else ''
        print(f"  {rank}. {team:<16} {pts:.2f} xPts{mark}")


'''
elo_tmp = defaultdict(lambda: 1000)
wc_host_rows = []

for _, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    h_elo, a_elo = elo_tmp[home], elo_tmp[away]
    hs, as_ = row['home_score'], row['away_score']

    # World Cup match that is NOT neutral = host nation playing at home
    if (row['tournament'] == 'FIFA World Cup'
            and not row['neutral']
            and row['date'].year >= 1998):
        r = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        wc_host_rows.append({
            'host': home, 'year': row['date'].year,
            'home_elo': h_elo, 'away_elo': a_elo, 'home_actual': r
        })

    # standard Elo update (same +50 logic as before)
    home_adv = 0 if row['neutral'] else 50
    exp_home = expected_score(h_elo + home_adv, a_elo)
    ah = 1 if hs > as_ else (0 if hs < as_ else 0.5)
    k = get_k(row['tournament'])
    elo_tmp[home] += k * (ah - exp_home)
    elo_tmp[away] += k * ((1 - ah) - (1 - exp_home))

host = pd.DataFrame(wc_host_rows)
print(f"World Cup host matches found: {len(host)}")
print(host['host'].value_counts(), "\n")

observed = host['home_actual'].mean()
def avg_exp(h):
    return (1 / (1 + 10 ** ((host['away_elo'] - (host['home_elo'] + h)) / 400))).mean()

print(f"Observed avg host score: {observed:.3f}")
print(f"Expected with no boost:  {avg_exp(0):.3f}")

best_h, best_err = 0, 1.0
for h in range(-50, 301):
    e = abs(avg_exp(h) - observed)
    if e < best_err:
        best_err, best_h = e, h

print(f"\n📊 Calibrated WORLD CUP HOST boost: +{best_h} Elo points")'''

''' # All team names that actually exist in the data
 known_teams = set(df['home_team']) | set(df['away_team'])

# Every team you listed across all groups
listed = [t for teams in groups.values() for t in teams]

missing = [t for t in listed if t not in known_teams]

if not missing:
    print("✅ All team names match the dataset.")
else:
    print("❌ These names are NOT in the dataset:")
    for t in missing:
        # Suggest close matches to help you fix them
        suggestions = [k for k in known_teams if t.lower()[:4] in k.lower()]
        print(f"   '{t}'  → did you mean: {suggestions}")'''


# ============================================================
# KNOCKOUT STAGE — Monte Carlo simulation
# ============================================================
import random
from math import exp, factorial
try:
    from scipy.optimize import minimize as _minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _pois(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def _wdl_from_lambdas(lh, la, maxg=10):
    """Independent-Poisson home/draw/away probabilities for given goal rates."""
    hp = [_pois(i, lh) for i in range(maxg + 1)]
    ap = [_pois(j, la) for j in range(maxg + 1)]
    ph = pdr = pa = 0.0
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            p = hp[i] * ap[j]
            if i > j: ph += p
            elif i == j: pdr += p
            else: pa += p
    return ph, pdr, pa


def implied_lambdas(p_home, p_draw, p_away):
    """Invert pre-match odds -> (lambda_home, lambda_away) best reproducing them."""
    def loss(x):
        lh, la = x
        if lh <= 0.02 or la <= 0.02 or lh > 6 or la > 6:
            return 1e9
        ph, pdr, pa = _wdl_from_lambdas(lh, la)
        return (ph - p_home) ** 2 + (pdr - p_draw) ** 2 + (pa - p_away) ** 2
    if _HAVE_SCIPY:
        r = _minimize(loss, [1.3, 1.1], method='Nelder-Mead',
                      options={'xatol': 1e-3, 'fatol': 1e-7})
        lh, la = r.x
    else:                                   # scipy-free fallback: coarse grid
        best, lh, la = 1e9, 1.3, 1.1
        grid = [0.1 + 0.05 * k for k in range(70)]
        for x in grid:
            for y in grid:
                e = loss((x, y))
                if e < best:
                    best, lh, la = e, x, y
    return max(0.05, round(lh, 3)), max(0.05, round(la, 3))


def _pois_sample(lam):
    """Draw a Poisson sample (Knuth) via random() so it respects the global seed."""
    if lam <= 0:
        return 0
    L = exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def build_r32(win, run, third_slot, third_first):
    """Build the Round-of-32 matchups from group positions (official 2026 structure)."""
    def tt(slot):
        g = third_slot.get(slot)
        return third_first[g] if g else None
    return {
        73: (run['A'], run['B']),   74: (win['E'], tt(74)),
        75: (win['F'], run['C']),   76: (win['C'], run['F']),
        77: (win['I'], tt(77)),     78: (run['E'], run['I']),
        79: (win['A'], tt(79)),     80: (win['L'], tt(80)),
        81: (win['D'], tt(81)),     82: (win['G'], tt(82)),
        83: (run['K'], run['L']),   84: (win['H'], run['J']),
        85: (win['B'], tt(85)),     86: (win['J'], run['H']),
        87: (win['K'], tt(87)),     88: (run['D'], run['G']),
    }


# ---- 1. Group positions (deterministic from standings: 1st / 2nd / 3rd) ----
winner = {g: s[0][0] for g, s in group_standings.items()}
runner = {g: s[1][0] for g, s in group_standings.items()}
third  = {g: s[2] for g, s in group_standings.items()}     # (team, pts)
# The 8 best third-place teams advance (points, then actual GD, then goals for).
best_third_groups = [g for g, _ in sorted(
    third.items(),
    key=lambda kv: (kv[1][1], team_gd.get(kv[1][0], 0), team_gf.get(kv[1][0], 0)),
    reverse=True)[:8]]
print(f"\nQualifiers: 32 (12 winners, 12 runners-up, 8 best thirds)")

# ---- 2. Assign the 8 best thirds to their allowed R32 slots (official FIFA structure) ----
# Each third-place slot can only take a third from a fixed set of groups.
THIRD_SLOTS = {74: set('ABCDF'), 77: set('CDFGH'), 79: set('CEFHI'), 80: set('EHIJK'),
               81: set('BEFIJ'), 82: set('AEHIJ'), 85: set('EFGIJ'), 87: set('DEIJL')}
def _assign_thirds(groups_in):
    """Bipartite matching: each qualifying third-group -> a slot whose set allows it."""
    match = {}   # slot -> group
    def aug(g, seen):
        for s, allowed in THIRD_SLOTS.items():
            if g in allowed and s not in seen:
                seen.add(s)
                if s not in match or aug(match[s], seen):
                    match[s] = g
                    return True
        return False
    for g in groups_in:
        aug(g, set())
    return {s: gg for s, gg in match.items()}
_third_slot = _assign_thirds(best_third_groups)
def third_team(slot):
    g = _third_slot.get(slot)
    return third[g][0] if g else None

# ---- 3. Advance probability (draw resolved ~50/50 on penalties) ----
_adv_cache = {}
def advance_prob(a, b):
    if (a, b) not in _adv_cache:
        host = a if a in HOSTS else (b if b in HOSTS else None)
        p = predict_match(a, b, host_team=host)
        _adv_cache[(a, b)] = p[f'{a} win'] + 0.5 * p['draw']
    return _adv_cache[(a, b)]
def _play(a, b):
    return a if random.random() < advance_prob(a, b) else b

# ---- 4. The REAL 2026 bracket: Round-of-32 matchups by group position ----
# Deterministic (projected) field — used for the bracket VISUAL + confirmed flags.
R32 = build_r32(winner, runner, _third_slot, {g: third[g][0] for g in third})
# Fixed tree — each match's two inputs are the WINNERS of earlier matches.
# Label = the stage a match WINNER reaches.
ROUNDS = [
    ('QF',       {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
                  93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}),
    ('SF',       {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}),
    ('Final',    {101: (97, 98), 102: (99, 100)}),
    ('Champion', {104: (101, 102)}),
]

# ---- 5. Condition on REAL knockout results: a played KO game is a fact, not
#         re-simulated. Eliminated teams fall to 0; everyone else recomputes. ----
_ko_lookup = {}
for _r in df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2026)].itertuples():
    _ko_lookup[frozenset((_r.home_team, _r.away_team))] = (
        _r.home_team, _r.away_team, _r.home_score, _r.away_score)
def _ko_win(a, b):
    if not a or not b:
        return None
    g = _ko_lookup.get(frozenset((a, b)))
    if not g or g[2] == g[3]:        # not played, or draw (penalties unknown)
        return None
    return g[0] if g[2] > g[3] else g[1]
FORCED = {}           # match id -> real winner (forced into every simulation)
_fk = {}
for _m, (_a, _b) in R32.items():
    _fk[_m] = _ko_win(_a, _b)
    if _fk[_m]:
        FORCED[_m] = _fk[_m]
for _stage, _ms in ROUNDS:
    for _m, (_x, _y) in _ms.items():
        _fk[_m] = _ko_win(_fk.get(_x), _fk.get(_y))
        if _fk[_m]:
            FORCED[_m] = _fk[_m]

# ---- Group-stage sim inputs: base (played) stats + unplayed games with goal rates ----
_all_group_teams = [t for ts in groups.values() for t in ts]
_base_pts = {t: int(team_pts.get(t, 0)) for t in _all_group_teams}
_base_gd = {t: int(team_gd.get(t, 0)) for t in _all_group_teams}
_base_gf = {t: int(team_gf.get(t, 0)) for t in _all_group_teams}
_unplayed = {}     # group -> [(home, away, lambda_home, lambda_away)]
for _g, _ms in group_matches.items():
    _lst = []
    for (_h, _a, _ph, _pd, _pa) in _ms:
        if frozenset((_h, _a)) in actual_wc_pts:
            continue                                      # already played -> in base
        _lh, _la = implied_lambdas(_ph, _pd, _pa)
        _lst.append((_h, _a, _lh, _la))
    _unplayed[_g] = _lst

# ---- Simulate the WHOLE tournament once: remaining group games -> R32 field ->
#      knockouts. Field varies per run, so odds become true probabilities. ----
def simulate_once(reached):
    win_s, run_s, third_first_s, third_stats = {}, {}, {}, {}
    for g, teams in groups.items():
        pts = {t: _base_pts[t] for t in teams}
        gd = {t: _base_gd[t] for t in teams}
        gf = {t: _base_gf[t] for t in teams}
        for (h, a, lh, la) in _unplayed[g]:               # play out remaining group games
            hg, ag = _pois_sample(lh), _pois_sample(la)
            if hg > ag: pts[h] += 3
            elif ag > hg: pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
            gd[h] += hg - ag; gd[a] += ag - hg; gf[h] += hg; gf[a] += ag
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        win_s[g], run_s[g], third_first_s[g] = order[0], order[1], order[2]
        third_stats[g] = (pts[order[2]], gd[order[2]], gf[order[2]])
    best = sorted(groups.keys(), key=lambda g: third_stats[g], reverse=True)[:8]
    r32 = build_r32(win_s, run_s, _assign_thirds(best), third_first_s)

    w = {}
    for a, b in r32.values():
        reached[a]['R32'] += 1; reached[b]['R32'] += 1
    for m, (a, b) in r32.items():
        w[m] = FORCED[m] if m in FORCED else _play(a, b)
        reached[w[m]]['R16'] += 1
    for stage, matches in ROUNDS:
        for m, (x, y) in matches.items():
            w[m] = FORCED[m] if m in FORCED else _play(w[x], w[y])
            reached[w[m]][stage] += 1
    return w[104]

# ---- 5. Run it 10,000 times and tally champions + rounds reached ----
random.seed(42)   # reproducible odds: same inputs -> same odds, only real results move them
N = 10_000
titles = defaultdict(int)
reached = defaultdict(lambda: defaultdict(int))   # team -> stage -> count
for _ in range(N):
    titles[simulate_once(reached)] += 1

print(f"\n===== CHAMPIONSHIP ODDS ({N:,} simulations) =====")
ranked = sorted(titles.items(), key=lambda x: x[1], reverse=True)
for team, wins in ranked[:15]:
    print(f"  {team:<18} {wins / N:6.1%}")


# ============================================================
# BACKTEST — past World Cups, point-in-time, no leakage
# ============================================================
from sklearn.metrics import accuracy_score, log_loss

# Build point-in-time snapshots for ALL matches in history, so we have
# pre-tournament training data even for 2014. (Same snapshot-before-update
# discipline as before — every row uses ONLY prior matches.)
elo_bt = defaultdict(lambda: 1000)
recent_bt = defaultdict(lambda: deque(maxlen=10))

def form_bt(team):
    games = recent_bt[team]
    if not games:
        return 1.0, 0.0
    return (sum(g[0] for g in games) / len(games),
            sum(g[1] for g in games) / len(games))

bt_rows = []
for _, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    hs, as_ = row['home_score'], row['away_score']
    neutral = row['neutral']

    h_elo, a_elo = elo_bt[home], elo_bt[away]      # SNAPSHOT (pre-match)
    h_pts, h_gd = form_bt(home)
    a_pts, a_gd = form_bt(away)
    h_squad = squad_strength(home, row['date'])
    a_squad = squad_strength(away, row['date'])
    result = 2 if hs > as_ else (0 if hs < as_ else 1)

    bt_rows.append({
        'date': row['date'], 'tournament': row['tournament'],
        'home_elo': h_elo, 'away_elo': a_elo, 'elo_diff': h_elo - a_elo,
        'home_form': h_pts, 'away_form': a_pts,
        'home_gd_form': h_gd, 'away_gd_form': a_gd,
        'home_squad': h_squad, 'away_squad': a_squad,
        'squad_diff': h_squad - a_squad,
        'neutral': int(neutral), 'result': result,
    })

    home_adv = 0 if neutral else 50                # UPDATE elo
    exp_home = expected_score(h_elo + home_adv, a_elo)
    ah = 1 if result == 2 else (0 if result == 0 else 0.5)
    k = get_k(row['tournament'])
    elo_bt[home] += k * (ah - exp_home)
    elo_bt[away] += k * ((1 - ah) - (1 - exp_home))

    gd = hs - as_                                   # UPDATE form
    hp = 3 if result == 2 else (1 if result == 1 else 0)
    ap = 3 if result == 0 else (1 if result == 1 else 0)
    recent_bt[home].append((hp, gd))
    recent_bt[away].append((ap, -gd))

bt = pd.DataFrame(bt_rows)

def backtest_wc(year, cutoff):
    """Train ONLY on matches before `cutoff`, predict that year's World Cup.
       Reports the RAW model and the CALIBRATED model side by side — both
       trained/calibrated solely on pre-tournament data (no leakage)."""
    from sklearn.calibration import CalibratedClassifierCV
    train = bt[bt['date'] < cutoff]
    wc = bt[(bt['tournament'] == 'FIFA World Cup') & (bt['date'].dt.year == year)]
    Xtr, ytr = train[feature_cols], train[target_col]
    Xwc, ywc = wc[feature_cols], wc[target_col]

    cfg = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
               objective='multi:softprob', eval_metric='mlogloss', random_state=42)

    # Raw model
    m = XGBClassifier(**cfg)
    m.fit(Xtr, ytr)
    acc_r = accuracy_score(ywc, m.predict(Xwc))
    ll_r  = log_loss(ywc, m.predict_proba(Xwc), labels=[0, 1, 2])

    # Calibrated model (isotonic, fit via CV on pre-tournament data only)
    cal = CalibratedClassifierCV(XGBClassifier(**cfg), method='isotonic', cv=3)
    cal.fit(Xtr, ytr)
    acc_c = accuracy_score(ywc, cal.predict(Xwc))
    ll_c  = log_loss(ywc, cal.predict_proba(Xwc), labels=[0, 1, 2])

    base = (ywc == 2).mean()
    print(f"  WC {year}: {len(wc):>2} matches | base(home) {base:5.1%}")
    print(f"           raw        -> acc {acc_r:5.1%} | logloss {ll_r:.3f}")
    print(f"           calibrated -> acc {acc_c:5.1%} | logloss {ll_c:.3f}")

print("\n===== BACKTEST: past World Cups (squad + calibration, no leakage) =====")
backtest_wc(2014, '2014-06-01')
backtest_wc(2018, '2018-06-01')
backtest_wc(2022, '2022-11-01')


# ============================================================
# PROBABILITY CALIBRATION (isotonic) — fix draw-inflation
# ============================================================
from sklearn.calibration import CalibratedClassifierCV

# A fresh base model, calibrated via internal cross-validation on the
# training set. CalibratedClassifierCV refits the model on CV folds and
# fits the isotonic correction on the held-out folds — so the correction
# is learned on data the model didn't train on.
base = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
    objective='multi:softprob', eval_metric='mlogloss', random_state=42,
)
calibrated = CalibratedClassifierCV(base, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)

# Compare log loss on the untouched test set (lower = better)
probs_raw = model.predict_proba(X_test)
probs_cal = calibrated.predict_proba(X_test)
print("\n===== CALIBRATION =====")
print(f"  Log loss  raw:        {log_loss(y_test, probs_raw):.3f}")
print(f"  Log loss  calibrated: {log_loss(y_test, probs_cal):.3f}")
print(f"  Accuracy  raw:        {accuracy_score(y_test, model.predict(X_test)):.1%}")
print(f"  Accuracy  calibrated: {accuracy_score(y_test, calibrated.predict(X_test)):.1%}")

# Did the draw-inflation actually shrink? Check the offending mismatches.
def compare_probs(home, away):
    h, a = elo_ratings[home], elo_ratings[away]
    hp, hg = form_stats(home)
    ap, ag = form_stats(away)
    hsq, asq = squad_strength(home), squad_strength(away)
    row = pd.DataFrame([{
        'home_elo': h, 'away_elo': a, 'elo_diff': h - a,
        'home_form': hp, 'away_form': ap,
        'home_gd_form': hg, 'away_gd_form': ag,
        'home_squad': hsq, 'away_squad': asq, 'squad_diff': hsq - asq,
        'neutral': 1,
    }])[feature_cols]
    r = model.predict_proba(row)[0]
    c = calibrated.predict_proba(row)[0]
    print(f"\n  {home} vs {away}   [away / draw / home]")
    print(f"    raw:        {r[0]:.0%} / {r[1]:.0%} / {r[2]:.0%}")
    print(f"    calibrated: {c[0]:.0%} / {c[1]:.0%} / {c[2]:.0%}")

compare_probs('Spain', 'Saudi Arabia')
compare_probs('Argentina', 'Jordan')


# ============================================================
# EXPORT — write all predictions to predictions.json for the dashboard
# ============================================================
import json
import os

STAGE_ORDER = ['R32', 'R16', 'QF', 'SF', 'Final', 'Champion']

# Freeze per-game predictions for games already played: never re-predict a
# match in hindsight. A WC pairing present in df (NaN scores were dropped) is
# a played game; keep its prediction from the existing predictions.json.
_played_pairs = {frozenset((r['home_team'], r['away_team']))
                 for _, r in df[(df['tournament'] == 'FIFA World Cup')
                                & (df['date'].dt.year == 2026)].iterrows()}
_locked_gm = {}
_prev_history = []                       # championship-odds snapshots from prior runs
if os.path.exists('predictions.json'):
    try:
        _prev = json.load(open('predictions.json', encoding='utf-8'))
        for _ms in _prev.get('group_matches', {}).values():
            for _m in _ms:
                _locked_gm[frozenset((_m['home'], _m['away']))] = _m
        _prev_history = _prev.get('odds_history', [])
    except Exception:
        pass

def _gm_entry(h, a, ph, pd_, pa):
    pair = frozenset((h, a))
    if pair in _played_pairs and pair in _locked_gm:
        return _locked_gm[pair]          # locked pre-game prediction (game over)
    lh, la = implied_lambdas(float(ph), float(pd_), float(pa))   # live-odds prior
    return {'home': h, 'away': a,
            'p_home': round(float(ph), 3), 'p_draw': round(float(pd_), 3),
            'p_away': round(float(pa), 3),
            'lh': lh, 'la': la}

# ---- Bracket export: tree structure + R32 confirmed flags + real KO winners ----
# A team's R32 slot is "confirmed" once its group's 6 games are all played
# (until then it's the model's projection). Knockout winners come from actual
# results — blank until a match is really played (penalty draws stay blank).
_team_group = {t: g for g, ts in groups.items() for t in ts}
_group_done = {g: sum(1 for pr in combinations(ts, 2)
                      if frozenset(pr) in actual_wc_pts) == 6
               for g, ts in groups.items()}

# Teams mathematically guaranteed a top-2 finish (so: guaranteed into the R32),
# found by enumerating every outcome of that group's remaining games. Ties are
# assumed to break AGAINST the team, so this never flags a false clinch.
from itertools import product
def _clinched_top2(group_teams):
    base = {t: 0 for t in group_teams}
    remaining = []
    for h, a in combinations(group_teams, 2):
        pair = frozenset((h, a))
        if pair in actual_wc_pts:
            base[h] += actual_wc_pts[pair][h]
            base[a] += actual_wc_pts[pair][a]
        else:
            remaining.append((h, a))
    clinched = set()
    for t in group_teams:
        safe = True
        for combo in product((0, 1, 2), repeat=len(remaining)):   # 0=home,1=draw,2=away
            pts = dict(base)
            for (h, a), o in zip(remaining, combo):
                if o == 0: pts[h] += 3
                elif o == 2: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
            above = sum(1 for x in group_teams if x != t and pts[x] >= pts[t])
            if above >= 2:           # some scenario drops t to 3rd or worse
                safe = False
                break
        if safe:
            clinched.add(t)
    return clinched

_clinched = set()
for _g, _ts in groups.items():
    _clinched |= _clinched_top2(_ts)

def _confirmed(t):
    return bool(t) and (t in _clinched or _group_done.get(_team_group.get(t), False))

_res_lookup = {}   # frozenset(pair) -> (home, away, home_score, away_score)
for _r in df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2026)].itertuples():
    _res_lookup[frozenset((_r.home_team, _r.away_team))] = (
        _r.home_team, _r.away_team, _r.home_score, _r.away_score)
def _real_winner(a, b):
    if not a or not b:
        return None
    g = _res_lookup.get(frozenset((a, b)))
    if not g or g[2] == g[3]:        # not played, or draw (penalties unknown)
        return None
    return g[0] if g[2] > g[3] else g[1]

_feeders = {}
_parent = {}                       # match id -> match its winner advances into
for _stage, _ms in ROUNDS:
    _feeders.update(_ms)
    for _m, (_x, _y) in _ms.items():
        _parent[_x] = _m
        _parent[_y] = _m
def _leaf_order(m):
    if m in R32:
        return [m]
    x, y = _feeders[m]
    return _leaf_order(x) + _leaf_order(y)
_r32_order = _leaf_order(104)

_bk_winners = {}
for _m, (_a, _b) in R32.items():
    _bk_winners[_m] = _real_winner(_a, _b)
for _stage, _ms in ROUNDS:
    for _m, (_x, _y) in _ms.items():
        _bk_winners[_m] = _real_winner(_bk_winners.get(_x), _bk_winners.get(_y))

bracket = {
    'r32_order': _r32_order,
    'r32': {str(m): {'a': R32[m][0], 'b': R32[m][1],
                     'a_conf': _confirmed(R32[m][0]), 'b_conf': _confirmed(R32[m][1])}
            for m in _r32_order},
    # each column holds the match ids whose WINNERS fill that round
    'columns': [
        _r32_order,                          # winners -> Round of 16
        [89, 90, 93, 94, 91, 92, 95, 96],    # winners -> Quarterfinals
        [97, 98, 99, 100],                   # winners -> Semifinals
        [101, 102],                          # winners -> Final
        [104],                               # winner  -> Champion
    ],
    'winners': {str(m): w for m, w in _bk_winners.items()},
    'parent': {str(c): p for c, p in _parent.items()},
}

# Actual points + games played so far (from played 2026 WC games)
_team_actual_pts = defaultdict(int)
_team_gp = defaultdict(int)
for _pair, _d in actual_wc_pts.items():
    for _t, _p in _d.items():
        _team_actual_pts[_t] += _p
        _team_gp[_t] += 1

# Per-team ACTUAL progress (facts) for the stage table: furthest round reached
# and whether the team is eliminated. Drives the ✅ / ❌ marks in the dashboard.
_stage_idx = {s: i for i, s in enumerate(STAGE_ORDER)}
_stage_of_winner = {m: 'R16' for m in R32}        # winning an R32 match reaches R16
for _stage, _ms in ROUNDS:
    for _m in _ms:
        _stage_of_winner[_m] = _stage
_all_teams = set(_team_group)
_r32_field = set()
for _a, _b in R32.values():
    _r32_field |= {_a, _b}
_all_groups_done = all(_group_done.values())
_reached, _elim = {}, set()
def _bump(team, stage):
    if team and (_reached.get(team) is None
                 or _stage_idx[stage] > _stage_idx[_reached[team]]):
        _reached[team] = stage
for _t in _all_teams:                              # reached R32: clinched, or field final
    if _t in _clinched or (_all_groups_done and _t in _r32_field):
        _bump(_t, 'R32')
if _all_groups_done:
    _elim |= (_all_teams - _r32_field)
_actual_entrants = {_m: (R32[_m][0], R32[_m][1]) for _m in R32}
for _stage, _ms in ROUNDS:
    for _m, (_x, _y) in _ms.items():
        _actual_entrants[_m] = (_bk_winners.get(_x), _bk_winners.get(_y))
for _m, (_a, _b) in _actual_entrants.items():
    _w = _bk_winners.get(_m)
    if not _w:
        continue
    _st = _stage_of_winner[_m]
    _bump(_w, _st)                                  # winner reached this round
    _loser = _b if _w == _a else _a
    if _loser:
        _elim.add(_loser)                          # loser eliminated here
        _bump(_loser, STAGE_ORDER[_stage_idx[_st] - 1])

export = {
    'generated': str(pd.Timestamp.now()),
    'n_sims': N,
    # Group tables: group -> [{team, xpts, advances}]
    'groups': {
        g: [{'team': t, 'gp': int(_team_gp.get(t, 0)),
             'pts': int(_team_actual_pts.get(t, 0)),
             'gd': int(team_gd.get(t, 0)), 'gf': int(team_gf.get(t, 0)),
             'xpts': round(float(p), 2), 'advances': i < 2,
             'clinched': t in _clinched}
            for i, (t, p) in enumerate(standings)]
        for g, standings in group_standings.items()
    },
    # Per-match group probabilities (played games keep their pre-game values)
    'group_matches': {
        g: [_gm_entry(h, a, ph, pd_, pa) for (h, a, ph, pd_, pa) in matches]
        for g, matches in group_matches.items()
    },
    # Championship odds (all 32, sorted)
    'champion_odds': [
        {'team': t, 'prob': round(float(w) / N, 4)}
        for t, w in sorted(titles.items(), key=lambda x: x[1], reverse=True)
    ],
    # Odds to reach each stage: team -> {stage: prob}
    'stage_odds': {
        t: {s: round(float(reached[t].get(s, 0)) / N, 4) for s in STAGE_ORDER}
        for t in reached
    },
    'bracket': bracket,
    # Per-team facts: furthest round actually reached + whether eliminated
    'stage_progress': {t: {'reached': _reached.get(t), 'eliminated': t in _elim}
                       for t in _all_teams},
}

# ---- Track record: model's pre-match pick vs actual result for every game ----
_actual_played = {}
for _r in df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2026)].itertuples():
    _actual_played[frozenset((_r.home_team, _r.away_team))] = (
        _r.home_team, int(_r.home_score), int(_r.away_score), str(_r.date.date()))
_track = []
for _g, _ms in export['group_matches'].items():
    for _m in _ms:
        _pair = frozenset((_m['home'], _m['away']))
        if _pair not in _actual_played:
            continue
        _ph, _pdr, _pa = _m['p_home'], _m['p_draw'], _m['p_away']
        _pred = (_m['home'] if (_ph >= _pa and _ph >= _pdr)
                 else (_m['away'] if (_pa >= _ph and _pa >= _pdr) else 'Draw'))
        _hh, _hs, _as, _dt = _actual_played[_pair]
        _oh, _oa = (_hs, _as) if _hh == _m['home'] else (_as, _hs)
        _act = _m['home'] if _oh > _oa else (_m['away'] if _oa > _oh else 'Draw')
        _track.append({'date': _dt, 'home': _m['home'], 'away': _m['away'],
                       'score': f'{_oh}-{_oa}', 'pred': _pred, 'actual': _act,
                       'correct': _pred == _act, 'stage': f'Group {_g}'})
_track.sort(key=lambda x: x['date'])
export['track_record'] = _track

# ---- Championship-odds history: append a snapshot only when the odds change ----
_champ_now = {x['team']: x['prob'] for x in export['champion_odds']}
_history = list(_prev_history)
if not _history or _history[-1].get('champ') != _champ_now:
    _history.append({'ts': str(pd.Timestamp.now()),
                     'n_played': len(_played_pairs), 'champ': _champ_now})
export['odds_history'] = _history

with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)

print("\n✅ Exported predictions.json for the dashboard")