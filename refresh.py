"""
Auto-refresh: pull FINISHED World Cup results from football-data.org,
append any new ones to results.csv, then re-run predictor.py so the
dashboard's standings / odds reflect what's actually happened.

Run with:        python refresh.py
Dry run (no write, no re-run):   python refresh.py --dry-run

Needs the same football-data.org API key as the dashboard
(env var FOOTBALL_DATA_API_KEY, api_key.txt, or .streamlit/secrets.toml).
"""
import os
import sys
import subprocess

import pandas as pd
import requests

HOME = os.path.expanduser('~')
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HOME, 'results.csv')
PREDICTOR = os.path.join(HERE, 'predictor.py')
HOSTS = {'Mexico', 'Canada', 'United States'}

# football-data.org team names -> our naming (keep in sync with dashboard.py)
API_NAME_MAP = {
    'Korea Republic': 'South Korea', 'Republic of Korea': 'South Korea',
    'USA': 'United States', 'United States of America': 'United States',
    'Czechia': 'Czech Republic', 'Türkiye': 'Turkey', 'Turkiye': 'Turkey',
    "Côte d'Ivoire": 'Ivory Coast', "Cote d'Ivoire": 'Ivory Coast',
    'Cape Verde Islands': 'Cape Verde', 'Congo DR': 'DR Congo',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
}


def norm(name):
    return API_NAME_MAP.get(name, name).strip() if name else ''


def get_api_key():
    if os.environ.get('FOOTBALL_DATA_API_KEY'):
        return os.environ['FOOTBALL_DATA_API_KEY'].strip()
    for path in (os.path.join(HOME, 'api_key.txt'), os.path.join(HERE, 'api_key.txt')):
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read().strip()
    return None


def fetch_finished():
    key = get_api_key()
    if not key:
        sys.exit("No API key found (set FOOTBALL_DATA_API_KEY or create api_key.txt).")
    r = requests.get('https://api.football-data.org/v4/competitions/WC/matches',
                     headers={'X-Auth-Token': key}, timeout=15)
    r.raise_for_status()
    rows = []
    for m in r.json().get('matches', []):
        if m.get('status') != 'FINISHED':
            continue
        ft = (m.get('score') or {}).get('fullTime') or {}
        if ft.get('home') is None:
            continue
        h = norm((m.get('homeTeam') or {}).get('name'))
        a = norm((m.get('awayTeam') or {}).get('name'))
        if not h or not a:
            continue
        rows.append({
            'date': m['utcDate'][:10],
            'home_team': h, 'away_team': a,
            'home_score': float(ft['home']), 'away_score': float(ft['away']),
            'tournament': 'FIFA World Cup', 'city': '', 'country': '',
            # dataset convention: host plays "home" (non-neutral), everyone else neutral
            'neutral': h not in HOSTS,
            # who advanced (incl. extra time / penalties): HOME_TEAM / AWAY_TEAM / DRAW
            'winner': (m.get('score') or {}).get('winner'),
        })
    return pd.DataFrame(rows)


def main():
    dry = '--dry-run' in sys.argv

    new = fetch_finished()
    print(f"Fetched {len(new)} finished World Cup matches from the API.")
    if new.empty:
        print("Nothing finished yet — nothing to do.")
        return

    res = pd.read_csv(RESULTS)
    if 'winner' not in res.columns:
        res['winner'] = ''                     # advancer flag (for KO shootouts/ET)
    res_dt = pd.to_datetime(res['date'], errors='coerce')

    filled, skipped, appended = [], 0, []

    for _, m in new.iterrows():
        api_date = pd.Timestamp(m['date'])
        teams = {m['home_team'], m['away_team']}
        # same pairing (either order), within a day of the API's UTC date
        same_pair = (res['home_team'].isin(teams) & res['away_team'].isin(teams)
                     & (res['home_team'] != res['away_team']))
        near = (res_dt - api_date).abs() <= pd.Timedelta(days=1)
        cand = res[same_pair & near]

        if (cand['home_score'].notna()).any():
            skipped += 1                       # already scored in the file
            continue

        unplayed = cand[cand['home_score'].isna()]
        if len(unplayed):
            idx = unplayed.index[0]            # fill the existing fixture row
            if res.at[idx, 'home_team'] == m['home_team']:
                res.at[idx, 'home_score'] = m['home_score']
                res.at[idx, 'away_score'] = m['away_score']
                res.at[idx, 'winner'] = m['winner']
            else:                              # row stores the pairing flipped
                res.at[idx, 'home_score'] = m['away_score']
                res.at[idx, 'away_score'] = m['home_score']
                _w = m['winner']               # flip the advancer flag to match row order
                res.at[idx, 'winner'] = ('AWAY_TEAM' if _w == 'HOME_TEAM'
                                         else 'HOME_TEAM' if _w == 'AWAY_TEAM' else _w)
            filled.append((res.at[idx, 'date'], res.at[idx, 'home_team'],
                           int(res.at[idx, 'home_score']), int(res.at[idx, 'away_score']),
                           res.at[idx, 'away_team']))
        else:
            appended.append(m)                 # truly missing from the file

    print(f"\n{len(filled)} fixture row(s) to fill with scores:")
    for d, h, hs, as_, a in filled:
        print(f"  {d}  {h} {hs}-{as_} {a}")
    if skipped:
        print(f"{skipped} match(es) already scored in results.csv — skipped.")
    if appended:
        print(f"{len(appended)} match(es) not found as fixtures — would append:")
        for m in appended:
            print(f"  {m['date']}  {m['home_team']} {int(m['home_score'])}-"
                  f"{int(m['away_score'])} {m['away_team']}")

    if not filled and not appended:
        print("results.csv already up to date.")
        return

    if dry:
        print("\n[dry-run] No changes written, predictor.py not re-run.")
        return

    if appended:
        res = pd.concat([res, pd.DataFrame(appended)], ignore_index=True)
    res.to_csv(RESULTS, index=False)
    print(f"\nUpdated results.csv ({len(filled)} filled, {len(appended)} appended).")

    print("Re-running predictor.py (this takes a minute)...")
    subprocess.run([sys.executable, PREDICTOR], cwd=HOME, check=True,
                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    print("Done — predictions.json refreshed. Reload the dashboard to see updated odds.")


if __name__ == '__main__':
    main()
