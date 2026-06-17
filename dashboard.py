"""
World Cup 2026 Predictor — dashboard.
Reads predictions.json (from predictor.py) + fixtures.csv (your schedule)
and overlays live scores from football-data.org.

Run with:   python -m streamlit run dashboard.py

Live scores need a free API key from https://www.football-data.org/client/register
Provide it via ANY of:
  - environment variable  FOOTBALL_DATA_API_KEY
  - a file  api_key.txt  (in home dir or next to this script)
  - .streamlit/secrets.toml  ->  FOOTBALL_DATA_API_KEY = "..."
"""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

ET = ZoneInfo('America/New_York')   # fixture times are entered in US Eastern (EST/EDT auto)

HOME = os.path.expanduser('~')
HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    """Find a data file in cwd, home dir, or next to this script."""
    for path in (name, os.path.join(HOME, name), os.path.join(HERE, name)):
        if os.path.exists(path):
            return path
    return None


# ----- Load predictions + fixtures -----
def load_json(name):
    path = _find(name)
    if not path:
        return None, None
    with open(path, encoding='utf-8') as f:
        return json.load(f), path


def load_fixtures():
    path = _find('fixtures.csv')
    if not path:
        return None
    df = pd.read_csv(path, dtype=str).fillna('')
    df = df[df['date'].str.strip() != '']          # only rows with a date
    if df.empty:
        return df
    # Times are entered in US Eastern; parse naive, localize to ET, store as UTC.
    naive = pd.to_datetime(
        df['date'].str.strip() + ' ' + df['time_et'].str.strip().replace('', '00:00'),
        errors='coerce')
    df['dt'] = (naive.dt.tz_localize(ET, ambiguous='NaT', nonexistent='shift_forward')
                     .dt.tz_convert('UTC'))
    return df.sort_values('dt')


# ----- API key + live scores -----
def get_api_key():
    if os.environ.get('FOOTBALL_DATA_API_KEY'):
        return os.environ['FOOTBALL_DATA_API_KEY'].strip()
    keyfile = _find('api_key.txt')
    if keyfile:
        with open(keyfile, encoding='utf-8') as f:
            return f.read().strip()
    try:
        return st.secrets['FOOTBALL_DATA_API_KEY']
    except Exception:
        return None


# Map football-data.org team names -> our naming (extend as needed)
API_NAME_MAP = {
    'Korea Republic': 'South Korea', 'Republic of Korea': 'South Korea',
    'USA': 'United States', 'United States of America': 'United States',
    'Czechia': 'Czech Republic', 'Türkiye': 'Turkey', 'Turkiye': 'Turkey',
    'Côte d\'Ivoire': 'Ivory Coast', 'Cote d\'Ivoire': 'Ivory Coast',
    'Cape Verde Islands': 'Cape Verde', 'Congo DR': 'DR Congo',
    'Curaçao': 'Curaçao', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
}


def norm(name):
    if not name:
        return ''
    return API_NAME_MAP.get(name, name).strip()


@st.cache_data(ttl=60)   # refresh live data at most once a minute
def fetch_live():
    """Return {frozenset({home,away}): {status, home_score, away_score, utcDate}}."""
    key = get_api_key()
    if not key:
        return {'_error': 'no_key'}
    try:
        r = requests.get(
            'https://api.football-data.org/v4/competitions/WC/matches',
            headers={'X-Auth-Token': key}, timeout=10)
        r.raise_for_status()
        out = {}
        for m in r.json().get('matches', []):
            h = norm((m.get('homeTeam') or {}).get('name'))
            a = norm((m.get('awayTeam') or {}).get('name'))
            if not h or not a:
                continue
            ft = (m.get('score') or {}).get('fullTime') or {}
            out[frozenset((h, a))] = {
                'status': m.get('status'),
                'home': h, 'away': a,
                'home_score': ft.get('home'),
                'away_score': ft.get('away'),
                'utcDate': m.get('utcDate'),
            }
        return out
    except Exception as e:
        return {'_error': str(e)}


# =====================================================================
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")
st.title("⚽ World Cup 2026 — Prediction Dashboard")

data, src = load_json('predictions.json')
if data is None:
    st.error("Could not find predictions.json. Run `python predictor.py` first.")
    st.stop()

# Prediction lookup: frozenset({home,away}) -> (p_home, p_draw, p_away, home, away)
pred_lookup = {}
for g, matches in data['group_matches'].items():
    for m in matches:
        pred_lookup[frozenset((m['home'], m['away']))] = m

fixtures = load_fixtures()
live = fetch_live()

st.caption(f"Predictions generated {data['generated']}  ·  {data['n_sims']:,} simulations")

tab_games, tab_champ, tab_runs, tab_groups = st.tabs(
    ["📅 Game by Game", "🏆 Championship odds", "📈 Run to the final", "🗂️ Groups"]
)

LIVE_STATUSES = {'IN_PLAY', 'PAUSED', 'LIVE'}


def _outcome(hs, as_):
    return 'home' if hs > as_ else ('away' if hs < as_ else 'draw')


def fixture_result(fx_row, live_info):
    """For a fixture + its live data, return (completed, pred_out, actual_out).
       completed=True only for finished games; pred_out is None if no model line."""
    home, away = fx_row['home'], fx_row['away']
    if not (live_info and live_info.get('home_score') is not None):
        return False, None, None
    if live_info.get('status') in LIVE_STATUSES:
        return False, None, None      # still in progress, not final
    if live_info.get('home') == home:
        hs, as_ = live_info['home_score'], live_info['away_score']
    else:
        hs, as_ = live_info['away_score'], live_info['home_score']
    pred = pred_lookup.get(frozenset((home, away)))
    pred_out = None
    if pred:
        if pred['home'] == home:
            ph, pd_, pa = pred['p_home'], pred['p_draw'], pred['p_away']
        else:
            ph, pd_, pa = pred['p_away'], pred['p_draw'], pred['p_home']
        pred_out = max((('home', ph), ('draw', pd_), ('away', pa)), key=lambda x: x[1])[0]
    return True, pred_out, _outcome(hs, as_)


def render_match_card(fx_row, live_info, show_pred=True):
    """Render one match: teams, completion status, score, and how the model did."""
    home, away = fx_row['home'], fx_row['away']
    when = (fx_row['dt'].tz_convert(ET).strftime('%b %d, %I:%M %p ET')
            if pd.notna(fx_row['dt']) else fx_row['date'])
    stage = fx_row['stage']

    # Model prediction, oriented to this fixture's home/away
    pred = pred_lookup.get(frozenset((home, away)))
    ph = pd_ = pa = pred_out = None
    if pred:
        if pred['home'] == home:
            ph, pd_, pa = pred['p_home'], pred['p_draw'], pred['p_away']
        else:
            ph, pd_, pa = pred['p_away'], pred['p_draw'], pred['p_home']
        pred_out = max((('home', ph), ('draw', pd_), ('away', pa)), key=lambda x: x[1])[0]

    has_score = live_info and live_info.get('home_score') is not None
    if has_score:
        # Orient the API score to the fixture's home/away
        if live_info.get('home') == home:
            hs, as_ = live_info['home_score'], live_info['away_score']
        else:
            hs, as_ = live_info['away_score'], live_info['home_score']
        status = live_info.get('status', '')
        st.markdown(f"### {home}  {int(hs)} – {int(as_)}  {away}")

        if status in LIVE_STATUSES:
            st.caption(f"{stage} · {when} · 🔴 LIVE")
        else:  # completed
            line = f"{stage} · {when} · 🏁 FINAL"
            if pred_out is not None:
                line += "  ·  model ✅ called it" if pred_out == _outcome(hs, as_) \
                        else "  ·  model ❌ missed"
            st.caption(line)
    else:
        st.markdown(f"**{home}**  vs  **{away}**")
        st.caption(f"{stage} · {when} · ⏳ scheduled")

    if show_pred and ph is not None:
        pick = {'home': home, 'draw': 'draw', 'away': away}[pred_out]
        st.caption(f"Model: {home} {ph:.0%} · draw {pd_:.0%} · {away} {pa:.0%}  ·  pick: **{pick}**")


# ===== TAB: Game by Game =====
with tab_games:
    if fixtures is None:
        st.warning("No `fixtures.csv` found. Create it (date,time_utc,stage,home,away) "
                   "to populate this tab.")
    elif fixtures.empty:
        st.info("`fixtures.csv` has no dated rows yet. Fill in the `date` column.")
    else:
        # Live-data status banner
        if live.get('_error') == 'no_key':
            st.info("💤 Live scores off — add a football-data.org API key to enable "
                    "(see dashboard.py header).")
        elif '_error' in live:
            st.warning(f"Live scores unavailable: {live['_error']}")

        # ---- Model accuracy scoreboard (completed games so far) ----
        graded = [(p, a) for _, fx in fixtures.iterrows()
                  for done, p, a in [fixture_result(fx, live.get(frozenset((fx['home'], fx['away']))))]
                  if done and p is not None]
        total = len(graded)
        correct = sum(p == a for p, a in graded)
        m1, m2 = st.columns(2)
        m1.metric("Model accuracy so far", f"{correct/total:.0%}" if total else "—")
        m2.metric("Games graded", f"{correct}/{total}" if total else "0")
        if not total:
            st.caption("No completed games with predictions yet — accuracy fills in as games finish.")
        st.divider()

        today = datetime.now(ET).date()

        # ---- Two big windows at the top ----
        col_today, col_live = st.columns(2)

        with col_today:
            st.subheader("📆 Today's matches")
            todays = fixtures[fixtures['dt'].dt.tz_convert(ET).dt.date == today]
            if todays.empty:
                st.write("_No matches scheduled today._")
            for _, fx in todays.iterrows():
                with st.container(border=True):
                    render_match_card(fx, live.get(frozenset((fx['home'], fx['away']))))

        with col_live:
            st.subheader("🔴 Live now")
            live_games = [v for k, v in live.items()
                          if isinstance(v, dict) and v.get('status') in LIVE_STATUSES]
            if not live_games:
                st.write("_No matches in progress right now._")
            for lg in live_games:
                with st.container(border=True):
                    st.markdown(f"### {lg['home']}  {lg['home_score']} – "
                                f"{lg['away_score']}  {lg['away']}")
                    st.caption(f"🔴 {lg.get('status')}")

        st.divider()

        # ---- Full schedule, chronological ----
        st.subheader("🗓️ Full schedule")
        for _, fx in fixtures.iterrows():
            with st.container(border=True):
                render_match_card(fx, live.get(frozenset((fx['home'], fx['away']))))


# ===== TAB: Championship odds =====
with tab_champ:
    st.subheader("Who wins it all?")
    champ = pd.DataFrame(data['champion_odds'])
    champ['prob'] = champ['prob'] * 100
    st.bar_chart(champ.head(15).set_index('team')['prob'], horizontal=True, height=500)
    st.dataframe(
        champ.rename(columns={'team': 'Team', 'prob': 'Title %'})
             .style.format({'Title %': '{:.1f}%'}),
        hide_index=True, use_container_width=True, height=400,
    )

# ===== TAB: Stage progression =====
with tab_runs:
    st.subheader("Probability of reaching each stage")
    stages = ['R32', 'R16', 'QF', 'SF', 'Final', 'Champion']
    rows = [{'Team': t, **{s: sd.get(s, 0) * 100 for s in stages}}
            for t, sd in data['stage_odds'].items()]
    df = pd.DataFrame(rows).sort_values('Champion', ascending=False)
    st.dataframe(
        df.style.format({s: '{:.0f}%' for s in stages}),
        hide_index=True, use_container_width=True, height=600,
    )

# ===== TAB: Groups =====
with tab_groups:
    st.subheader("Group stage")
    cols = st.columns(3)
    for i, g in enumerate(sorted(data['groups'].keys())):
        with cols[i % 3]:
            st.markdown(f"### Group {g}")
            standings = pd.DataFrame(data['groups'][g])
            standings['team'] = standings.apply(
                lambda r: f"✅ {r['team']}" if r['advances'] else f"　 {r['team']}", axis=1)
            st.dataframe(
                standings[['team', 'xpts']].rename(columns={'team': 'Team', 'xpts': 'xPts'}),
                hide_index=True, use_container_width=True,
            )
            with st.expander("Match predictions"):
                for m in data['group_matches'][g]:
                    st.write(f"**{m['home']}** {m['p_home']:.0%} · "
                             f"draw {m['p_draw']:.0%} · {m['p_away']:.0%} **{m['away']}**")
