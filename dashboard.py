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
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

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


import time

@st.cache_data(ttl=60)   # 60s: live odds stay fresh. Cache is shared across all
                         # users (one fetch per window), so this is rate-limit safe.
def _fetch_live_cached():
    """Fetch live scores, retrying transient drops. RAISES on hard failure so
    the failure is NOT cached and the next page load retries."""
    key = get_api_key()
    if not key:
        return {'_error': 'no_key'}
    last = None
    for attempt in range(3):
        try:
            r = requests.get(
                'https://api.football-data.org/v4/competitions/WC/matches',
                headers={'X-Auth-Token': key}, timeout=15)
            r.raise_for_status()
            out = {}
            for m in r.json().get('matches', []):
                h = norm((m.get('homeTeam') or {}).get('name'))
                a = norm((m.get('awayTeam') or {}).get('name'))
                if not h or not a:
                    continue
                ft = (m.get('score') or {}).get('fullTime') or {}
                out[frozenset((h, a))] = {
                    'status': m.get('status'), 'home': h, 'away': a,
                    'home_score': ft.get('home'), 'away_score': ft.get('away'),
                    'utcDate': m.get('utcDate'),
                    'winner': (m.get('score') or {}).get('winner'),  # advancer (ET/pens)
                }
            return out
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last)   # not cached -> retried on next rerun

def fetch_live():
    """Serve fresh/cached live data; fall back to last-known on API failure."""
    try:
        data = _fetch_live_cached()
        if '_error' not in data:
            st.session_state['_last_live'] = data
        return data
    except Exception as e:
        if st.session_state.get('_last_live'):          # API down -> show last-good scores
            stale = dict(st.session_state['_last_live']); stale['_stale'] = str(e)
            return stale
        return {'_error': str(e)}


# =====================================================================
st.set_page_config(page_title="2026 FIFA World Cup Predictor Dashboard", page_icon="⚽", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; }
  h1 { margin-bottom: 0.25rem; padding-bottom: 0; }
</style>
""", unsafe_allow_html=True)
st.markdown(
    '<h1 style="margin:0; padding:0;">⚽ FIFA World Cup 2026 — Real Time Calibrated Prediction Dashboard</h1>',
    unsafe_allow_html=True,
)
data, src = load_json('predictions.json')
if data is None:
    st.error("Could not find predictions.json. Run `python predictor.py` first.")
    st.stop()

# Prediction lookup: frozenset({home,away}) -> match prediction entry
pred_lookup = {}
for g, matches in data['group_matches'].items():
    for m in matches:
        pred_lookup[frozenset((m['home'], m['away']))] = m
for m in data.get('knockout_matches', []):          # R32+ matchups (binary: someone advances)
    pred_lookup[frozenset((m['home'], m['away']))] = m

# Predictor-resolved knockout advancers (override-aware; the API can leave shootout
# winners blank, so we trust the predictor's bracket over the live feed for grading).
ko_advancer = {}
_bw = data.get('bracket', {}).get('winners', {})
for m in data.get('knockout_matches', []):
    w = _bw.get(str(m.get('match')))
    if w:
        ko_advancer[frozenset((m['home'], m['away']))] = w

fixtures = load_fixtures()
live = fetch_live()

st.caption(f"Odds from {data['n_sims']:,} Monte Carlo simulations")

tab_games, tab_runs, tab_champ, tab_track, tab_groups, tab_about = st.tabs(
    ["📅 Game by Game", "📈 Run to the final", "🏆 Championship odds",
     "📋 Track record", "🗂️ Groups", "About/Methodology"]
)

LIVE_STATUSES = {'IN_PLAY', 'PAUSED', 'LIVE'}
# A game only counts as finished when the API says so. Do NOT infer "final" from
# "not live" — the free tier sometimes attaches a bogus fullTime score to a
# TIMED/SCHEDULED (not-started) game, which must never render or grade as a result.
FINAL_STATUSES = {'FINISHED', 'AWARDED'}


# ----- Shared accuracy record: group stage (frozen) + knockouts (live) -----
# Computed once here so the Game-by-Game and Track Record tabs always agree.
def _ko_record(fx):
    """Full record for a COMPLETED knockout game (graded by who advanced), else None."""
    home, away = fx['home'], fx['away']
    li = live.get(frozenset((home, away)))
    pred = pred_lookup.get(frozenset((home, away)))
    if not (li and li.get('home_score') is not None
            and li.get('status') in FINAL_STATUSES):
        return None
    if not (pred and 'p_home_adv' in pred):
        return None
    hs, as_ = ((li['home_score'], li['away_score']) if li.get('home') == home
               else (li['away_score'], li['home_score']))
    adv_h, adv_a = ((pred['p_home_adv'], pred['p_away_adv']) if pred['home'] == home
                    else (pred['p_away_adv'], pred['p_home_adv']))
    pick = home if adv_h >= adv_a else away
    actual = ko_advancer.get(frozenset((home, away)))    # predictor's resolution first
    if actual is None:                                   # fall back to the live result
        if hs > as_:
            actual = home
        elif as_ > hs:
            actual = away
        else:
            wf = li.get('winner')
            actual = (li.get('home') if wf == 'HOME_TEAM'
                      else li.get('away') if wf == 'AWAY_TEAM' else None)
    if actual is None:
        return None
    return {'date': fx.get('date', ''), 'home': home, 'away': away,
            'score': f'{int(hs)}-{int(as_)}', 'pred': pick, 'actual': actual,
            'correct': pick == actual, 'stage': fx.get('stage', 'Knockout')}


_group_rec = data.get('track_record', [])                  # frozen group-stage record
_ko_rec = ([r for _, fx in fixtures.iterrows() for r in [_ko_record(fx)] if r]
           if fixtures is not None and not fixtures.empty else [])
g_correct, g_total = sum(1 for r in _group_rec if r['correct']), len(_group_rec)
k_correct, k_total = sum(1 for r in _ko_rec if r['correct']), len(_ko_rec)
o_correct, o_total = g_correct + k_correct, g_total + k_total


# ----- Live (in-game) win probability: Bayesian update of pre-match odds -----
def _pois_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def live_odds(hs, as_, minutes_left, lh, la, maxg=10):
    """Updated win/draw/loss probs given the current score and minutes left.
    Models the REMAINING goals as Poisson (rate scaled by fraction of match left),
    so at minute 0 (0-0, 90 left) it returns the pre-match baseline."""
    frac = max(0.0, minutes_left) / 90.0
    rh, ra = lh * frac, la * frac
    hp = [_pois_pmf(i, rh) for i in range(maxg + 1)]
    ap = [_pois_pmf(j, ra) for j in range(maxg + 1)]
    ph = pd_ = pa = 0.0
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            p = hp[i] * ap[j]
            fh, fa = hs + i, as_ + j
            if fh > fa:
                ph += p
            elif fh == fa:
                pd_ += p
            else:
                pa += p
    return ph, pd_, pa


def estimate_minute(utc_kickoff, status):
    """Estimate match minute + minutes left from kickoff (API gives no live clock).
    Returns (minute, minutes_left); PAUSED is treated as halftime."""
    if status == 'PAUSED':
        return 45, 45
    try:
        kickoff = pd.to_datetime(utc_kickoff)
        wall = (pd.Timestamp.now(tz='UTC') - kickoff).total_seconds() / 60.0
    except Exception:
        return None, 45
    if wall <= 0:
        return 0, 90
    minute = wall if wall <= 45 else min(90, wall - 15)   # subtract ~15' halftime
    minute = max(0, min(90, minute))
    return int(round(minute)), max(0, 90 - minute)


def render_live_probs(home, away, hs, as_, mins_left):
    """Render the live win-probability lines for one match (live or preview).
    Display-only and self-contained — never feeds the model's accuracy/track record."""
    pred = pred_lookup.get(frozenset((home, away)))
    if not (pred and 'lh' in pred):
        st.caption("_Live model unavailable for this match._")
        return
    ohs, oas = (hs, as_) if home == pred['home'] else (as_, hs)   # orient to pred
    lp = live_odds(ohs, oas, mins_left, pred['lh'], pred['la'])
    if 'p_home_adv' in pred:        # knockout: binary (fold the draw 50/50 into advance)
        adv_h, adv_a = lp[0] + 0.5 * lp[1], lp[2] + 0.5 * lp[1]
        st.caption("Live chance to advance  ·  (pre-match)")
        for name, lv, bp in [(pred['home'], adv_h, pred['p_home_adv']),
                             (pred['away'], adv_a, pred['p_away_adv'])]:
            d = (lv - bp) * 100
            arrow = "🔺" if d > 0.5 else ("🔻" if d < -0.5 else "▪")
            st.write(f"{arrow} **{name}**  {lv:.0%}  _({bp:.0%})_")
        st.caption(f"_{lp[1]:.0%} chance still level → penalties_")
    else:
        base = (pred['p_home'], pred['p_draw'], pred['p_away'])
        st.caption("Live win probability  ·  (pre-match)")
        for name, lv, bp in [(pred['home'], lp[0], base[0]),
                             ("Draw", lp[1], base[1]),
                             (pred['away'], lp[2], base[2])]:
            d = (lv - bp) * 100
            arrow = "🔺" if d > 0.5 else ("🔻" if d < -0.5 else "▪")
            st.write(f"{arrow} **{name}**  {lv:.0%}  _({bp:.0%})_")


def _outcome(hs, as_):
    return 'home' if hs > as_ else ('away' if hs < as_ else 'draw')


def fixture_result(fx_row, live_info):
    """For a fixture + its live data, return (completed, pred_out, actual_out).
       completed=True only for finished games; pred_out is None if no model line."""
    home, away = fx_row['home'], fx_row['away']
    if not (live_info and live_info.get('home_score') is not None
            and live_info.get('status') in FINAL_STATUSES):
        return False, None, None      # only a FINISHED game is a result
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

    status = live_info.get('status', '') if live_info else ''
    is_live = status in LIVE_STATUSES
    is_final = status in FINAL_STATUSES
    # Only trust a score once the game is actually live or finished. The API can
    # attach a bogus fullTime score to a not-yet-started (TIMED) game — ignore it.
    has_score = (live_info and live_info.get('home_score') is not None
                 and (is_live or is_final))
    if has_score:
        # Orient the API score to the fixture's home/away
        if live_info.get('home') == home:
            hs, as_ = live_info['home_score'], live_info['away_score']
        else:
            hs, as_ = live_info['away_score'], live_info['home_score']
        st.markdown(f"### {home}  {int(hs)} – {int(as_)}  {away}")

        if is_live:
            st.caption(f"{stage} · {when} · 🔴 LIVE")
        else:  # finished
            line = f"{stage} · {when} · 🏁 FINAL"
            if pred and 'p_home_adv' in pred:        # knockout: grade by who advanced
                adv_pick = home if (ph + 0.5 * pd_) >= (pa + 0.5 * pd_) else away
                if hs > as_:
                    actual_adv = home
                elif as_ > hs:
                    actual_adv = away
                else:                                 # level -> use advancer flag (pens)
                    wf = live_info.get('winner')
                    actual_adv = (live_info.get('home') if wf == 'HOME_TEAM'
                                  else live_info.get('away') if wf == 'AWAY_TEAM' else None)
                if actual_adv:
                    line += "  ·  model ✅ called it" if adv_pick == actual_adv \
                            else "  ·  model ❌ missed"
            elif pred_out is not None:
                line += "  ·  model ✅ called it" if pred_out == _outcome(hs, as_) \
                        else "  ·  model ❌ missed"
            st.caption(line)
    else:
        st.markdown(f"**{home}**  vs  **{away}**")
        st.caption(f"{stage} · {when} · ⏳ scheduled")

    if show_pred and ph is not None:
        if pred and 'p_home_adv' in pred:        # knockout: binary (someone advances)
            adv_h, adv_a = ph + 0.5 * pd_, pa + 0.5 * pd_
            pick = home if adv_h >= adv_a else away
            st.caption(f"Model: **{home} {adv_h:.0%}** vs **{away} {adv_a:.0%}** to advance  ·  "
                       f"{pd_:.0%} chance of penalties  ·  pick: **{pick}**")
        else:
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

        # ---- Accuracy: overall (live) + group stage (frozen) + knockouts (live) ----
        # Uses the shared module-level record, so this matches the Track Record tab.
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall accuracy", f"{o_correct/o_total:.0%}" if o_total else "—")
        c2.metric("Group stage", f"{g_correct/g_total:.0%}" if g_total else "—")
        c3.metric("Knockouts", f"{k_correct/k_total:.0%}" if k_total else "—")
        st.caption(f"Overall **{o_correct}/{o_total}**  ·  Group stage {g_correct}/{g_total} (frozen)  "
                   f"·  Knockouts {k_correct}/{k_total} (R32 → Final)")
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
            if live_games:
                for lg in live_games:                       # one model per live game
                    with st.container(border=True):
                        st.markdown(f"### {lg['home']}  {lg['home_score']} – "
                                    f"{lg['away_score']}  {lg['away']}")
                        minute, mins_left = estimate_minute(lg.get('utcDate'), lg.get('status'))
                        tag = ("HT" if lg.get('status') == 'PAUSED'
                               else (f"~{minute}'" if minute is not None else "live"))
                        st.caption(f"🔴 {lg.get('status')} · {tag} (est.)")
                        render_live_probs(lg['home'], lg['away'],
                                          lg.get('home_score') or 0,
                                          lg.get('away_score') or 0, mins_left)
            else:
                # No live games -> preview the next kickoff(s). Replaced by the live
                # model automatically once a match starts.
                nxt = None
                if fixtures is not None and not fixtures.empty:
                    future = fixtures[fixtures['dt'] > pd.Timestamp.now(tz='UTC')]
                    if not future.empty:
                        nxt = future[future['dt'] == future['dt'].min()]
                if nxt is None or nxt.empty:
                    st.write("_No matches in progress, and no upcoming fixtures found._")
                else:
                    st.caption("No match live right now — previewing the next kickoff "
                               "(the live model takes over here once it starts).")
                    for _, fx in nxt.iterrows():             # show both if two kick off together
                        with st.container(border=True):
                            when = (fx['dt'].tz_convert(ET).strftime('%b %d, %I:%M %p ET')
                                    if pd.notna(fx['dt']) else fx['date'])
                            st.markdown(f"### {fx['home']}  vs  {fx['away']}")
                            st.caption(f"🕒 Preview · kicks off {when}")
                            render_live_probs(fx['home'], fx['away'], 0, 0, 90)

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
def build_bracket_html(bk, stage_odds):
    """Left-to-right knockout bracket that fills as real results come in.
    Confirmed qualifiers are solid, projections faded/italic, future slots TBD.
    Hover a team to light up its route to the final and show its stage odds."""
    import html as _html
    import json as _json
    winners = bk['winners']
    r32, order = bk['r32'], bk['r32_order']

    col0 = []          # (team, state, r32_match_id)
    for m in order:
        e, w = r32[str(m)], winners.get(str(m))
        for side in ('a', 'b'):
            t, conf = e[side], e[side + '_conf']
            if w and t != w:
                state = 'elim'
            elif w or conf:
                state = 'won'
            else:
                state = 'proj'
            col0.append((t, state, m))

    # later columns: each cell is the WINNER of a match id -> carries data-slot=id
    labels = ['Round of 16', 'Quarterfinals', 'Semifinals', 'Final', 'Champion']
    win_cols = []      # (header, [(team, state, match_id)])
    for idx, ids in enumerate(bk['columns']):
        champ = idx == len(bk['columns']) - 1
        cells = [(winners.get(str(m)), 'champ' if champ else ('won' if winners.get(str(m)) else 'tbd'), m)
                 for m in ids]
        win_cols.append((labels[idx], cells))

    def cell_html(team, state, slot, entrant=False):
        label = _html.escape(team) if team else 'TBD'
        attrs = f' data-{"eslot" if entrant else "slot"}="{slot}"'
        if team:
            attrs += f' data-team="{_html.escape(team)}" title="{_html.escape(team)}"'
        prefix = '★ ' if state == 'champ' else ''
        return f'<div class="bk-cell bk-{state}"{attrs}>{prefix}{label}</div>'

    cols_html = '<div class="bk-col"><div class="bk-h">Round of 32</div><div class="bk-col-inner">'
    cols_html += ''.join(cell_html(t, s, m, entrant=True) for t, s, m in col0)
    cols_html += '</div></div>'
    for header, cells in win_cols:
        inner = ''.join(cell_html(t, s, m) for t, s, m in cells)
        cols_html += (f'<div class="bk-col"><div class="bk-h">{header}</div>'
                      f'<div class="bk-col-inner">{inner}</div></div>')

    parent_js = _json.dumps(bk.get('parent', {}))
    odds_js = _json.dumps(stage_odds)
    return f"""<div class="bk-wrap">
<div class="bk-tip" id="bk-tip"></div>
<div class="bk-legend">
  <span><span class="lg bk-won">won</span> qualified / won</span>
  <span><span class="lg bk-proj">proj</span> model projection</span>
  <span><span class="lg bk-tbd">TBD</span> awaiting result</span>
  <span style="margin-left:auto;font-style:italic">hover a team for its route &amp; odds</span>
</div>
<div class="bk-cols">{cols_html}</div></div>
<style>
.bk-wrap{{position:relative;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:11.5px}}
.bk-tip{{position:absolute;z-index:5;display:none;background:#1c1917;color:#fafaf9;border:1px solid #44403c;padding:5px 9px;border-radius:6px;white-space:nowrap;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.25)}}
.bk-tip b{{font-weight:500}}
.bk-legend{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:10px;color:#78716c}}
.bk-legend .lg{{display:inline-block;padding:1px 8px;border-radius:4px;font-size:10.5px;border:1px solid transparent;margin-right:3px;vertical-align:middle}}
.bk-cols{{display:flex;align-items:stretch}}
.bk-col{{display:flex;flex-direction:column;flex:1;min-width:90px;padding:0 3px}}
.bk-h{{text-align:center;font-weight:500;margin-bottom:6px;color:#78716c}}
.bk-col-inner{{display:flex;flex-direction:column;justify-content:space-around;flex:1}}
.bk-cell{{height:20px;line-height:18px;padding:0 7px;margin:2px 0;border-radius:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent}}
.bk-won{{background:#f5f5f4;border-color:#d6d3cd;color:#1c1917;cursor:pointer}}
.bk-proj{{border:1px dashed #cfcabf;color:#78716c;font-style:italic;cursor:pointer}}
.bk-tbd{{border:1px dashed #e0ddd5;color:#a8a29e}}
.bk-elim{{color:#bcb8af;text-decoration:line-through}}
.bk-champ{{border:1px solid #d4960a;color:#b45309;text-align:center;font-weight:500}}
.bk-hl{{box-shadow:0 0 0 2px #2563eb;background:rgba(37,99,235,.12)}}
.bk-route{{box-shadow:0 0 0 2px #2563eb}}
@media (prefers-color-scheme:dark){{
 .bk-legend,.bk-h{{color:#a8a29e}}
 .bk-won{{background:#292524;border-color:#57534e;color:#e7e5e4}}
 .bk-proj{{border-color:#57534e;color:#a8a29e}}
 .bk-tbd{{border-color:#44403c;color:#78716c}}
 .bk-elim{{color:#57534e}}
 .bk-champ{{border-color:#f59e0b;color:#fbbf24}}
 .bk-hl{{box-shadow:0 0 0 2px #3b82f6;background:rgba(59,130,246,.16)}}
 .bk-route{{box-shadow:0 0 0 2px #3b82f6}}
}}
</style>
<script>
(function(){{
var PARENT={parent_js}, ODDS={odds_js};
var wrap=document.querySelector('.bk-wrap');
var tip=document.getElementById('bk-tip');
var cells=document.querySelectorAll('.bk-cell');
function pct(v){{return Math.round((v||0)*100)+'%';}}
function routeFrom(slot){{var c=[String(slot)];while(PARENT[slot]!=null){{slot=PARENT[slot];c.push(String(slot));}}return c;}}
function clearAll(){{cells.forEach(function(x){{x.classList.remove('bk-hl');x.classList.remove('bk-route');}});tip.style.display='none';}}
cells.forEach(function(el){{
 el.addEventListener('mouseenter',function(){{
  clearAll();
  el.classList.add('bk-hl');
  var start=el.getAttribute('data-eslot')||el.getAttribute('data-slot');
  if(start!=null){{
   routeFrom(start).forEach(function(s){{
    var w=document.querySelector('.bk-cell[data-slot="'+s+'"]');
    if(w&&w!==el) w.classList.add('bk-route');
   }});
  }}
  var team=el.getAttribute('data-team');
  if(team&&ODDS[team]){{
   var o=ODDS[team];
   tip.innerHTML='<b>'+team+'</b>&nbsp; R16 '+pct(o.R16)+' &middot; QF '+pct(o.QF)+
     ' &middot; SF '+pct(o.SF)+' &middot; Final '+pct(o.Final)+' &middot; win '+pct(o.Champion);
   tip.style.display='block';
   var top=el.offsetTop-tip.offsetHeight-6;
   if(top<0) top=el.offsetTop+el.offsetHeight+6;
   var left=el.offsetLeft+el.offsetWidth/2-tip.offsetWidth/2;
   var maxL=wrap.clientWidth-tip.offsetWidth-4;
   if(left<4) left=4;
   if(maxL>4&&left>maxL) left=maxL;
   tip.style.top=top+'px'; tip.style.left=left+'px';
  }}
 }});
 el.addEventListener('mouseleave',clearAll);
}});
}})();
</script>"""


with tab_runs:
    st.subheader("Run to the final")
    bk = data.get('bracket')
    if bk:
        bracket_h = len(bk['r32_order']) * 2 * 26 + 130
        components.html(build_bracket_html(bk, data['stage_odds']),
                        height=bracket_h, scrolling=False)
        st.caption("Solid = qualified or won · faded italic = model projection (firms up "
                   "as groups finish) · TBD fills in as knockout results come in.")
    else:
        st.info("Bracket not available yet — re-run predictor.py to generate it.")

    st.subheader("Probability of reaching each stage")
    stages = ['R32', 'R16', 'QF', 'SF', 'Final', 'Champion']
    sidx = {s: i for i, s in enumerate(stages)}
    prog = data.get('stage_progress', {})

    def stage_cell(team, s, val):
        p = prog.get(team)
        if p:
            r = p.get('reached')
            if r is not None and sidx[s] <= sidx[r]:
                return f'✅ {val * 100:.0f}%'
            if p.get('eliminated') and (r is None or sidx[s] > sidx[r]):
                return f'❌ {val * 100:.0f}%'
        return f'{val * 100:.0f}%'

    ordered = sorted(data['stage_odds'].items(),
                     key=lambda kv: kv[1].get('Champion', 0), reverse=True)
    rows = [{'Team': t, **{s: stage_cell(t, s, sd.get(s, 0)) for s in stages}}
            for t, sd in ordered]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True, height=600)
    st.caption("✅ reached (confirmed) · ❌ eliminated · otherwise model probability. "
               "Every % is a true probability from simulating the rest of the tournament "
               "10,000 times.")

# ===== TAB: Groups =====
with tab_groups:
    st.subheader("Group stage")
    cols = st.columns(3)
    for i, g in enumerate(sorted(data['groups'].keys())):
        with cols[i % 3]:
            st.markdown(f"### Group {g}")
            standings = pd.DataFrame(data['groups'][g])
            # Table shows REAL standings: actual points, then GD, then goals-for
            standings = standings.sort_values(
                ['pts', 'gd', 'gf'], ascending=False).reset_index(drop=True)

            def _mark(r):
                if r.get('clinched'):
                    return f"✅ {r['team']}"       # mathematically into the R32
                if r['advances']:
                    return f"▲ {r['team']}"        # projected to advance
                return f"　 {r['team']}"
            standings['team'] = standings.apply(_mark, axis=1)
            st.dataframe(
                standings[['team', 'gp', 'pts', 'gd', 'xpts']].rename(
                    columns={'team': 'Team', 'gp': 'GP', 'pts': 'Points',
                             'gd': 'GD', 'xpts': 'xPts'}),
                hide_index=True, use_container_width=True,
            )
            st.caption("✅ qualified (clinched) · ▲ projected to advance")
            with st.expander("Match predictions"):
                for m in data['group_matches'][g]:
                    st.write(f"**{m['home']}** {m['p_home']:.0%} · "
                             f"draw {m['p_draw']:.0%} · {m['p_away']:.0%} **{m['away']}**")

    # ----- Third-place race: the 8 best thirds reach the R32 -----
    st.divider()
    st.subheader("Third-place teams — best 8 advance")
    thirds = []
    for g in sorted(data['groups']):
        t = data['groups'][g][2]        # projected 3rd (group list is xPts-ordered)
        thirds.append({'Group': g, 'Team': t['team'], 'GP': t['gp'],
                       'Points': t['pts'], 'GD': t['gd'], 'xPts': t['xpts'],
                       '_gf': t['gf']})
    thirds.sort(key=lambda r: (r['xPts'], r['GD'], r['_gf']), reverse=True)
    for i, r in enumerate(thirds):
        r['Team'] = f"✅ {r['Team']}" if i < 8 else f"❌ {r['Team']}"
        del r['_gf']
    st.dataframe(
        pd.DataFrame(thirds)[['Group', 'Team', 'GP', 'Points', 'GD', 'xPts']],
        hide_index=True, use_container_width=True, height=460,
    )
    st.caption("The 8 best third-place teams reach the R32, ranked by xPts → GD → "
               "goals for. ✅ projected in · ❌ projected out. Re-ranks every refresh "
               "as standings (and who holds each group's 3rd slot) change.")

# ===== TAB: Track record =====
with tab_track:
    st.subheader("How the model has done")

    # --- Prediction record on completed games (group + knockouts, same as Game-by-Game) ---
    rec = _group_rec + _ko_rec
    if rec:
        c1, c2 = st.columns(2)
        c1.metric("Predictions correct", f"{o_correct}/{o_total}")
        c2.metric("Hit rate", f"{o_correct / o_total:.0%}" if o_total else "—")
        st.caption("The model's pre-match pick vs the actual result, for every completed game.")
        rows = [{'Date': r['date'], 'Match': f"{r['home']} {r['score']} {r['away']}",
                 'Predicted': r['pred'], 'Actual': r['actual'],
                 'Result': '✅' if r['correct'] else '❌', 'Stage': r['stage']}
                for r in sorted(rec, key=lambda x: x['date'], reverse=True)]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True, height=420)
    else:
        st.info("No completed games yet — the record fills in as games finish.")

    st.divider()

    # --- Championship odds over time ---
    st.subheader("Championship odds over time")
    hist = data.get('odds_history', [])
    if len(hist) >= 2:
        latest = hist[-1]['champ']
        top = [t for t, _ in sorted(latest.items(), key=lambda x: -x[1])[:6]]
        rows = []
        for snap in hist:
            row = {'Date': pd.to_datetime(snap['ts'])}
            for t in top:
                row[t] = round(snap['champ'].get(t, 0) * 100, 1)
            rows.append(row)
        chart_df = pd.DataFrame(rows).set_index('Date')
        st.line_chart(chart_df, height=360)
        st.caption("Title odds (%) for the current top 6, tracked over time as results came in.")
    else:
        st.info("Need at least two odds updates to chart a trend — this builds up "
                "as the tournament progresses.")

    st.divider()

    # --- Model updates / changelog (edit CHANGELOG.md — it renders here) ---
    _cl_path = _find('CHANGELOG.md')
    if _cl_path:
        with open(_cl_path, encoding='utf-8') as _f:
            st.markdown(_f.read())
    else:
        st.subheader("Model updates")
        st.caption("CHANGELOG.md not found.")


# ===== TAB: About =====
with tab_about:
    st.subheader("About this project")
    st.markdown("""
The goal of this dashboard is to predict the 2026 FIFA World Cup using a machine-learning model
I conceptualized and built from scratch, utilizing the help of AI, which then simulates the whole tournament thousands of times using the Monte Carlo simulations method.

**How it works**
- The concept of **team strength** is measured using Elo ratings, the team's recent form in games, as well as the team's squad quality (based on data regarding player performance for the top 23 players in the team).
- An **XGBoost** classifier turns those into win/draw/loss probabilities for each match, which I then calibrate! This means that when the model says 75%, the outcomes should really happen 75% of the time!
- A **Monte Carlo simulation** then plays the knockout bracket 10,000 times to produce championship and stage-by-stage odds. For instance, if a team wins a simulation 1,000 times against 10,000 total, it has a 10% probability for that scenario.
- It pulls live scores using football data api (football-data.org), grades each prediction against the real result, flags teams the moment they've mathematically clinched advancement to the next round, and re-computes odds as games finish.

**How good is it?**
**60% accuracy** so far on three-way (win/draw/loss) outcomes — vs 33% in theoretical probability for random guessing. It's shown live on the Game-by-Game tab and validated with leak-free backtesting on past World Cups (the model never sees the future).

**Skills I used in order to make this project possible**
Python · XGBoost · scikit-learn · pandas library · Streamlit (for dashboard) · football-data.org API (for live scores)

**AI-use disclosure!**
I built this as a learning project and used AI (Claude Opus 4.8 model) as a programming partner and teacher. Claude wrote most of this code, but only under my direction and vision. I specified what I had in mind, what each feature should do, as well as reviewing and testing the output, making sure I understood every change that was being made so that I could explain, debug, and extend the model. Beyond this model, Claude helped me by explaining ML concepts and pressure-testing ideas. With that being said, every modeling decision was mine to make and verify: I designed the approach, ran the backtests (with the AI's assistance for large scale backtests consisting of hundreds of games), and cut changes that didn't hold up (including a newer dataset that backtested worse results for this model). I utilized AI to speed up the build and teach me new concepts, the judgment and final decision was all mine.

**Known limitations**
The goal of this model is to try and predict outcomes, not specific scores (so the probability of a team winning doesn't necessarily correspond with the actual end score). That being said, I still have ideas for improvements on the horizon, so stay tuned!

Built by Krishna Vankayala - [GitHub repo](https://github.com/krishna-on-gh/fifa-worldcup-2026-predictor)

AI Assistance from: Claude Opus 4.8 (Anthropic)
    """)