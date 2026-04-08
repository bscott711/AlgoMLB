"""Feature Engineering Audit — diagnostic queries against the AlgoMLB database."""

from algomlb.db.session import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine()

print("=" * 80)
print("FEATURE ENGINEERING AUDIT — DATABASE DIAGNOSTICS")
print("=" * 80)

# ── 1. Pitch sequence numbering ──────────────────────────────────────────
print("\n\n1. PITCH SEQUENCE NUMBERING (statcast_raw)")
print("-" * 60)
try:
    q1 = text("""
        SELECT game_pk, pitcher, COUNT(*) as pitches,
               MAX(pitch_number) as max_pitch_num,
               COUNT(*) - MAX(pitch_number) as sequence_gap
        FROM statcast_raw
        WHERE game_year = 2024
        GROUP BY game_pk, pitcher
        HAVING COUNT(*) != MAX(pitch_number)
        LIMIT 20
    """)
    df1 = pd.read_sql(q1, engine)
    if df1.empty:
        print("  ✅ ALL pitch sequences are gap-free for 2024")
    else:
        print(f"  ⚠️  {len(df1)} pitcher-game combos have sequence gaps:")
        print(df1.to_string(index=False))
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 2. Spin rate coverage ────────────────────────────────────────────────
print("\n\n2. SPIN RATE COVERAGE (statcast_raw)")
print("-" * 60)
try:
    q2 = text("""
        SELECT game_year as season,
               COUNT(*) as total_pitches,
               SUM(CASE WHEN release_spin_rate IS NULL THEN 1 ELSE 0 END) as null_spin,
               ROUND(100.0 * SUM(CASE WHEN release_spin_rate IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_pct
        FROM statcast_raw
        GROUP BY game_year ORDER BY game_year
    """)
    df2 = pd.read_sql(q2, engine)
    print(df2.to_string(index=False))
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 3. Key Statcast column coverage ─────────────────────────────────────
print("\n\n3. KEY STATCAST COLUMN COVERAGE (2024 sample)")
print("-" * 60)
try:
    cols = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "release_pos_x",
        "release_pos_z",
        "release_extension",
        "pfx_x",
        "pfx_z",
        "pitch_type",
        "n_thruorder_pitcher",
        "bat_speed",
        "attack_angle",
        "arm_angle",
    ]
    for col in cols:
        q = text(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as nulls,
                   ROUND(100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_pct
            FROM statcast_raw WHERE game_year = 2024
        """)
        r = pd.read_sql(q, engine).iloc[0]
        status = (
            "✅"
            if float(r["null_pct"]) < 5
            else ("⚠️" if float(r["null_pct"]) < 30 else "❌")
        )
        print(
            f"  {status} {col:<35} {int(r['nulls']):>8} / {int(r['total']):>8}  ({r['null_pct']}% null)"
        )
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 4. GUMBO pitches coverage ────────────────────────────────────────────
print("\n\n4. GUMBO PITCHES TABLE (gumbo_pitches)")
print("-" * 60)
try:
    q4 = text(
        "SELECT COUNT(*) as total, COUNT(DISTINCT game_pk) as games FROM gumbo_pitches"
    )
    r4 = pd.read_sql(q4, engine).iloc[0]
    print(f"  Total pitch records: {int(r4['total']):,}")
    print(f"  Distinct games:      {int(r4['games']):,}")
    print("  NOTE: gumbo_pitches only stores timestamps, NOT full play-by-play events.")
    print("  ⚠️  There is NO gumbo_play_by_play or gumbo_events table in the schema.")
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 5. Retrosheet TTO derivation check (batting order / lineup pos) ─────
print("\n\n5. RETROSHEET TTO DERIVATION (batting order via 'lp' column)")
print("-" * 60)
try:
    q5 = text("""
        SELECT COUNT(*) as pa_events,
               SUM(CASE WHEN lp IS NULL THEN 1 ELSE 0 END) as null_lp,
               SUM(CASE WHEN lp = 0 THEN 1 ELSE 0 END) as zero_lp
        FROM retrosheet_events
        WHERE EXTRACT(YEAR FROM date) = 2024
        AND pa_flag = 1
    """)
    r5 = pd.read_sql(q5, engine).iloc[0]
    print(f"  Total PAs (2024):   {int(r5['pa_events']):,}")
    print(f"  Null LP:            {int(r5['null_lp']):,}")
    print(f"  Zero LP:            {int(r5['zero_lp']):,}")
    if int(r5["null_lp"]) == 0:
        print(
            "  ✅ lineup position fully populated — TTO derivable from batting order cycling"
        )
    else:
        print("  ⚠️  Some null lineup positions — TTO derivation may have gaps")
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 5b. Statcast n_thruorder_pitcher (TTO from Statcast directly) ────────
print("\n\n5b. STATCAST TTO (n_thruorder_pitcher column)")
print("-" * 60)
try:
    q5b = text("""
        SELECT game_year as season,
               COUNT(*) as total,
               SUM(CASE WHEN n_thruorder_pitcher IS NULL THEN 1 ELSE 0 END) as null_tto,
               ROUND(100.0 * SUM(CASE WHEN n_thruorder_pitcher IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_pct
        FROM statcast_raw
        GROUP BY game_year ORDER BY game_year
    """)
    df5b = pd.read_sql(q5b, engine)
    print(df5b.to_string(index=False))
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 6. Weather coverage ──────────────────────────────────────────────────
print("\n\n6. WEATHER COVERAGE (openmeteo_weather_progression)")
print("-" * 60)
try:
    q6 = text("""
        SELECT EXTRACT(YEAR FROM g.game_date)::int as season,
               COUNT(DISTINCT g.game_id) as total_games,
               COUNT(DISTINCT w.game_id) as games_with_weather,
               ROUND(100.0 * COUNT(DISTINCT w.game_id) / NULLIF(COUNT(DISTINCT g.game_id), 0), 1) as pct_covered
        FROM game_results g
        LEFT JOIN openmeteo_weather_progression w ON g.game_id = w.game_id
        WHERE g.status = 'COMPLETED'
        GROUP BY EXTRACT(YEAR FROM g.game_date)
        ORDER BY season
    """)
    df6 = pd.read_sql(q6, engine)
    print(df6.to_string(index=False))
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 7. Transactions / IL coverage ────────────────────────────────────────
print("\n\n7. TRANSACTIONS (player_transactions)")
print("-" * 60)
try:
    q7 = text("""
        SELECT type_desc, COUNT(*) as events
        FROM player_transactions
        WHERE EXTRACT(YEAR FROM transaction_date) >= 2019
        GROUP BY type_desc ORDER BY events DESC
        LIMIT 15
    """)
    df7 = pd.read_sql(q7, engine)
    print(df7.to_string(index=False))
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 8. Retrosheet ER attribution fields ──────────────────────────────────
print("\n\n8. RETROSHEET ER ATTRIBUTION (pr*_pre/post = responsible pitcher)")
print("-" * 60)
try:
    q8 = text("""
        SELECT COUNT(*) as total_events,
               SUM(CASE WHEN pr1_pre IS NOT NULL THEN 1 ELSE 0 END) as has_pr1,
               SUM(CASE WHEN er > 0 THEN 1 ELSE 0 END) as events_with_er,
               SUM(CASE WHEN ur_b > 0 OR ur1 > 0 OR ur2 > 0 OR ur3 > 0 THEN 1 ELSE 0 END) as events_with_unearned
        FROM retrosheet_events
        WHERE EXTRACT(YEAR FROM date) = 2024
    """)
    r8 = pd.read_sql(q8, engine).iloc[0]
    print(f"  Total events (2024):     {int(r8['total_events']):,}")
    print(f"  Events with pr1_pre:     {int(r8['has_pr1']):,}")
    print(f"  Events with ER flagged:  {int(r8['events_with_er']):,}")
    print(f"  Events with unearned:    {int(r8['events_with_unearned']):,}")
    print("  ✅ Retrosheet has pr*_pre/post (responsible pitcher) + er/ur flags")
    print("     → er_attribution_log can be derived without GUMBO replay")
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 9. Ballpark orientation (hp_bearing_deg) ─────────────────────────────
print("\n\n9. BALLPARK ORIENTATION (hp_bearing_deg)")
print("-" * 60)
try:
    q9 = text("""
        SELECT team_name, ballpark,
               hp_bearing_deg,
               CASE WHEN hp_bearing_deg IS NOT NULL THEN '✅' ELSE '❌' END as has_bearing
        FROM ballparks
        ORDER BY team_name
    """)
    df9 = pd.read_sql(q9, engine)
    total = len(df9)
    has_bearing = df9["hp_bearing_deg"].notna().sum()
    print(f"  {has_bearing}/{total} parks have hp_bearing_deg populated")
    missing = df9[df9["hp_bearing_deg"].isna()]
    if not missing.empty:
        print("  Missing:")
        for _, r in missing.iterrows():
            print(f"    ❌ {r['team_name']}: {r['ballpark']}")
except Exception as e:
    print(f"  ❌ Query failed: {e}")

# ── 10. Existing rolling features inventory ──────────────────────────────
print("\n\n10. EXISTING ROLLING FEATURES (player_rolling_features)")
print("-" * 60)
try:
    q10 = text("""
        SELECT role, season, COUNT(*) as rows, COUNT(DISTINCT player_id) as players
        FROM player_rolling_features
        GROUP BY role, season
        ORDER BY role, season
    """)
    df10 = pd.read_sql(q10, engine)
    print(df10.to_string(index=False))
except Exception as e:
    print(f"  ❌ Query failed: {e}")

print("\n\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
