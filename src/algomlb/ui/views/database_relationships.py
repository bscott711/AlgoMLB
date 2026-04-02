import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from algomlb.db.session import get_session_factory
from sqlalchemy import MetaData


def get_table_stats():
    metadata = MetaData()
    engine = get_session_factory().kw["bind"]
    metadata.reflect(bind=engine)

    stats = []
    with get_session_factory()() as session:
        for table_name in metadata.tables.keys():
            from sqlalchemy import text

            try:
                count = session.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
                stats.append({"Table": table_name, "Rows": count})
            except Exception:
                stats.append({"Table": table_name, "Rows": "N/A"})
    return pd.DataFrame(stats)


def get_table_sample(table_name, limit=5):
    engine = get_session_factory().kw["bind"]
    try:
        return pd.read_sql(f"SELECT * FROM {table_name} LIMIT {limit}", engine)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def get_table_metadata():
    # (existing function, slightly modified to match new data structures below)
    metadata = MetaData()
    engine = get_session_factory().kw["bind"]
    metadata.reflect(bind=engine)

    tables_info = []
    for table_name, table in metadata.tables.items():
        for column in table.columns:
            tables_info.append(
                {
                    "Table": table_name,
                    "Column": column.name,
                    "Type": str(column.type),
                    "PK": "🔑" if column.primary_key else "",
                    "FK": "🔗" if len(column.foreign_keys) > 0 else "",
                    "Nullable": "✅" if column.nullable else "❌",
                }
            )
    return pd.DataFrame(tables_info)


# Page Layout
st.set_page_config(layout="wide", page_title="AlgoMLB System Architecture")

# Custom Styles
st.markdown(
    """
<style>
    .main {
        background: #0f172a;
        color: #f8fafc;
    }
    .stMarkdown h1 {
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem !important;
    }
    .card {
        background: rgba(30, 41, 59, 0.4);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 0.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("System Architecture")

# Relationship Diagram
st.markdown("### 🏗️ Entity Relationships")
st.info(
    "The diagram below explains the data flow. You can use your browser zoom or right-click 'Open Image in New Tab' for high detail."
)

# Mermaid code with larger font and better spacing
mermaid_code = """
erDiagram
    %% SECTION 1: THE GAME ANCHOR
    GAME_RESULTS ||--o{ OPENMETEO_WEATHER_PROGRESSION : "hourly signals"
    GAME_RESULTS ||--o{ OPENMETEO_DAILY_FORECASTS : "daily forecasts"
    GAME_RESULTS ||--o{ LIVE_ODDS : "snapshot pricing"
    GAME_RESULTS ||--o{ HISTORICAL_ODDS : "settled outcomes"
    GAME_RESULTS ||--o{ PITCH_EVENTS : "Statcast telemetry"
    GAME_RESULTS ||--o{ RETROSHEET_EVENTS : "Retrosheet plays"
    GAME_RESULTS ||--o{ UMPIRE_SCORECARDS : "accuracy metrics"
    GAME_RESULTS ||--o{ BANKROLL_LEDGER : "bets placed"
    BALLPARKS ||--o{ GAME_RESULTS : "stadium context"

    %% SECTION 2: THE PLAYER & ML ANCHOR
    PLAYERS ||--o{ PLAYER_TRANSACTIONS : "injury history"
    PLAYERS ||--o{ HISTORICAL_DATA : "season stats"
    PLAYERS ||--o{ PLAYER_ROLLING_FEATURES : "predictive windowing"
    
    %% INTER-SECTION LINKS
    PLAYERS ||--o{ PITCH_EVENTS : "as performer"
    PLAYERS ||--o{ RETROSHEET_EVENTS : "as performer"

    GAME_RESULTS {
        string game_id PK
        int ballpark_id FK
        date game_date
    }
    OPENMETEO_WEATHER_PROGRESSION {
        string game_id PK
        float temp_t0_f
        float headwind_t0_mph
    }
    OPENMETEO_DAILY_FORECASTS {
        string game_id PK
        float temp_max_f
        float precip_sum_mm
    }
    PITCH_EVENTS {
        int id PK
        string game_id FK
        int pitcher_id FK
        int batter_id FK
    }
    PLAYER_TRANSACTIONS {
        string transaction_id PK
        int player_id FK
        string type_desc
    }
    PLAYER_ROLLING_FEATURES {
        int id PK
        int player_id FK
        string feature_name
    }
    HISTORICAL_DATA {
        int id PK
        int player_id FK
        float metric_value
    }
"""

components.html(
    f"""
    <div style="background: transparent; padding: 20px; overflow: auto; width: 100%; height: 1150px; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px;">
        <pre class="mermaid" style="background: transparent; width: max-content; margin: 0 auto;">
            {mermaid_code}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'dark',
            fontSize: 32,
            securityLevel: 'loose',
            er: {{ useMaxWidth: false }},
            flowchart: {{ curve: 'basis' }}
        }});
    </script>
    """,
    height=1200,
)

st.divider()

# Table Inspector
st.markdown("### 🔍 Intelligent Table Inspector")

try:
    df_meta = get_table_metadata()
    df_stats = get_table_stats()

    col_nav, col_details = st.columns([1, 4])

    with col_nav:
        selected_table = st.selectbox(
            "Select Entity", sorted(df_meta["Table"].unique())
        )
        row_count = df_stats[df_stats["Table"] == selected_table]["Rows"].values[0]
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #94a3b8;">TOTAL ROWS</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #60a5fa;">{row_count:,}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_details:
        tab_schema, tab_data = st.tabs(["📋 Schema Definition", "📁 Data Preview"])

        with tab_schema:
            table_detail = df_meta[df_meta["Table"] == selected_table].copy()
            st.dataframe(
                table_detail.drop(columns=["Table"]),
                use_container_width=True,
                hide_index=True,
            )

        with tab_data:
            st.markdown(f"Showing lastest 10 rows from `{selected_table}`")
            sample_df = get_table_sample(selected_table, limit=10)
            st.dataframe(sample_df, use_container_width=True)

except Exception as e:
    st.error(f"Error loading system metadata: {e}")

# Details Section
with st.expander("Architecture Decisions & Mapping Notes"):
    st.markdown("""
    - **Composite Keys**: Tables like `pitch_events` use composite keys (`game_id`, `at_bat`, `pitch_num`) for absolute uniqueness despite MLB's changing PK formats.
    - **Implicit Relationships**: Many relationships (like Player ID) are managed at the domain layer rather than strict SQL Foreign Keys to allow for "soft-links" with legacy Retrosheet data where PKs may differ from current MLB Stats API.
    - **Computed Columns**: `player_transactions` utilizes SQLite/Postgres computed columns for real-time IL duration calculation.
    """)
