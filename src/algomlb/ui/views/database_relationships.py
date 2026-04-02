import streamlit as st
import streamlit.components.v1 as components


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

# Details Section
with st.expander("Architecture Decisions & Mapping Notes"):
    st.markdown("""
    - **Composite Keys**: Tables like `pitch_events` use composite keys (`game_id`, `at_bat`, `pitch_num`) for absolute uniqueness despite MLB's changing PK formats.
    - **Implicit Relationships**: Many relationships (like Player ID) are managed at the domain layer rather than strict SQL Foreign Keys to allow for "soft-links" with legacy Retrosheet data where PKs may differ from current MLB Stats API.
    - **Computed Columns**: `player_transactions` utilizes SQLite/Postgres computed columns for real-time IL duration calculation.
    """)
