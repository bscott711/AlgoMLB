import streamlit as st

# --- Design Tokens ---
# HSL Colors (Saturation 60-80%, Lightness 40-70% for neon/premium feel)
COLORS = {
    "primary": "#3D5AFE",  # Indigo Accent
    "secondary": "#00E5FF",  # Cyan Accent
    "success": "#00E676",  # Green Accent
    "warning": "#FFEA00",  # Yellow Accent
    "danger": "#FF1744",  # Red Accent
    "background": "#0E1117",  # Streamlit Dark
    "surface": "#1E2227",  # Surface/Card background
    "text": "#E0E0E0",
    "text_muted": "#9E9E9E",
}

# --- CSS Design System ---
PREMIUM_CSS = f"""
<style>
    /* Global Background & Typography */
    .stApp {{
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(30, 34, 39, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }}

    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: {COLORS["secondary"]} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {COLORS["text_muted"]} !important;
    }}

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {{
        background-color: #05070A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Buttons & Interaction */
    .stButton>button {{
        border-radius: 8px;
        border: 1px solid {COLORS["primary"]};
        background-color: transparent;
        color: white;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {COLORS["primary"]};
        box-shadow: 0 0 15px {COLORS["primary"]}44;
    }}

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: {COLORS["text_muted"]};
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS["secondary"]} !important;
        border-bottom-color: {COLORS["secondary"]} !important;
    }}
</style>
"""


def apply_premium_styles():
    """Injects custom CSS into the Streamlit session."""
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


def get_plotly_template():
    """Returns a standardized Plotly theme for all charts."""
    return "plotly_dark"


def get_color_scale(metric: str = "default"):
    """Returns a semantic color scale for Plotly charts."""
    scales = {
        "default": ["#3D5AFE", "#00E5FF"],
        "hot": ["#FFEA00", "#FF1744"],
        "cold": ["#00E5FF", "#3D5AFE"],
        "diverging": "RdBu_r",
    }
    return scales.get(metric, scales["default"])
