import streamlit as st


def get_theme(is_dark_mode):
    if is_dark_mode:
        return {
            "bg_style": (
                "radial-gradient(circle at 50% 30%, rgba(255,255,255,0.05), transparent 60%), "
                "radial-gradient(circle at center, #1e293b 0%, #020617 100%)"
            ),
            "sidebar_bg": "#020617",
            "text_color": "#ffffff",
            "muted_text_color": "#cbd5e1",
            "metric_bg": "rgba(255,255,255,0.05)",
            "card_bg": "rgba(255,255,255,0.06)",
            "card_border": "1px solid rgba(255,255,255,0.10)",
            "input_bg": "#0f172a",
            "input_border": "1px solid rgba(255,255,255,0.12)",
            "dropdown_bg": "#0f172a",
            "dropdown_hover_bg": "#1e293b",
            "dropdown_selected_bg": "#273449",
            "plotly_template": "plotly_dark",
            "grid_color": "rgba(255,255,255,0.10)",
        }

    return {
        "bg_style": (
            "radial-gradient(circle at 50% 30%, rgba(0,0,0,0.12), transparent 55%), "
            "radial-gradient(circle at center, #ffffff 0%, #cbd5e1 100%)"
        ),
        "sidebar_bg": "#ffffff",
        "text_color": "#000000",
        "muted_text_color": "#334155",
        "metric_bg": "#ffffff",
        "card_bg": "rgba(255,255,255,0.92)",
        "card_border": "1px solid rgba(15,23,42,0.08)",
        "input_bg": "#ffffff",
        "input_border": "1px solid rgba(15,23,42,0.16)",
        "dropdown_bg": "#ffffff",
        "dropdown_hover_bg": "#f1f5f9",
        "dropdown_selected_bg": "#e2e8f0",
        "plotly_template": "plotly_white",
        "grid_color": "rgba(0,0,0,0.10)",
    }


def apply_theme(theme):
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {theme["bg_style"]} !important;
        }}

        [data-testid="stSidebar"] {{
            background-color: {theme["sidebar_bg"]};
        }}

        .block-container {{
            padding-top: 2rem;
        }}

        [data-testid="stMetric"] {{
            background: {theme["metric_bg"]};
            padding: 15px;
            border-radius: 10px;
        }}

        h1, h2, h3, h4, h5, p, label, span, div {{
            color: {theme["text_color"]};
        }}

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: {theme["text_color"]} !important;
        }}

        [data-testid="stMetricValue"] div,
        button[data-baseweb="tab"] div {{
            color: {theme["text_color"]} !important;
        }}

        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox input {{
            background: {theme["input_bg"]} !important;
            border: {theme["input_border"]} !important;
            color: {theme["text_color"]} !important;
            -webkit-text-fill-color: {theme["text_color"]} !important;
        }}

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox input {{
            background: {theme["input_bg"]} !important;
            color: {theme["text_color"]} !important;
            -webkit-text-fill-color: {theme["text_color"]} !important;
        }}

        div[data-baseweb="popover"] ul {{
            background: {theme["dropdown_bg"]} !important;
            border: {theme["input_border"]} !important;
        }}

        div[data-baseweb="popover"] li {{
            background: {theme["dropdown_bg"]} !important;
            color: {theme["text_color"]} !important;
        }}

        div[data-baseweb="popover"] li:hover {{
            background: {theme["dropdown_hover_bg"]} !important;
        }}

        div[data-baseweb="popover"] li[aria-selected="true"] {{
            background: {theme["dropdown_selected_bg"]} !important;
            color: {theme["text_color"]} !important;
        }}

        .stButton > button p {{
            color: white !important;
            font-weight: 700 !important;
        }}

        .themed-card {{
            background: {theme["card_bg"]};
            border: {theme["card_border"]};
            border-radius: 15px;
        }}

        .signal-card {{
            background: {theme["card_bg"]};
            border: {theme["card_border"]};
            border-radius: 15px;
            text-align: center;
        }}

        .signal-card-title {{
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .signal-card-meta {{
            margin-top: 10px;
            font-size: 1rem;
            color: {theme["text_color"]};
        }}

        .signal-buy {{
            color: #22c55e !important;
        }}

        .signal-sell {{
            color: #ef4444 !important;
        }}

        .app-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            margin: 0 0 24px 0;
        }}

        .app-header-logo {{
            width: 86px;
            height: 86px;
            object-fit: contain;
            display: block;
        }}

        .app-header-title {{
            margin: 0;
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1;
            color: {theme["text_color"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
