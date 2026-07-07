import base64
from datetime import datetime

import streamlit as st


def render_app_header(logo_path, title):
    with open(logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode("utf-8")

    st.markdown(
        f"""
        <div class="app-header">
            <img class="app-header-logo" src="data:image/png;base64,{encoded_logo}" alt="Lupa logo">
            <h1 class="app-header-title">{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_value_card(title, value, signal_text, signal_color, theme, extra_text=None):
    st.markdown(
        f"""
        <div class="themed-card" style="padding: 20px; margin-top: 10px;">
            <p style="color:{theme["muted_text_color"]};">{title}</p>
            <h2>${value:.2f}</h2>
            <span style="color:{signal_color}; font-weight:600;">{signal_text}</span>
            {f'<p style="color:{theme["muted_text_color"]}; margin-top: 10px;">{extra_text}</p>' if extra_text else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(forecast_result):
    signal_class = "signal-buy" if forecast_result["signal_text"] == "BUY" else "signal-sell"
    arrow = "\u2191" if forecast_result["signal_text"] == "BUY" else "\u2193"
    reference_close_date = forecast_result.get("reference_close_date", "last completed")
    predicted_label = forecast_result.get("predicted_label", "")

    st.markdown(
        f"""
        <div class="signal-card" style="padding: 25px; margin-bottom: 10px;">
            <div class="signal-card-title {signal_class}">{arrow} {forecast_result["signal_text"]}</div>
            <div class="signal-card-meta">
                Forecast for {forecast_result["predicted_date"]} |
                {forecast_result["predicted_change_pct"]:+.2f}% vs {reference_close_date} close
            </div>
            <div class="signal-card-meta">{predicted_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_news_tab(symbol, news_items, scored_news, theme, news_limit):
    st.subheader(f"{symbol} News")

    scored_lookup = {item["headline"]: item for item in scored_news}

    for news_item in news_items[:news_limit]:
        headline = news_item.get("headline", "No title")
        url = news_item.get("url", "#")
        summary = news_item.get("summary", "")
        date = datetime.fromtimestamp(news_item.get("datetime", 0)).strftime("%Y-%m-%d")
        st.markdown(f"**[{headline}]({url})**")
        if headline in scored_lookup:
            sentiment = scored_lookup[headline]
            st.caption(
                f'FinBERT: {sentiment["label"].title()} | compound {sentiment["compound"]:+.2f}'
            )
        st.markdown(
            f'<div style="color:{theme["text_color"]}; white-space: pre-wrap;">{summary}</div>',
            unsafe_allow_html=True,
        )
        st.caption(date)
        st.divider()


def render_almanac_tab(almanac):
    st.header("Market Seasonality (Stock Trader's Almanac)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("January Barometer", almanac["jan_signal"])
    with col2:
        st.metric("First Five Days", almanac["five_signal"])
    with col3:
        st.metric("Best Six Months", almanac["best6"])
    st.subheader("Presidential Cycle")
    st.info(almanac["pres"])
