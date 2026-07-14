import pandas as pd

from charts import build_heatmap_chart, build_price_chart, build_sentiment_gauge
from theme import get_theme


def sample_price_frame():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [103, 104, 105],
            "Low": [99, 100, 101],
            "Close": [102, 103, 104],
            "MA20": [101, 102, 103],
            "Volume": [1000, 1200, 1100],
        },
        index=index,
    )


def test_dark_theme_uses_dark_plotly_template():
    theme = get_theme(True)

    assert theme["plotly_template"] == "plotly_dark"
    assert theme["text_color"] == "#ffffff"
    assert theme["sidebar_bg"] == "#020617"


def test_light_theme_uses_light_plotly_template():
    theme = get_theme(False)

    assert theme["plotly_template"] == "plotly_white"
    assert theme["text_color"] == "#000000"
    assert theme["sidebar_bg"] == "#ffffff"


def test_sentiment_gauge_uses_sentiment_value_and_ranges():
    theme = get_theme(True)
    fig = build_sentiment_gauge(72.5, theme)

    assert fig.data[0].value == 72.5
    assert fig.data[0].gauge.axis.range == (0, 100)
    assert len(fig.data[0].gauge.steps) == 3
    assert fig.layout.height == 420


def test_price_chart_contains_price_ma_and_volume_traces():
    theme = get_theme(True)
    fig = build_price_chart(sample_price_frame(), theme)

    assert len(fig.data) == 3
    assert fig.data[0].type == "candlestick"
    assert fig.data[1].name == "MA20"
    assert fig.data[2].type == "bar"


def test_price_chart_enables_range_slider_and_pan():
    theme = get_theme(False)
    fig = build_price_chart(sample_price_frame(), theme)

    assert fig.layout.dragmode == "pan"
    assert fig.layout.xaxis.rangeslider.visible is True
    assert fig.layout.height == 650


def test_heatmap_chart_uses_ticker_change_data():
    theme = get_theme(True)
    heatmap_df = pd.DataFrame(
        {"Ticker": ["AAPL", "MSFT"], "Change": [1.25, -0.75]}
    )
    fig = build_heatmap_chart(heatmap_df, theme)

    assert fig.data[0].x.tolist() == ["AAPL", "MSFT"]
    assert fig.data[0].y.tolist() == [1.25, -0.75]
    assert fig.layout.height == 450
