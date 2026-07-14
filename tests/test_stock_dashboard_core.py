from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import stock_dashboard as dashboard


def technical_frame(closes=None):
    closes = closes or [100] * 10 + [110]
    return pd.DataFrame(
        {
            "Close": closes,
            "MA20": [95] * len(closes),
            "RSI": [65] * len(closes),
            "MACD": [2] * len(closes),
            "MACD_signal": [1] * len(closes),
            "Volume_momentum": [1.2] * len(closes),
        }
    )


def target_context():
    return {
        "latest_date": date(2026, 7, 10),
        "target_date": date(2026, 7, 13),
        "target_label": "US market closed; next session opens in 12h 0m",
    }


def test_clean_json_value_converts_nan_and_infinity_to_none():
    payload = {
        "normal": 10.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "items": [1, float("-inf")],
    }

    cleaned = dashboard.clean_json_value(payload)

    assert cleaned == {"normal": 10.5, "nan": None, "inf": None, "items": [1, None]}


def test_clamp_keeps_values_inside_bounds():
    assert dashboard.clamp(5, 0, 10) == 5
    assert dashboard.clamp(-1, 0, 10) == 0
    assert dashboard.clamp(11, 0, 10) == 10


def test_signal_style_returns_bullish_when_value_above_reference():
    text, color = dashboard.get_signal_style(105, 100)

    assert text == "Bullish"
    assert color == "#22c55e"


def test_signal_style_returns_bearish_when_value_not_above_reference():
    text, color = dashboard.get_signal_style(100, 100)

    assert text == "Bearish"
    assert color == "#ef4444"


def test_weekday_helpers_calculate_known_dates():
    assert dashboard.nth_weekday_of_month(2026, 1, 0, 3) == date(2026, 1, 19)
    assert dashboard.last_weekday_of_month(2026, 5, 0) == date(2026, 5, 25)


def test_observed_holiday_moves_weekend_dates():
    assert dashboard.observed_holiday(date(2026, 7, 4)) == date(2026, 7, 3)
    assert dashboard.observed_holiday(date(2027, 7, 4)) == date(2027, 7, 5)


def test_easter_date_matches_known_good_friday_source():
    assert dashboard.calculate_easter_date(2026) == date(2026, 4, 5)


def test_us_trading_day_rejects_weekends_and_holidays():
    assert dashboard.is_us_trading_day("2026-07-04") is False
    assert dashboard.is_us_trading_day("2026-07-03") is False
    assert dashboard.is_us_trading_day("2026-07-06") is True


def test_next_trading_day_skips_weekend():
    assert dashboard.get_next_trading_day(date(2026, 7, 10)) == date(2026, 7, 13)


def test_format_time_delta_outputs_hours_and_minutes():
    assert dashboard.format_time_delta(timedelta(hours=2, minutes=45, seconds=30)) == "2h 45m"
    assert dashboard.format_time_delta(timedelta(seconds=-5)) == "0h 0m"


def test_technical_sentiment_scores_strong_positive_case():
    score = dashboard.calculate_technical_sentiment(technical_frame())

    assert score == 95


def test_aggregate_news_sentiment_uses_neutral_fallback_and_average():
    assert dashboard.aggregate_news_sentiment([]) == 50.0

    score = dashboard.aggregate_news_sentiment(
        [{"compound": 0.6}, {"compound": -0.2}]
    )

    assert score == 60.0


def test_news_sentiment_summary_labels_positive_negative_and_neutral():
    assert dashboard.build_news_sentiment_summary([{"compound": 0.4}]).startswith("Positive")
    assert dashboard.build_news_sentiment_summary([{"compound": -0.4}]).startswith("Negative")
    assert dashboard.build_news_sentiment_summary([{"compound": 0.05}]).startswith("Neutral")


def test_prompt_news_summary_uses_latest_and_strongest_headlines():
    news_items = [
        {"headline": "Latest product launch lifts outlook"},
        {"headline": "Older neutral update"},
    ]
    scored_news = [
        {"headline": "Latest product launch lifts outlook", "compound": 0.2},
        {"headline": "Analyst downgrade pressures shares", "compound": -0.9},
    ]

    summary = dashboard.build_prompt_news_summary(news_items, scored_news)

    assert "Latest product launch lifts outlook" in summary
    assert "Analyst downgrade pressures shares" in summary


def test_parse_llm_response_accepts_valid_json_and_clamps_confidence():
    llm_text = '{"target_price": 123.45, "confidence": 1.5, "reason": "Strong setup"}'

    price, confidence, reason, error = dashboard.parse_llm_response(llm_text, 100)

    assert price == 123.45
    assert confidence == 1
    assert reason == "Strong setup"
    assert error is None


def test_parse_llm_response_falls_back_on_invalid_json():
    price, confidence, reason, error = dashboard.parse_llm_response("not-json", 100)

    assert price == 100
    assert confidence == 0.5
    assert reason == "No analysis available"
    assert error == "not-json"


def test_build_forecast_result_combines_model_prices(monkeypatch):
    monkeypatch.setattr(
        dashboard,
        "get_dynamic_blend_weights",
        lambda ticker, fallback: {
            "weight_xgb": 0.4,
            "weight_llm": 0.6,
            "source": "dynamic",
            "sample_count": 7,
            "mae_xgb": 2.0,
            "mae_llm": 1.0,
        },
    )

    result = dashboard.build_forecast_result(
        ticker="AAPL",
        current_price=100,
        pred_price=110,
        llm_price=120,
        llm_conf=0.9,
        llm_reason="test reason",
        target_context=target_context(),
    )

    assert result["ensemble_price"] == 116
    assert result["llm_conf"] == dashboard.MAX_BLEND_WEIGHT
    assert result["signal_text"] == "BUY"
    assert result["predicted_change_pct"] == 16
    assert result["predicted_date"] == "2026-07-13"


def test_price_forecast_falls_back_to_last_close_when_training_data_is_short():
    df = pd.DataFrame(
        {
            "Close": [100, 101],
            "MA20": [100, 100],
            "RSI": [50, 55],
            "Returns": [0.0, 0.01],
            "Volatility": [0.1, 0.1],
            "MACD": [0.1, 0.2],
            "MACD_signal": [0.1, 0.15],
            "BB_upper": [105, 106],
            "BB_lower": [95, 96],
            "Volume_momentum": [1.0, 1.1],
        }
    )

    assert dashboard.price_forecast(df, window=20) == 101


def test_load_price_data_drops_rows_with_missing_close(monkeypatch):
    raw_df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.0, 101.0, np.nan],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2026-07-01", periods=3, freq="D"),
    )
    monkeypatch.setattr(dashboard, "download_single_ticker_history", lambda symbol, period: raw_df)
    if hasattr(dashboard.load_price_data, "clear"):
        dashboard.load_price_data.clear()

    loaded = dashboard.load_price_data("AAPL", "1mo")

    assert len(loaded) == 2
    assert loaded["Close"].iloc[-1] == 101.0
