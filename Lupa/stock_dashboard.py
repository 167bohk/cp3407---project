import json
import os
import re
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import finnhub
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI
from xgboost import XGBRegressor

from charts import build_heatmap_chart, build_price_chart, build_sentiment_gauge
from theme import apply_theme, get_theme
from ui_components import (
    render_almanac_tab,
    render_app_header,
    render_news_tab,
    render_signal_card,
    render_value_card,
)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception


# ---------- App Setup: page metadata, API clients, shared constants ----------

finnhub_client = None
openai_client = None

BIG_TECHS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AMD"]
MAX_HEATMAP_TICKERS = 20
PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y", "5y"]
FORECAST_STATE_KEY = "forecast_result"
VALUATION_STATE_KEY = "valuation_result"
MAX_VALUATION_PEERS = 8
VALUATION_METRICS = {
    "trailing_pe": "Trailing P/E",
    "forward_pe": "Forward P/E",
    "price_to_book": "Price / Book",
    "price_to_sales": "Price / Sales",
    "enterprise_to_ebitda": "EV / EBITDA",
}
US_MARKET_TZ = ZoneInfo("America/New_York")
FINBERT_MIN_AVAILABLE_MB = 900
ENABLE_FINBERT_DEFAULT = False
MARKET_CLOSE_STABILIZATION_HOURS = 2
MAX_DYNAMIC_BLEND_AGE_DAYS = 30
MIN_BLEND_WEIGHT = 0.2
MAX_BLEND_WEIGHT = 0.8
NEWS_HEADLINE_LIMIT = 10
TECHNICAL_SENTIMENT_WEIGHT = 0.7
NEWS_SENTIMENT_WEIGHT = 0.3
PREDICTION_LOG_PATH = os.path.join(os.path.dirname(__file__), "llm_prediction_log.csv")
PREDICTION_LOG_COLUMNS = [
    "ticker",
    "created_at",
    "target_date",
    "reference_close_date",
    "reference_close_price",
    "xgb_pred_price",
    "llm_pred_price",
    "llm_conf",
    "ensemble_price",
    "weight_xgb_used",
    "weight_llm_used",
    "actual_close",
    "xgb_abs_error",
    "llm_abs_error",
    "ensemble_abs_error",
    "status",
]


# ---------- DataFrame factories ----------


def empty_prediction_log_df():
    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)


# ---------- Session State: initialize and reset derived forecast output ----------

def clear_forecast_state():
    st.session_state.pop(FORECAST_STATE_KEY, None)
    st.session_state.pop(VALUATION_STATE_KEY, None)
    st.session_state.pop("valuation_peers", None)


def initialize_session_state():
    st.session_state.setdefault("ticker", "AAPL")
    st.session_state.setdefault("ticker_search", None)


def on_ticker_search_changed():
    selected_ticker = st.session_state.get("ticker_search")
    if selected_ticker:
        st.session_state.ticker = selected_ticker.strip().upper()
    clear_forecast_state()


def get_supabase_config():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def clean_json_value(value):
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def supabase_request(method, path, payload=None):
    config = get_supabase_config()
    if config is None:
        return None

    request = Request(
        f'{config["url"]}{path}',
        method=method,
        headers={
            "apikey": config["key"],
            "Authorization": f'Bearer {config["key"]}',
            "Content-Type": "application/json",
        },
        data=None if payload is None else json.dumps(clean_json_value(payload), allow_nan=False).encode("utf-8"),
    )
    with urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def load_prediction_log():
    supabase_config = get_supabase_config()
    if supabase_config is not None:
        records = supabase_request(
            "GET",
            "/rest/v1/prediction_log?select=*&order=created_at.desc",
        )
        if not records:
            return empty_prediction_log_df()
        return pd.DataFrame(records)

    if not os.path.exists(PREDICTION_LOG_PATH):
        return empty_prediction_log_df()
    return pd.read_csv(PREDICTION_LOG_PATH)


def prediction_record_exists(ticker, target_date, reference_close_date):
    supabase_config = get_supabase_config()
    if supabase_config is not None:
        existing = supabase_request(
            "GET",
            "/rest/v1/prediction_log?select=id"
            f"&ticker=eq.{quote(ticker)}"
            f"&target_date=eq.{quote(target_date)}"
            f"&reference_close_date=eq.{quote(reference_close_date)}"
            "&limit=1",
        )
        return bool(existing)

    log_df = load_prediction_log()
    if log_df.empty:
        return False

    existing = log_df[
        (log_df["ticker"] == ticker)
        & (log_df["target_date"] == target_date)
        & (log_df["reference_close_date"] == reference_close_date)
    ]
    return not existing.empty


def append_prediction_log_record(ticker, reference_close_price, forecast_result):
    target_date = forecast_result["predicted_date"]
    reference_close_date = forecast_result["reference_close_date"]

    if prediction_record_exists(ticker, target_date, reference_close_date):
        return "duplicate", None

    row_dict = {
        "ticker": ticker,
        "created_at": datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S"),
        "target_date": target_date,
        "reference_close_date": reference_close_date,
        "reference_close_price": float(reference_close_price),
        "xgb_pred_price": float(forecast_result["pred_price"]),
        "llm_pred_price": float(forecast_result["llm_price"]),
        "llm_conf": float(forecast_result["llm_conf"]),
        "ensemble_price": float(forecast_result["ensemble_price"]),
        "weight_xgb_used": float(forecast_result["weight_xgb"]),
        "weight_llm_used": float(forecast_result["weight_llm"]),
        "actual_close": None,
        "xgb_abs_error": None,
        "llm_abs_error": None,
        "ensemble_abs_error": None,
        "status": "pending",
    }

    supabase_config = get_supabase_config()
    if supabase_config is not None:
        try:
            supabase_request("POST", "/rest/v1/prediction_log", [row_dict])
            return "supabase", None
        except HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = exc.reason
            return "csv", f"Supabase insert failed ({exc.code}): {error_body}"
        except (URLError, TimeoutError, ValueError) as exc:
            return "csv", f"Supabase insert failed: {exc}"

    row = pd.DataFrame([row_dict], columns=PREDICTION_LOG_COLUMNS)
    if os.path.exists(PREDICTION_LOG_PATH):
        row.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(PREDICTION_LOG_PATH, index=False)
    return "csv", None


def normalize_download_history(df, ticker):
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    return df


def download_ticker_history_safe(ticker, period=None, start=None, end=None):
    download_kwargs = {
        "progress": False,
        "auto_adjust": False,
    }
    history_kwargs = {"auto_adjust": False}

    if period is not None:
        download_kwargs["period"] = period
        history_kwargs["period"] = period
    else:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
        history_kwargs["start"] = start
        history_kwargs["end"] = end

    try:
        df = yf.download(ticker, **download_kwargs)
    except YFRateLimitError:
        raise
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        return df

    try:
        return yf.Ticker(ticker).history(**history_kwargs)
    except YFRateLimitError:
        raise
    except Exception:
        return pd.DataFrame()


def fetch_actual_close_map_for_ticker(ticker, target_dates):
    if not target_dates:
        return {}

    parsed_dates = [
        datetime.strptime(str(target_date), "%Y-%m-%d").date()
        for target_date in target_dates
    ]
    start = (min(parsed_dates) - timedelta(days=2)).strftime("%Y-%m-%d")
    end = (max(parsed_dates) + timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        df = download_ticker_history_safe(ticker, start=start, end=end)
    except YFRateLimitError:
        return {}

    df = normalize_download_history(df, ticker)
    if df.empty:
        return {}

    close_map = {}
    for ts, row in df.iterrows():
        close_map[pd.Timestamp(ts).date().isoformat()] = float(row["Close"])

    return close_map


def update_actual_closes_in_log():
    log_df = load_prediction_log()
    if log_df.empty:
        return {"updated": 0, "skipped": 0}

    supabase_config = get_supabase_config()
    pending_df = log_df[log_df["status"].fillna("pending") != "completed"].copy()
    updated = 0
    skipped = 0

    for ticker, ticker_rows in pending_df.groupby("ticker"):
        target_dates = [str(target_date) for target_date in ticker_rows["target_date"].tolist()]
        close_map = fetch_actual_close_map_for_ticker(ticker, target_dates)

        for _, row in ticker_rows.iterrows():
            record_id = row.get("id")
            target_date = str(row["target_date"])
            actual_close = close_map.get(target_date)
            if actual_close is None:
                skipped += 1
                continue

            xgb_abs_error = abs(float(row["xgb_pred_price"]) - actual_close)
            llm_abs_error = abs(float(row["llm_pred_price"]) - actual_close)
            ensemble_abs_error = abs(float(row["ensemble_price"]) - actual_close)

            update_payload = {
                "actual_close": actual_close,
                "xgb_abs_error": xgb_abs_error,
                "llm_abs_error": llm_abs_error,
                "ensemble_abs_error": ensemble_abs_error,
                "status": "completed",
            }

            if supabase_config is not None and pd.notna(record_id):
                supabase_request(
                    "PATCH",
                    f"/rest/v1/prediction_log?id=eq.{int(record_id)}",
                    update_payload,
                )
            else:
                mask = (
                    (log_df["ticker"] == ticker)
                    & (log_df["target_date"] == target_date)
                    & (log_df["reference_close_date"] == row["reference_close_date"])
                )
                for key, value in update_payload.items():
                    log_df.loc[mask, key] = value
            updated += 1

    if supabase_config is None and updated > 0:
        log_df.to_csv(PREDICTION_LOG_PATH, index=False)

    return {"updated": updated, "skipped": skipped}


def get_dynamic_blend_weights(ticker, fallback_llm_weight, min_samples=5, window=10):
    fallback_llm_weight = min(max(float(fallback_llm_weight), MIN_BLEND_WEIGHT), MAX_BLEND_WEIGHT)
    fallback_xgb_weight = 1 - fallback_llm_weight
    default_result = {
        "weight_xgb": fallback_xgb_weight,
        "weight_llm": fallback_llm_weight,
        "source": "default",
        "sample_count": 0,
        "mae_xgb": None,
        "mae_llm": None,
    }

    log_df = load_prediction_log()
    if log_df.empty:
        return default_result

    ticker_df = log_df[log_df["ticker"] == ticker].copy()
    if ticker_df.empty:
        return default_result

    ticker_df = ticker_df[ticker_df["status"].fillna("pending") == "completed"].copy()
    if ticker_df.empty:
        return default_result

    ticker_df["target_date"] = pd.to_datetime(ticker_df["target_date"], errors="coerce")
    ticker_df["created_at"] = pd.to_datetime(ticker_df["created_at"], errors="coerce")
    ticker_df["xgb_abs_error"] = pd.to_numeric(ticker_df["xgb_abs_error"], errors="coerce")
    ticker_df["llm_abs_error"] = pd.to_numeric(ticker_df["llm_abs_error"], errors="coerce")
    cutoff_date = pd.Timestamp(datetime.now(ZoneInfo("Asia/Singapore")).date() - timedelta(days=MAX_DYNAMIC_BLEND_AGE_DAYS))
    ticker_df = ticker_df.dropna(subset=["target_date", "xgb_abs_error", "llm_abs_error"])
    ticker_df = ticker_df[ticker_df["target_date"] >= cutoff_date].sort_values("target_date")
    if ticker_df.empty:
        return default_result

    recent_df = ticker_df.tail(window)
    sample_count = len(recent_df)

    if sample_count < min_samples:
        return {
            "weight_xgb": fallback_xgb_weight,
            "weight_llm": fallback_llm_weight,
            "source": "default",
            "sample_count": sample_count,
            "mae_xgb": None,
            "mae_llm": None,
        }

    mae_xgb = float(recent_df["xgb_abs_error"].mean())
    mae_llm = float(recent_df["llm_abs_error"].mean())

    raw_xgb = 1 / (mae_xgb + 1e-6)
    raw_llm = 1 / (mae_llm + 1e-6)
    total = raw_xgb + raw_llm
    weight_xgb = raw_xgb / total
    weight_llm = raw_llm / total

    weight_xgb = max(MIN_BLEND_WEIGHT, min(MAX_BLEND_WEIGHT, weight_xgb))
    weight_llm = 1 - weight_xgb

    return {
        "weight_xgb": weight_xgb,
        "weight_llm": weight_llm,
        "source": "dynamic",
        "sample_count": sample_count,
        "mae_xgb": mae_xgb,
        "mae_llm": mae_llm,
    }


# ---------- Shared Helpers: small reusable calculations and formatters ----------

def get_signal_style(value, reference):
    if value > reference:
        return "Bullish", "#22c55e"
    return "Bearish", "#ef4444"


def nth_weekday_of_month(year, month, weekday, occurrence):
    first_day = datetime(year, month, 1).date()
    offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + (occurrence - 1) * 7)


def last_weekday_of_month(year, month, weekday):
    if month == 12:
        next_month = datetime(year + 1, 1, 1).date()
    else:
        next_month = datetime(year, month + 1, 1).date()
    current = next_month - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def observed_holiday(holiday_date):
    if holiday_date.weekday() == 5:
        return holiday_date - timedelta(days=1)
    if holiday_date.weekday() == 6:
        return holiday_date + timedelta(days=1)
    return holiday_date


def calculate_easter_date(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day).date()


@st.cache_data(ttl=86400)
def get_us_market_holidays_for_year(year):
    holidays = {
        observed_holiday(datetime(year, 1, 1).date()),
        nth_weekday_of_month(year, 1, 0, 3),   # MLK Day
        nth_weekday_of_month(year, 2, 0, 3),   # Presidents' Day
        calculate_easter_date(year) - timedelta(days=2),  # Good Friday
        last_weekday_of_month(year, 5, 0),     # Memorial Day
        observed_holiday(datetime(year, 6, 19).date()),   # Juneteenth
        observed_holiday(datetime(year, 7, 4).date()),    # Independence Day
        nth_weekday_of_month(year, 9, 0, 1),   # Labor Day
        nth_weekday_of_month(year, 11, 3, 4),  # Thanksgiving
        observed_holiday(datetime(year, 12, 25).date()),  # Christmas
    }
    return {holiday.isoformat() for holiday in holidays}


def is_us_trading_day(check_date_str):
    check_date = datetime.strptime(check_date_str, "%Y-%m-%d").date()
    if check_date.weekday() >= 5:
        return False

    return check_date.isoformat() not in get_us_market_holidays_for_year(check_date.year)


def get_next_trading_day(base_date):
    next_day = base_date + timedelta(days=1)
    while not is_us_trading_day(next_day.strftime("%Y-%m-%d")):
        next_day += timedelta(days=1)
    return next_day


def format_time_delta(delta):
    total_seconds = max(int(delta.total_seconds()), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def has_stable_completed_close(market_now):
    market_close = market_now.replace(hour=16, minute=0, second=0, microsecond=0)
    stable_after = market_close + timedelta(hours=MARKET_CLOSE_STABILIZATION_HOURS)
    return market_now >= stable_after


def get_prediction_target_context(latest_trading_timestamp):
    latest_date = pd.Timestamp(latest_trading_timestamp).date()
    market_now = datetime.now(US_MARKET_TZ)
    market_date = market_now.date()
    market_open = market_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = market_now.replace(hour=16, minute=0, second=0, microsecond=0)
    market_is_open_day = is_us_trading_day(market_date.strftime("%Y-%m-%d"))
    stable_completed_close = has_stable_completed_close(market_now)

    if market_is_open_day and market_open <= market_now < market_close:
        target_date = market_date if market_date > latest_date else latest_date
        target_label = f"US market open; closes in {format_time_delta(market_close - market_now)}"
    elif market_is_open_day and market_now < market_open:
        target_date = market_date if market_date > latest_date else latest_date
        target_label = f"US market not open yet; opens in {format_time_delta(market_open - market_now)}"
    elif market_is_open_day and not stable_completed_close:
        target_date = market_date if market_date > latest_date else latest_date
        stable_after = market_close + timedelta(hours=MARKET_CLOSE_STABILIZATION_HOURS)
        target_label = f"US market closed; waiting {format_time_delta(stable_after - market_now)} for stable close data"
    else:
        reference_date = market_date if market_date > latest_date else latest_date
        target_date = get_next_trading_day(reference_date)
        next_open = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            9,
            30,
            tzinfo=US_MARKET_TZ,
        )
        target_label = f"US market closed; next session opens in {format_time_delta(next_open - market_now)}"

    return {
        "latest_date": latest_date,
        "target_date": target_date,
        "target_label": target_label,
        "market_now": market_now,
    }


def keep_completed_market_data(df):
    if df.empty:
        return df

    market_now = datetime.now(US_MARKET_TZ)
    market_date = market_now.date()
    last_row_date = pd.Timestamp(df.index[-1]).date()
    stable_completed_close = has_stable_completed_close(market_now)

    if market_date.weekday() < 5 and not stable_completed_close and last_row_date == market_date:
        completed_df = df.iloc[:-1]
        if not completed_df.empty:
            return completed_df

    return df


def coerce_series(values):
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def has_enough_memory_for_finbert(min_available_mb=FINBERT_MIN_AVAILABLE_MB):
    if psutil is None:
        return True

    available_mb = psutil.virtual_memory().available / 1024 / 1024
    return available_mb >= min_available_mb


def is_finbert_enabled():
    try:
        configured_value = st.secrets.get("ENABLE_FINBERT", os.getenv("ENABLE_FINBERT", ENABLE_FINBERT_DEFAULT))
    except Exception:
        configured_value = os.getenv("ENABLE_FINBERT", ENABLE_FINBERT_DEFAULT)

    return str(configured_value).lower() in {"1", "true", "yes", "on"}


def configure_huggingface_token():
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = None

    if hf_token:
        os.environ["HF_TOKEN"] = str(hf_token)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = str(hf_token)


# ---------- Data Layer: market history, news, and seasonality inputs ----------

@st.cache_data(ttl=180)
def download_single_ticker_history(symbol, period):
    try:
        return download_ticker_history_safe(symbol, period=period)
    except YFRateLimitError:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def download_multi_ticker_history(tickers, period):
    return yf.download(list(tickers), period=period, progress=False, auto_adjust=False, group_by="ticker")


@st.cache_data(ttl=900)
def load_price_data(symbol, period):
    df = download_single_ticker_history(symbol, period)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol]
        elif symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return df

    df["MA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * df["BB_std"]
    df["BB_lower"] = df["MA20"] - 2 * df["BB_std"]

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_momentum"] = df["Volume"] / df["Volume_MA20"]
    return df


@st.cache_data(ttl=600)
def get_news(symbol):
    today = datetime.today().strftime("%Y-%m-%d")
    last_week = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        return finnhub_client.company_news(symbol, _from=last_week, to=today)
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_almanac_signals():
    spy = download_single_ticker_history("SPY", "2y")

    if spy.empty:
        return {
            "jan_signal": "Neutral",
            "five_signal": "Neutral",
            "best6": "Neutral Season",
            "pres": "Unknown",
        }

    jan = spy[spy.index.month == 1]

    jan_signal = "Neutral"
    if len(jan) > 5:
        jan_close = coerce_series(jan["Close"])
        jan_return = float((jan_close.iloc[-1] / jan_close.iloc[0]) - 1)
        jan_signal = "Bullish" if jan_return > 0 else "Bearish"

    five_signal = "Neutral"
    jan5 = jan.head(5)
    if len(jan5) == 5:
        jan5_close = coerce_series(jan5["Close"])
        jan5_return = float((jan5_close.iloc[-1] / jan5_close.iloc[0]) - 1)
        five_signal = "Bullish" if jan5_return > 0 else "Bearish"

    current_month = datetime.now().month
    best6 = "Bullish Season" if current_month in [11, 12, 1, 2, 3, 4] else "Weak Season"

    year = datetime.now().year
    cycle = year % 4
    if cycle == 0:
        pres = "Election Year"
    elif cycle == 1:
        pres = "Post Election"
    elif cycle == 2:
        pres = "Midterm Weakness"
    else:
        pres = "Pre Election Bullish"

    return {
        "jan_signal": jan_signal,
        "five_signal": five_signal,
        "best6": best6,
        "pres": pres,
    }


@st.cache_resource
def load_finbert():
    if not is_finbert_enabled():
        return None, None

    configure_huggingface_token()

    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None, None

    if not has_enough_memory_for_finbert():
        return None, None

    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def score_text_with_finbert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0].tolist()

    labels = ["positive", "negative", "neutral"]
    scores = dict(zip(labels, probabilities))
    compound = scores["positive"] - scores["negative"]

    return {
        "label": max(scores, key=scores.get),
        "scores": scores,
        "compound": compound,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def score_news_with_finbert(headlines):
    tokenizer, model = load_finbert()
    if tokenizer is None or model is None:
        return []

    scored_items = []
    for headline in headlines:
        stripped = headline.strip()
        if not stripped:
            continue

        sentiment = score_text_with_finbert(stripped, tokenizer, model)
        scored_items.append(
            {
                "headline": stripped,
                "label": sentiment["label"],
                "compound": float(sentiment["compound"]),
            }
        )

    return scored_items


def calculate_technical_sentiment(df):
    price = df["Close"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_signal"].iloc[-1]
    ret_5d = df["Close"].pct_change(5).iloc[-1]
    volume_momentum = df["Volume_momentum"].iloc[-1]

    score = 50.0
    score += 10 if price > ma20 else -10
    score += 10 if rsi > 60 else -10 if rsi < 40 else 0
    score += 10 if macd > macd_signal else -10
    score += clamp(ret_5d * 100, -10, 10) if pd.notna(ret_5d) else 0
    score += 5 if volume_momentum > 1.1 else -5 if volume_momentum < 0.9 else 0

    return clamp(score, 0, 100)


def aggregate_news_sentiment(scored_news):
    if not scored_news:
        return 50.0

    average_compound = sum(item["compound"] for item in scored_news) / len(scored_news)
    return clamp((average_compound + 1) * 50, 0, 100)


def build_news_sentiment_summary(scored_news):
    if not scored_news:
        if psutil is not None and not has_enough_memory_for_finbert():
            return "FinBERT disabled due to memory limits; using technical signals only."
        return "FinBERT unavailable; using headlines without structured sentiment score."

    average_compound = sum(item["compound"] for item in scored_news) / len(scored_news)
    overall_label = "Positive" if average_compound > 0.1 else "Negative" if average_compound < -0.1 else "Neutral"
    return f"{overall_label} ({average_compound:+.2f} compound from recent headlines)"


def build_prompt_news_summary(news_items, scored_news):
    headlines = []

    latest_headline = next((item.get("headline", "").strip() for item in news_items if item.get("headline")), "")
    if latest_headline:
        headlines.append(latest_headline[:120])

    if scored_news:
        strongest_item = max(scored_news, key=lambda item: abs(item["compound"]))
        strongest_headline = strongest_item["headline"].strip()
        if strongest_headline and strongest_headline != latest_headline:
            headlines.append(strongest_headline[:120])
    else:
        fallback_headline = next(
            (
                item.get("headline", "").strip()
                for item in news_items[1:]
                if item.get("headline") and item.get("headline").strip() != latest_headline
            ),
            "",
        )
        if fallback_headline:
            headlines.append(fallback_headline[:120])

    return " | ".join(headlines) if headlines else "No significant recent news."


# ---------- Forecasting: XGBoost model, LLM prompt, and ensemble output ----------

def train_model(X, y):
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def price_forecast(df, window=20):
    feature_columns = [
        "Close",
        "MA20",
        "RSI",
        "Returns",
        "Volatility",
        "MACD",
        "MACD_signal",
        "BB_upper",
        "BB_lower",
        "Volume_momentum",
    ]

    training_df = df.tail(350).dropna()
    if len(training_df) <= window:
        return float(df["Close"].iloc[-1])

    feature_matrix = training_df[feature_columns].values

    X = []
    y = []
    for index in range(window, len(feature_matrix)):
        X.append(feature_matrix[index - window : index].flatten())
        y.append(feature_matrix[index][0])

    X = np.array(X)
    y = np.array(y)

    model = train_model(X, y)
    last_window = feature_matrix[-window:].flatten().reshape(1, -1)
    return float(model.predict(last_window)[0])



def build_llm_prompt(symbol, price, trend, df, news_summary, news_sentiment_summary, almanac, target_context):
    reference_close_date = target_context["latest_date"]
    target_date = target_context["target_date"]
    market_status = target_context["target_label"]

    return f"""
    You are a professional quantitative hedge fund analyst.

    [DATA]
    Stock: {symbol}
    Reference close date: {reference_close_date}
    Reference close price: {price}
    Target close date: {target_date}
    US market status: {market_status}

    RSI: {df['RSI'].iloc[-1]:.2f}
    Volatility: {df['Volatility'].iloc[-1]:.2%}
    Trend (MA20): {trend}

    Recent News Headlines:
    {news_summary}

    News Sentiment:
    {news_sentiment_summary}

    Almanac and seasonality signals (weak context only):
    - January Barometer: {almanac["jan_signal"]} (January direction signal)
    - First 5 Trading Days: {almanac["five_signal"]} (early-year momentum signal)
    - Best 6 Months: {almanac["best6"]} (seasonal strength signal)
    - Presidential Cycle: {almanac["pres"]} (election-cycle context)

    [INSTRUCTIONS]

    1. Predict the CLOSE price for the target US trading session.

    - Reference close date: {reference_close_date}
    - Reference close price: {price}
    - Target close date: {target_date}
    - target_price MUST be the closing price of the target date

    2. Provide:
    - target_price: realistic closing price (within +/-10%)
    - confidence: 0 to 1

    3. Use:
    - technical indicators
    - news sentiment
    - almanac/seasonality only as weak supporting context

    4. Rules:
    - Do NOT predict intraday high/low
    - Do NOT output a price range
    - Output ONE single closing price

    5. Be decisive

    [OUTPUT FORMAT - JSON ONLY]
    {{
      "target_price": 210.5,
      "confidence": 0.72,
      "reason": "max 15 sentences"
    }}
    """


@st.cache_data(ttl=600)
def run_llm(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_llm_response(llm_text, fallback_price):
    try:
        llm_data = json.loads(llm_text)
        llm_price = float(llm_data.get("target_price", fallback_price) or fallback_price)
        llm_conf = float(llm_data.get("confidence", 0.5) or 0.5)
        llm_conf = min(max(llm_conf, 0), 1)
        llm_reason = (llm_data.get("reason") or "No reasoning provided")[:2000]
        return llm_price, llm_conf, llm_reason, None
    except Exception:
        return fallback_price, 0.5, "No analysis available", llm_text


def build_forecast_result(ticker, current_price, pred_price, llm_price, llm_conf, llm_reason, target_context):
    llm_conf = min(max(llm_conf, MIN_BLEND_WEIGHT), MAX_BLEND_WEIGHT)
    blend_info = get_dynamic_blend_weights(ticker, llm_conf)
    weight_xgb = blend_info["weight_xgb"]
    weight_llm = blend_info["weight_llm"]
    ensemble_price = (pred_price * weight_xgb) + (llm_price * weight_llm)

    return {
        "ensemble_price": ensemble_price,
        "llm_price": llm_price,
        "pred_price": pred_price,
        "llm_reason": llm_reason,
        "llm_conf": llm_conf,
        "weight_xgb": weight_xgb,
        "weight_llm": weight_llm,
        "weight_source": blend_info["source"],
        "weight_sample_count": blend_info["sample_count"],
        "mae_xgb": blend_info["mae_xgb"],
        "mae_llm": blend_info["mae_llm"],
        "signal_text": "BUY" if ensemble_price > current_price else "SELL",
        "predicted_change_pct": ((ensemble_price - current_price) / current_price) * 100,
        "reference_close_date": target_context["latest_date"].strftime("%Y-%m-%d"),
        "predicted_date": target_context["target_date"].strftime("%Y-%m-%d"),
        "predicted_label": target_context["target_label"],
    }


# ---------- Tab data helpers ----------

def parse_custom_tickers(raw_tickers):
    tickers = []
    invalid_tickers = []
    for value in re.split(r"[\s,;]+", raw_tickers.strip().upper()):
        if not value:
            continue
        if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,14}", value):
            invalid_tickers.append(value)
        elif value not in tickers:
            tickers.append(value)
    return tickers, invalid_tickers


def build_heatmap_ticker_list(raw_tickers, include_big_tech=True):
    custom_tickers, invalid_tickers = parse_custom_tickers(raw_tickers)
    tickers = list(BIG_TECHS) if include_big_tech else []
    tickers.extend(ticker for ticker in custom_tickers if ticker not in tickers)
    omitted_count = max(0, len(tickers) - MAX_HEATMAP_TICKERS)
    return tickers[:MAX_HEATMAP_TICKERS], invalid_tickers, omitted_count


def positive_finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


@st.cache_data(ttl=3600, show_spinner=False)
def load_valuation_metrics(symbol):
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        return {"ticker": symbol, "error": str(exc)}

    return {
        "ticker": symbol,
        "current_price": positive_finite_number(
            info.get("currentPrice") or info.get("regularMarketPrice")
        ),
        "trailing_pe": positive_finite_number(info.get("trailingPE")),
        "forward_pe": positive_finite_number(info.get("forwardPE")),
        "price_to_book": positive_finite_number(info.get("priceToBook")),
        "price_to_sales": positive_finite_number(info.get("priceToSalesTrailing12Months")),
        "enterprise_to_ebitda": positive_finite_number(info.get("enterpriseToEbitda")),
        "earnings_growth": info.get("earningsGrowth"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


def analyze_relative_valuation(company_metrics, peer_metrics):
    comparisons = []
    signals = []
    implied_prices = []
    current_price = positive_finite_number(company_metrics.get("current_price"))

    for metric_key, metric_label in VALUATION_METRICS.items():
        company_value = positive_finite_number(company_metrics.get(metric_key))
        peer_values = [
            value
            for peer in peer_metrics
            if (value := positive_finite_number(peer.get(metric_key))) is not None
        ]
        if company_value is None or len(peer_values) < 2:
            continue

        peer_median = float(np.median(peer_values))
        ratio = company_value / peer_median
        if ratio < 0.8:
            signal = -1
            signal_label = "Below peers"
        elif ratio > 1.2:
            signal = 1
            signal_label = "Above peers"
        else:
            signal = 0
            signal_label = "Near peers"

        comparisons.append(
            {
                "Metric": metric_label,
                "Company": company_value,
                "Peer median": peer_median,
                "Difference": (ratio - 1) * 100,
                "Signal": signal_label,
            }
        )
        signals.append(signal)
        if current_price is not None:
            implied_prices.append(current_price * peer_median / company_value)

    if len(signals) < 3:
        label = "Insufficient Data"
        score = None
    else:
        score = float(np.mean(signals))
        if score <= -0.35:
            label = "Potentially Undervalued"
        elif score >= 0.35:
            label = "Potentially Overvalued"
        else:
            label = "Fairly Valued"

    fair_value_low = None
    fair_value_high = None
    fair_value_mid = None
    if implied_prices:
        fair_value_low = float(np.percentile(implied_prices, 25))
        fair_value_high = float(np.percentile(implied_prices, 75))
        fair_value_mid = float(np.median(implied_prices))

    return {
        "label": label,
        "score": score,
        "valid_metric_count": len(signals),
        "comparisons": comparisons,
        "fair_value_low": fair_value_low,
        "fair_value_high": fair_value_high,
        "fair_value_mid": fair_value_mid,
    }


def load_heatmap_data(tickers):
    tickers = tuple(tickers)
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Change"])

    batch_data = download_multi_ticker_history(tickers, "5d")
    if batch_data.empty:
        return pd.DataFrame(columns=["Ticker", "Change"])

    rows = []
    for ticker in tickers:
        try:
            if isinstance(batch_data.columns, pd.MultiIndex):
                try:
                    ticker_frame = batch_data[ticker]
                except KeyError:
                    ticker_frame = batch_data.xs(ticker, axis=1, level=-1)
                close = coerce_series(ticker_frame["Close"])
            else:
                close = coerce_series(batch_data["Close"])
            close = pd.to_numeric(close, errors="coerce").dropna()
            if len(close) < 2 or close.iloc[0] == 0:
                continue
            change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            rows.append({"Ticker": ticker, "Change": float(change)})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------- Main Page: assemble sidebar, load data, and render dashboard ----------

def main():
    global finnhub_client, openai_client

    st.set_page_config(
        page_title="Lupa AI Stock Terminal",
        layout="wide",
        page_icon=":chart_with_upwards_trend:",
        initial_sidebar_state="expanded",
    )

    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    initialize_session_state()

    dark_mode = st.sidebar.toggle("Night Mode", value=True)
    theme = get_theme(dark_mode)
    apply_theme(theme)

    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    render_app_header(logo_path, "Lupa AI Stock Terminal")

    st.sidebar.selectbox(
        "Ticker",
        BIG_TECHS,
        index=None,
        key="ticker_search",
        placeholder="Enter any ticker (e.g. AAPL)",
        accept_new_options=True,
        filter_mode="prefix",
        on_change=on_ticker_search_changed,
    )
    period = st.sidebar.selectbox("Analysis Window", PERIOD_OPTIONS, index=2)

    symbol = st.session_state.ticker.upper()
    raw_df = load_price_data(symbol, period)
    df = keep_completed_market_data(raw_df)

    if df.empty:
        st.error("Ticker not found or latest completed close is not available yet")
        st.stop()

    news = get_news(symbol)
    almanac = get_almanac_signals()
    headline_list = [item.get("headline", "") for item in news[:NEWS_HEADLINE_LIMIT] if item.get("headline")]
    scored_news = score_news_with_finbert(headline_list)

    price = df["Close"].iloc[-1]
    ret = df["Returns"].iloc[-1]
    trend = "Bullish" if price > df["MA20"].iloc[-1] else "Bearish"
    technical_sentiment = calculate_technical_sentiment(df)
    news_sentiment = aggregate_news_sentiment(scored_news)
    sentiment = (
        TECHNICAL_SENTIMENT_WEIGHT * technical_sentiment
        + NEWS_SENTIMENT_WEIGHT * news_sentiment
    )
    pred_price = price_forecast(df)
    target_context = get_prediction_target_context(df.index[-1])

    news_sentiment_summary = build_news_sentiment_summary(scored_news)
    news_summary = build_prompt_news_summary(news, scored_news)

    st.markdown(f"## {symbol} Market Overview")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Price", f"${price:.2f}", f"{ret:.2%}")
    with metric_col2:
        st.metric("Trend", trend)
    with metric_col3:
        st.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2%}")
    with metric_col4:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

    _, sentiment_center, _ = st.columns([1, 2, 1])
    with sentiment_center:
        st.plotly_chart(build_sentiment_gauge(sentiment, theme), width="content")
        with st.expander("How Market Sentiment Is Calculated"):
            st.markdown(
                f"""
                `Market Sentiment` is a composite 0-100 score:

                - `70%` Technical sentiment
                - `30%` FinBERT news sentiment

                Current breakdown:

                - Technical sentiment: `{technical_sentiment:.1f}`
                - News sentiment: `{news_sentiment:.1f}`
                - Final score: `{sentiment:.1f}`

                Technical sentiment is derived from:
                - Price vs `MA20`
                - `RSI`
                - `MACD` vs signal line
                - 5-day return
                - Volume momentum

                News sentiment is derived from recent headlines scored by `FinBERT`.
                """
            )

    tab_chart, tab_short_term, tab_valuation, tab_almanac, tab_heat, tab_news = st.tabs(
        ["Chart", "Short-Term Forecast", "Valuation", "Almanac", "Heatmap", "News"]
    )

    with tab_chart:
        st.plotly_chart(
            build_price_chart(df, theme),
            width="stretch",
            config={"scrollZoom": True},
        )

    with tab_short_term:
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Short-Term Price Forecast")
            st.caption(f'Prediction horizon: next US trading session ({target_context["target_date"]:%Y-%m-%d})')
            signal_text, signal_color = get_signal_style(pred_price, price)
            render_value_card("Predicted Price", pred_price, signal_text, signal_color, theme)

        llm_prompt = build_llm_prompt(
            symbol,
            price,
            trend,
            df,
            news_summary,
            news_sentiment_summary,
            almanac,
            target_context,
        )
        run_llm_clicked = False

        with right_col:
            st.markdown('<div style="height: 150px;"></div>', unsafe_allow_html=True)
            _, button_col, _ = st.columns([1, 2, 1])
            with button_col:
                run_llm_clicked = st.button(
                    "Run LLM Analysis",
                    key="llm_button",
                    width="stretch",
                )

        if run_llm_clicked:
            try:
                update_actual_closes_in_log()
            except YFRateLimitError:
                pass
            llm_text = run_llm(llm_prompt)
            llm_price, llm_conf, llm_reason, llm_parse_error = parse_llm_response(llm_text, price)

            if llm_parse_error is not None:
                st.error("LLM parsing failed")
                st.write(llm_parse_error)

            st.session_state[FORECAST_STATE_KEY] = build_forecast_result(
                ticker=symbol,
                current_price=price,
                pred_price=pred_price,
                llm_price=llm_price,
                llm_conf=llm_conf,
                llm_reason=llm_reason,
                target_context=target_context,
            )
            record_status, record_error = append_prediction_log_record(
                ticker=symbol,
                reference_close_price=price,
                forecast_result=st.session_state[FORECAST_STATE_KEY],
            )
            if record_status == "supabase":
                st.caption("Prediction logged to Supabase for dynamic weighting.")
            elif record_status == "csv":
                st.caption("Prediction logged locally for dynamic weighting.")
                if record_error:
                    st.caption(record_error)
            else:
                st.caption("Prediction for this ticker and target date is already logged.")

        forecast_result = st.session_state.get(FORECAST_STATE_KEY)
        if forecast_result:
            forecast_result.setdefault("reference_close_date", "last completed")
            forecast_result.setdefault("predicted_label", "")
            st.markdown("### LLM Analysis")
            st.markdown(
                f"""
                <div class="themed-card" style="
                    padding: 15px;
                    border-radius: 10px;
                    font-size: 15px;
                    line-height: 1.6;
                    margin-bottom: 10px;
                ">
                    {forecast_result["llm_reason"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_signal_card(forecast_result)

            for title, value in [
                ("Ensemble Price", forecast_result["ensemble_price"]),
                ("LLM Price", forecast_result["llm_price"]),
                ("XGBoost Price", forecast_result["pred_price"]),
            ]:
                signal_text, signal_color = get_signal_style(value, price)
                extra_text = None
                if title == "Ensemble Price":
                    blend_label = "Dynamic blend" if forecast_result.get("weight_source") == "dynamic" else "Default blend"
                    extra_text = (
                        f'{blend_label}: XGBoost {forecast_result["weight_xgb"]:.0%} '
                        f'+ LLM {forecast_result["weight_llm"]:.0%}'
                    )
                    if forecast_result.get("weight_source") == "dynamic":
                        extra_text += f' | based on {forecast_result.get("weight_sample_count", 0)} completed runs'
                    else:
                        extra_text += f' | waiting for {max(0, 5 - forecast_result.get("weight_sample_count", 0))} more completed runs'
                    if forecast_result.get("mae_xgb") is not None and forecast_result.get("mae_llm") is not None:
                        extra_text += (
                            f' | XGB MAE: ${forecast_result["mae_xgb"]:.2f}'
                            f' | LLM MAE: ${forecast_result["mae_llm"]:.2f}'
                        )
                if title == "LLM Price":
                    extra_text = f'Confidence: {forecast_result["llm_conf"]:.0%}'
                render_value_card(title, value, signal_text, signal_color, theme, extra_text=extra_text)

    with tab_valuation:
        st.subheader(f"{symbol} Valuation Analysis")
        st.caption(
            "Compare valuation multiples with similar companies. Use peers from the same industry for a more meaningful result."
        )

        default_peers = ", ".join(ticker for ticker in BIG_TECHS if ticker != symbol)
        peer_input = st.text_input(
            "Peer tickers",
            value=default_peers,
            key="valuation_peers",
            placeholder="Enter peer tickers, separated by commas (e.g. MSFT, GOOGL, META)",
        )
        peer_tickers, invalid_peers = parse_custom_tickers(peer_input)
        peer_tickers = [ticker for ticker in peer_tickers if ticker != symbol]
        omitted_peers = max(0, len(peer_tickers) - MAX_VALUATION_PEERS)
        peer_tickers = peer_tickers[:MAX_VALUATION_PEERS]

        if invalid_peers:
            st.warning(f"Invalid ticker format: {', '.join(invalid_peers)}")
        if omitted_peers:
            st.warning(f"Only the first {MAX_VALUATION_PEERS} peer tickers are used.")

        if st.button("Run Valuation Analysis", key="valuation_button", type="primary"):
            if len(peer_tickers) < 2:
                st.error("Add at least two valid peer tickers.")
            else:
                with st.spinner("Loading company and peer valuation data..."):
                    company_metrics = load_valuation_metrics(symbol)
                    if company_metrics.get("current_price") is None:
                        company_metrics["current_price"] = float(price)
                    peer_metrics = [load_valuation_metrics(ticker) for ticker in peer_tickers]
                    analysis = analyze_relative_valuation(company_metrics, peer_metrics)
                    st.session_state[VALUATION_STATE_KEY] = {
                        "ticker": symbol,
                        "peers": peer_tickers,
                        "company": company_metrics,
                        "analysis": analysis,
                    }

        valuation_result = st.session_state.get(VALUATION_STATE_KEY)
        if valuation_result and valuation_result.get("ticker") == symbol:
            analysis = valuation_result["analysis"]
            company_metrics = valuation_result["company"]
            result_col, range_col, gap_col = st.columns(3)
            result_col.metric("Valuation", analysis["label"])

            if analysis["fair_value_low"] is not None:
                range_col.metric(
                    "Peer-Implied Fair Value",
                    f'${analysis["fair_value_low"]:.2f} - ${analysis["fair_value_high"]:.2f}',
                )
                valuation_gap = (
                    (company_metrics["current_price"] - analysis["fair_value_mid"])
                    / analysis["fair_value_mid"]
                )
                gap_col.metric("Price vs Midpoint", f"{valuation_gap:+.1%}")
            else:
                range_col.metric("Peer-Implied Fair Value", "Unavailable")
                gap_col.metric("Valid Metrics", analysis["valid_metric_count"])

            if analysis["comparisons"]:
                comparison_df = pd.DataFrame(analysis["comparisons"])
                st.dataframe(
                    comparison_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Company": st.column_config.NumberColumn(format="%.2f"),
                        "Peer median": st.column_config.NumberColumn(format="%.2f"),
                        "Difference": st.column_config.NumberColumn(format="%+.1f%%"),
                    },
                )
            else:
                st.info("Not enough comparable valuation data is available for these tickers.")

            with st.expander("How This Valuation Is Calculated"):
                st.markdown(
                    """
                    The result compares positive, available valuation multiples with the peer median.

                    - More than 20% below peers: undervaluation signal
                    - Within 20% of peers: neutral signal
                    - More than 20% above peers: overvaluation signal
                    - At least three valid metrics are required for a classification

                    The fair-value range is implied by peer multiples and is not a price forecast or investment recommendation.
                    """
                )

    with tab_heat:
        control_col, toggle_col = st.columns([3, 1])
        with control_col:
            custom_heatmap_tickers = st.text_input(
                "Custom stocks",
                key="heatmap_custom_tickers",
                placeholder="Enter tickers, separated by commas (e.g. NFLX, JPM)",
            )
        with toggle_col:
            include_big_tech = st.toggle("Include Big Tech", value=True)

        heatmap_tickers, invalid_tickers, omitted_count = build_heatmap_ticker_list(
            custom_heatmap_tickers,
            include_big_tech,
        )
        if invalid_tickers:
            st.warning(f"Invalid ticker format: {', '.join(invalid_tickers)}")
        if omitted_count:
            st.warning(f"Only the first {MAX_HEATMAP_TICKERS} tickers are shown.")

        heatmap_df = load_heatmap_data(heatmap_tickers)
        if not heatmap_df.empty:
            st.plotly_chart(build_heatmap_chart(heatmap_df, theme), width="stretch")
            unavailable_tickers = [
                ticker for ticker in heatmap_tickers
                if ticker not in set(heatmap_df["Ticker"])
            ]
            if unavailable_tickers:
                st.caption(f"No recent price data: {', '.join(unavailable_tickers)}")
        elif not heatmap_tickers:
            st.info("Add at least one custom stock or include Big Tech.")
        else:
            st.info("Heatmap data is temporarily unavailable.")

    with tab_news:
        render_news_tab(symbol, news, scored_news, theme, news_limit=NEWS_HEADLINE_LIMIT)

    with tab_almanac:
        render_almanac_tab(almanac)


if __name__ == "__main__":
    main()
