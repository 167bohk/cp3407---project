# Lupa AI Stock Terminal Testing Plan

## Textbook Testing Approach

Chapter 7 of *Head First Software Development* explains that testing should be planned from several views of the system:

- **Black-box testing:** test the system from the user's view by checking inputs, outputs, user stories, validation, error conditions, state transitions, and boundary cases.
- **Grey-box testing:** inspect data saved or generated inside the system, such as prediction logs, timestamps, and records sent to external systems.
- **White-box testing:** exercise branches and internal functions such as sentiment scoring, date calculations, and fallback logic.
- **Automation and CI:** automate repeatable tests so the build fails when important behavior breaks.

The textbook example around page 242 uses user stories as the basis for test cases. Following that idea, the test cases below are tied directly to completed GitHub issues from the project's closed issue list:

- [#1 Website framework](https://github.com/167bohk/cp3407---project/issues/1)
- [#2 Light and Dark mode](https://github.com/167bohk/cp3407---project/issues/2)
- [#3 XGBoost prediction model](https://github.com/167bohk/cp3407---project/issues/3)
- [#13 Trading date and market timing logic](https://github.com/167bohk/cp3407---project/issues/13)
- [#14 Prediction record storage and backtesting support](https://github.com/167bohk/cp3407---project/issues/14)
- [#15 Dynamic prediction weighting and fallback](https://github.com/167bohk/cp3407---project/issues/15)
- [#16 Final ensemble prediction](https://github.com/167bohk/cp3407---project/issues/16)
- [#17 LLM market prediction](https://github.com/167bohk/cp3407---project/issues/17)
- [#18 Market sentiment score](https://github.com/167bohk/cp3407---project/issues/18)
- [#19 News and sentiment analysis](https://github.com/167bohk/cp3407---project/issues/19)
- [#20 Price and market visualisation](https://github.com/167bohk/cp3407---project/issues/20)
- [#21 Stock data loading and stability handling](https://github.com/167bohk/cp3407---project/issues/21)

## Selected User Stories and Test Cases

### User Story 1: LLM Market Prediction (#17)

Use OpenAI to generate an AI-based market prediction from price data, technical indicators, recent news, news sentiment, almanac signals, and the target trading date.

| ID | Test Type | Input / Action | Expected Result |
| --- | --- | --- | --- |
| AI-01 | Black-box happy path | Run LLM analysis with valid JSON containing `target_price`, `confidence`, and `reason`. | The app extracts the target price, confidence, and reason successfully. |
| AI-02 | Failure case | LLM returns invalid JSON text. | The app falls back to the current price, confidence `0.5`, and a safe message. |
| AI-03 | Boundary case | LLM returns confidence below 0 or above 1. | Parsed confidence is clamped into the 0 to 1 range. |

### User Story 2: Final Ensemble Prediction (#16)

Combine XGBoost and LLM predictions into one final predicted price, include LLM confidence, show BUY or SELL signal, and explain the model weights used in the ensemble result.

| ID | Test Type | Input / Action | Expected Result |
| --- | --- | --- | --- |
| FP-01 | Black-box happy path | XGBoost predicts 110 and LLM predicts 120 with 50/50 weights. | Ensemble prediction is 115. |
| FP-02 | Boundary case | Ensemble price is greater than current price. | Signal is `BUY`. |
| FP-03 | Boundary case | Ensemble price is less than or equal to current price. | Signal is `SELL`. |

### User Story 3: News and Sentiment Analysis (#19)

Fetch recent company news, display headlines and summaries in a News tab, and use FinBERT to classify news sentiment as positive, neutral, or negative with fallback handling when FinBERT is unavailable.

| ID | Test Type | Input / Action | Expected Result |
| --- | --- | --- | --- |
| NS-01 | Black-box happy path | Positive average headline compound score. | Summary label is `Positive`. |
| NS-02 | Boundary case | Empty news sentiment list. | Neutral fallback score of `50.0` is used. |
| NS-03 | White-box branch | Multiple headlines with different compound strengths. | Prompt summary includes the latest headline and strongest sentiment headline. |

### User Story 4: Price and Market Visualisation (#20)

Display stock price movement using an interactive candlestick chart with MA20, volume, zooming, panning, and range slider support. Also include a Big Tech heatmap showing recent percentage changes.

| ID | Test Type | Input / Action | Expected Result |
| --- | --- | --- | --- |
| PV-01 | Black-box happy path | Build chart with valid OHLCV data. | Figure contains candlestick, MA20 line, and volume traces. |
| PV-02 | UI output | Build chart with theme settings. | Plot uses the selected theme template and text colour. |
| PV-03 | Interaction check | Build chart. | Range slider and pan interaction settings are present. |

### User Story 5: Prediction Record Storage and Backtesting Support (#14)

Store prediction records in Supabase or local CSV, avoid duplicate records, update pending predictions with actual close prices, and calculate XGBoost, LLM, and ensemble absolute errors for evaluation.

| ID | Test Type | Input / Action | Expected Result |
| --- | --- | --- | --- |
| PR-01 | Grey-box data check | Record payload contains normal numeric values. | JSON payload can be encoded for Supabase. |
| PR-02 | Failure case | Record payload contains `NaN` or infinity. | Invalid JSON values are converted to `null`. |
| PR-03 | Duplicate/state check | Existing ticker/date/reference date is found. | App skips inserting a duplicate record. |

### User Story 6: Trading Date and Market Timing Logic (#13)

Calculate the correct prediction target date by considering US trading days, weekends, holidays, market close timing, and next market opening time.

| ID | Test Type | Input / Action | Expected Result |
| --- | --- | --- | --- |
| TD-01 | Boundary case | Ask for the next trading day from a Friday. | The next trading day skips the weekend. |
| TD-02 | Boundary case | Check a known US market holiday. | Holiday is not treated as a trading day. |
| TD-03 | White-box branch | Format a future market-open time delta. | Time delta is shown as hours and minutes. |

## Automation Plan

The automated pytest suite focuses on logic that can be tested without live API calls:

- Theme dictionaries and Plotly figure configuration.
- JSON cleaning for Supabase-safe payloads.
- LLM response parsing and fallback behavior.
- Sentiment scoring and news prompt summaries.
- Trading-day, holiday, and time-format helpers.
- Ensemble prediction calculation and signal generation.
- Price-data cleanup for rows with missing `Close` values.

Live tests for Yahoo Finance, Finnhub, OpenAI, and Supabase should be handled separately as integration tests with mocked/staged API responses or a test database. For continuous integration, the recommended command is:

```bash
python -m pytest
```
