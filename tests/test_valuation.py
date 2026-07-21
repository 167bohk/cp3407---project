import stock_dashboard as dashboard


def metrics(multiplier, price=100):
    return {
        "current_price": price,
        "trailing_pe": 20 * multiplier,
        "forward_pe": 18 * multiplier,
        "price_to_book": 5 * multiplier,
        "price_to_sales": 4 * multiplier,
        "enterprise_to_ebitda": 12 * multiplier,
    }


def test_positive_finite_number_rejects_invalid_values():
    assert dashboard.positive_finite_number(None) is None
    assert dashboard.positive_finite_number(-5) is None
    assert dashboard.positive_finite_number(float("nan")) is None
    assert dashboard.positive_finite_number("12.5") == 12.5


def test_relative_valuation_identifies_undervalued_company():
    result = dashboard.analyze_relative_valuation(
        metrics(0.6), [metrics(1.0), metrics(1.1), metrics(0.9)]
    )

    assert result["label"] == "Potentially Undervalued"
    assert result["valid_metric_count"] == 5
    assert result["fair_value_mid"] > 100


def test_relative_valuation_identifies_overvalued_company():
    result = dashboard.analyze_relative_valuation(
        metrics(1.5), [metrics(1.0), metrics(1.1), metrics(0.9)]
    )

    assert result["label"] == "Potentially Overvalued"
    assert result["fair_value_mid"] < 100


def test_relative_valuation_identifies_fairly_valued_company():
    result = dashboard.analyze_relative_valuation(
        metrics(1.05), [metrics(1.0), metrics(1.1), metrics(0.9)]
    )

    assert result["label"] == "Fairly Valued"


def test_relative_valuation_requires_three_valid_metrics():
    company = {"current_price": 100, "trailing_pe": 10, "forward_pe": 9}
    peers = [
        {"trailing_pe": 20, "forward_pe": 18},
        {"trailing_pe": 22, "forward_pe": 19},
    ]

    result = dashboard.analyze_relative_valuation(company, peers)

    assert result["label"] == "Insufficient Data"
    assert result["valid_metric_count"] == 2
