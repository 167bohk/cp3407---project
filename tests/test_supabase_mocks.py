import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import stock_dashboard as dashboard


MOCK_CONFIG = {
    "url": "https://mock-project.supabase.co",
    "key": "mock-api-key",
}


def mock_http_response(payload):
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def forecast_result():
    return {
        "predicted_date": "2026-07-22",
        "reference_close_date": "2026-07-21",
        "pred_price": 215.0,
        "llm_price": 218.0,
        "llm_conf": 0.7,
        "ensemble_price": 217.1,
        "weight_xgb": 0.3,
        "weight_llm": 0.7,
    }


def test_supabase_request_reads_mock_prediction_records():
    records = [{"ticker": "AAPL", "ensemble_price": 217.1}]
    response = mock_http_response(records)

    with patch.object(
        dashboard, "get_supabase_config", return_value=MOCK_CONFIG
    ), patch.object(dashboard, "urlopen", return_value=response) as mock_urlopen:
        result = dashboard.supabase_request(
            "GET", "/rest/v1/prediction_log?select=*"
        )

    assert result == records
    mock_urlopen.assert_called_once()
    request = mock_urlopen.call_args.args[0]
    assert request.get_method() == "GET"
    assert request.full_url == (
        "https://mock-project.supabase.co/rest/v1/prediction_log?select=*"
    )
    assert request.get_header("Apikey") == "mock-api-key"
    assert request.get_header("Authorization") == "Bearer mock-api-key"


def test_supabase_request_sends_clean_json_to_mock_service():
    response = mock_http_response([{"id": 1}])
    payload = [{"ticker": "AAPL", "actual_close": float("nan")}]

    with patch.object(
        dashboard, "get_supabase_config", return_value=MOCK_CONFIG
    ), patch.object(dashboard, "urlopen", return_value=response) as mock_urlopen:
        dashboard.supabase_request("POST", "/rest/v1/prediction_log", payload)

    request = mock_urlopen.call_args.args[0]
    sent_payload = json.loads(request.data.decode("utf-8"))
    assert request.get_method() == "POST"
    assert sent_payload == [{"ticker": "AAPL", "actual_close": None}]
    assert mock_urlopen.call_args.kwargs["timeout"] == 15


def test_supabase_request_skips_network_when_not_configured():
    with patch.object(
        dashboard, "get_supabase_config", return_value=None
    ), patch.object(dashboard, "urlopen") as mock_urlopen:
        result = dashboard.supabase_request("GET", "/rest/v1/prediction_log")

    assert result is None
    mock_urlopen.assert_not_called()


def test_prediction_log_handles_mock_supabase_network_failure():
    with patch.object(
        dashboard, "prediction_record_exists", return_value=False
    ), patch.object(
        dashboard, "get_supabase_config", return_value=MOCK_CONFIG
    ), patch.object(
        dashboard,
        "supabase_request",
        side_effect=URLError("mock connection failure"),
    ) as mock_request:
        storage, error = dashboard.append_prediction_log_record(
            "AAPL", 210.0, forecast_result()
        )

    assert storage == "csv"
    assert "mock connection failure" in error
    mock_request.assert_called_once()
    assert mock_request.call_args.args[0:2] == (
        "POST",
        "/rest/v1/prediction_log",
    )
    assert mock_request.call_args.args[2][0]["ticker"] == "AAPL"
