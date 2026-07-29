import sys
import os
from unittest.mock import MagicMock, patch

# Dynamically add the parent directory (mcp/) to Python's import search path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from fraud_mcpserver import analyze, recent_cases


@patch("fraud_mcpserver.httpx.post")
def test_analyze_fraud_detection(mock_post):
    # Fake the response from Ollama
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "Conclusion: High Risk - Transaction blocked."
    }
    mock_post.return_value = mock_response

    # Run the actual tool function
    result = analyze("Urgent overseas wire request for $28,500")

    # Assert the logic works as expected
    assert result["verdict"] == "fraud"
    assert "High Risk" in result["raw_model_output"]


def test_recent_cases_returns_list():
    # Test reading log file without network calls
    cases = recent_cases(limit=1)
    assert isinstance(cases, list)
