import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

# Dynamically locate the folder containing server.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_LOG = os.path.join(BASE_DIR, "recent_cases.jsonl")

mcp = FastMCP("fraud-detection-server")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "fraud_model_q4km_V4"


@mcp.tool()
def analyze(document_text: str) -> dict:
    """
    Run fraud classification on extracted document text
    (e.g. from AWS Textract output on a receipt or transfer confirmation).
    Returns the model's binary fraud verdict and raw response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": document_text,
        "stream": False,
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
    resp.raise_for_status()
    raw_output = resp.json().get("response", "")

    # NOTE: legacy <sentiment> tag still shows up in output — parse it out here
    # rather than exposing it upstream, since it's not semantically sentiment.
    verdict = "fraud" if "fraud" in raw_output.lower() else "legitimate"

    result = {
        "verdict": verdict,
        "raw_model_output": raw_output,
    }

    # log the case so recent_cases can retrieve it
    with open(CASES_LOG, "a") as f:
        f.write(json.dumps({"input": document_text[:200], **result}) + "\n")

    return result


@mcp.tool()
def recent_cases(limit: int = 5) -> list[dict]:
    """
    Return the most recently analyzed fraud cases.
    """
    try:
        with open(CASES_LOG, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    cases = [json.loads(line) for line in lines[-limit:]]
    return list(reversed(cases))


if __name__ == "__main__":
    mcp.run(transport="stdio")
