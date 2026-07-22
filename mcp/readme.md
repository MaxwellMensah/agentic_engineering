feat(mcp): add fraud detection MCP server and Claude Desktop integration

Implement a FastMCP server (fraud_mcpserver.py) providing local fraud 
detection capabilities to Claude Desktop over STDIO transport.

Key changes:
- Add `analyze` tool to evaluate transaction text via local Ollama daemon (fraud-model-v4:latest).
- Add `recent_cases` tool to retrieve and display historical audit logs from `recent_cases.jsonl`.
- Implement dynamic base path resolution to ensure reliable file logging regardless of host execution directory.
- Configure `claude_desktop_config.json` to launch the server using the `.pixi` virtual environment Python interpreter.
- Validate end-to-end pipeline across 4 synthetic test scenarios (offshore wire, clean ACH, BEC executive pressure, vendor account redirection).