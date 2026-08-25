# Context-Engineered ReAct Agent

An production-grade ReAct agent architecture built with LangChain, LangGraph, and Ollama. This system integrates dynamic\ 
middleware for policy enforcement and dynamic prompt routing alongside **ContextBuilder**: a context-engineering pipeline\ 
designed to mitigate LLM context rot, respect token limits, and handle RAG retrieval gracefully.

---

## Architecture Overview

```
                                +---------------------------+
                                |    User Input / Query     |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |  Adaptive System Prompt   |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |  Middleware Stack         |
                                |  - Step Limiter           |
                                |  - Tool RBAC Policy       |
                                |  - Model Router           |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |       Agent LLM           |
                                |  (gemma4:e2b-it-q4_K_M)   |
                                +-------------+-------------+
                                     /                 \
                     (Tool Call Request)             (Final Output)
                           /                             \
                          v                               v
             +------------------------+          +-------------------+
             |    Tool Execution      |          | User Output Return|
             | - Search (Tavily)      |          +-------------------+
             | - AST Calculator       |
             | - Weather Checker      |
             +-----------+------------+
                         |
                         v
             +------------------------+
             |   ContextBuilder Pipeline  
             | - Term Normalization   |
             | - Token Budgeting      |
             | - LLM Compression      |
             | - Lost-in-the-Middle   |
             | - Quality Gating       |
             +------------------------+

```

| Component | Responsibility | Technical Implementation |
| --- | --- | --- |
| **`ContextBuilder`** | RAG payload optimization & token budgeting | Exact token budgeting, TF-overlap ranking, opportunistic compression, Lost-in-the-Middle reordering, and audit logging. |
| **Agent Middleware** | Runtime safety and dynamic policy control | Step capping (`MAX_STEPS`), role-based tool authorization (RBAC), automatic long-context model routing, and error boundaries. |
| **Tool Compression** | Context bloat prevention from large tool responses | Truncates/summarizes tool responses exceeding character thresholds ($>5000$ chars) using `qwen2.5-coder:1.5b`. |
| **AST Calculator** | Safe mathematical evaluation | Abstract Syntax Tree parsing that restricts execution to safe operations without using `eval()`. |

---

## Context Engine Pipeline (`ContextBuilder`)

`ContextBuilder` standardizes how unstructured retrieval data (such as web search results) is processed before being fed into model context windows.

```
Raw Chunks ──► Cleaning & Term Normalization
                    │
                    ▼
          Relevance Ranking (Overlap Ratio)
                    │
                    ▼
          Token Budgeting (With Separator Overhead)
                    │
                    ├── Fits Budget? ────────► Select Chunk
                    └── Over Budget? ────────► Opportunistic LLM Compression
                                                    │
                                                    ├── Passes Budget? ──► Select Compressed
                                                    └── Fails Budget? ───► Discard Chunk
                    ┌───────────────────────────────┘
                    ▼
          Lost-in-the-Middle Reordering ([Best, *Middle, Second-Best])
                    │
                    ▼
          Quality Gate Check (0.70 * Coverage + 0.30 * Length Score)
                    │
                    ├── Pass ──► Final Context String
                    └── Fail ──► Empty Context (Fallback Handled)

```

**Key Features**

* **Separator Overhead Accounting:** Calculates precise token counts by including joining delimiters (`\n\n---\n\n`)\ during selection and compression checks.
* **Lost-in-the-Middle Reordering:** Positionally places the highest relevance chunk at the start ($P_0$) and the second-highest at the end ($P_N$) to leverage LLM attention curves.
* **Quality Gating:** Evaluates query term coverage and context density ($0.70 \cdot \text{Coverage} + 0.30 \cdot \text{Length}$). Fails gracefully if context quality drops below threshold (default: `0.60`).
* **Auditing:** Records full pipeline metrics per run for inspection via `context_builder.audit_json()`.

---

## Agent Middleware Stack

The middleware pipeline wraps model calls and tool execution, enforcing application rules declaratively:

1. **`adaptive_system_prompt`**: Injects user persona (`user_role`, `expertise_level`, `environment`) into the model prompt dynamically on every turn.
2. **`step_limiter`**: Enforces hard execution caps (`MAX_STEPS = 6`). Strips tool access on the final step to force the agent into producing a response.
3. **`role_based_tools`**: Enforces strict tool authorization based on role context (`viewer` gets search/weather only; `calculator_only` gets math only).
4. **`dynamic_model_selection`**: Dynamically upgrades model context capacity to a heavier model (`heavy_llm`) when message depth exceeds 14 turns.
5. **`ToolOutputCompressionMiddleware`**: Intercepts verbose tool payloads and compresses them via a fast local model (`qwen2.5-coder:1.5b`).
6. **`ToolErrorMiddleware`**: Catches tool execution exceptions and converts them into structured feedback messages for the model to self-correct.

---

## Getting Started

### Prerequisites

* [Pixi](https://pixi.sh/) installed locally
* [Ollama](https://ollama.ai/) running locally with the required models pulled:
```bash
ollama pull gemma4:e2b-it-q4_K_M
ollama pull qwen2.5-coder:1.5b

```



### Environment Setup with Pixi

1. **Initialize Project & Add Dependencies**
```bash
pixi init
pixi add langchain langchain-ollama langgraph tavily-python

```


2. **Activate Environment Shell**
```bash
pixi shell

```


3. **Configure API Keys**
```bash
export TAVILY_API_KEY="your-tavily-api-key"

```



---

## Usage

### Basic Execution

Run queries directly inside your active `pixi shell`:

```bash
python main_agent.py

```

Or execute directly via `pixi run`:

```bash
pixi run python main_agent.py

```

In code:

```python
from main_agent import run

# Run query with default user persona
result = run(
    "Who is the current president of Ghana in 2026?",
    thread_id="session_001",
    user_role="user",
    expertise_level="advanced",
    environment="production"
)

```

### Inspecting Context Engineering Audits

`ContextBuilder` logs diagnostic statistics for every retrieval step:

```python
from main_agent import context_builder

# Print detailed JSON audit logs
print(context_builder.audit_json())

```

**Sample Audit Output:**

```json
[
  {
    "timestamp": "2026-08-25T17:21:24.000000+00:00",
    "query": "Who is the current president of Ghana in 2026?",
    "chunks_in": 5,
    "chunks_used": 3,
    "chunks_discarded": 2,
    "raw_tokens": 1850,
    "final_tokens": 980,
    "token_budget": 1200,
    "compressed_chunks": 1,
    "average_relevance": 0.75,
    "max_relevance": 1.0,
    "query_term_coverage": 0.85,
    "length_score": 1.0,
    "quality_score": 0.895,
    "quality_threshold": 0.6,
    "passed_gate": true
  }
]

```

### Role-Based Access Control (RBAC) Testing

Restrict available tools per user session by adjusting runtime context parameters:

```python
# Viewer role: Calculator disabled automatically by role_based_tools middleware
run(
    "Calculate 125 * 48",
    thread_id="session_viewer",
    user_role="viewer"
)

```

---

## Configuration Reference

| Parameter | Default | Location | Description |
| --- | --- | --- | --- |
| `token_budget` | `1200` | `ContextBuilder` | Maximum total tokens allowed in assembled RAG context. |
| `quality_threshold` | `0.60` | `ContextBuilder` | Minimum acceptable score for the quality gate before discarding context. |
| `separator` | `"\n\n---\n\n"` | `ContextBuilder` | Delimiter string used between context chunks. |
| `MAX_STEPS` | `6` | `main_agent.py` | Upper bound for tool-call reasoning iterations per request. |
| `SUMMARY_TRIGGER_TOKENS` | `4000` | `main_agent.py` | Conversation history token limit triggering auto-summarization. |
| `MAX_TOOL_OUTPUT_CHARS` | `5000` | `main_agent.py` | Character limit triggering automatic LLM compression of tool outputs. |