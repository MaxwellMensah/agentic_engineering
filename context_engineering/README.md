# Context-Engineered ReAct Agent

A production-oriented ReAct agent built with **LangChain, LangGraph, and Google Gemini**, with a two-layer context-engineering architecture.

The system separates context engineering into two distinct responsibilities:

1. **Agent-level context engineering** - the mandatory control layer around every model call. It manages runtime context, dynamic prompting, input guardrails, model routing, tool authorization, execution limits, production telemetry, tool errors, tool-output protection, and conversation summarization.
2. **Retrieval-level context engineering** - a custom `ContextBuilder` that is activated **only when the `search` tool is invoked**. It optimizes retrieved web content through relevance ranking, token budgeting, compression, Lost-in-the-Middle reordering, quality gating, and auditing.

This separation keeps general agent orchestration independent from retrieval-specific context optimization.

---

## Architecture Overview

### Core execution rule

*Every model call passes through the agent-level middleware. `ContextBuilder` is only activated when the ReAct agent invokes the `search` tool.*

The agent therefore behaves as a cyclic ReAct system rather than a single linear pipeline.

```text
                                  ┌──────────────────────┐
                                  │      User Query      │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                               ┌──────────────────────────┐
                               │       ReAct Agent        │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                ╔══════════════════════════════════════════════════╗
                ║            AGENT-LEVEL MIDDLEWARE                ║
                ║                MANDATORY PATH                    ║
                ║                                                  ║
                ║  • Adaptive System Prompt                        ║
                ║  • Input Security Guardrail                      ║
                ║  • Model Routing (Dynamic Model Selection)       ║
                ║  • Role-Based Tool Policy                        ║
                ║  • Step Limiter                                  ║
                ║  • Production Telemetry                          ║
                ║  • Tool Error Handling                           ║
                ║  • Tool Output Compression                       ║
                ║  • Conversation Summarization                    ║
                ╚═══════════════════════╤══════════════════════════╝
                                        │
                                        ▼
                               ┌──────────────────────┐
                               │      Agent LLM       │
                               │   gemini-2.5-flash   │
                               └──────────┬───────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │                           │
                       Tool Call                   Final Answer
                            │                           │
                            ▼                           ▼
                  ┌───────────────────┐       ┌─────────────────┐
                  │  Tool Execution   │       │   Return User   │
                  └─────────┬─────────┘       └─────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
         ┌──────────┐  ┌──────────┐  ┌──────────────┐
         │Calculator│  │ Weather  │  │    Search    │
         │   AST    │  │ Checker  │  │    Tavily    │
         └────┬─────┘  └────┬─────┘  └──────┬───────┘
              │             │               │
              │             │               ▼
              │             │      ╔══════════════════════╗
              │             │      ║    CONTEXTBUILDER    ║
              │             │      ║     SEARCH ONLY      ║
              │             │      ╠══════════════════════╣
              │             │      ║ • Relevance Ranking  ║
              │             │      ║ • Token Budgeting    ║
              │             │      ║ • Compression        ║
              │             │      ║ • Lost-in-Middle     ║
              │             │      ║ • Quality Gate       ║
              │             │      ║ • Audit Logging      ║
              │             │      ╚═══════════╤══════════╝
              │             │                  │
              └─────────────┴──────────────────┘
                            │
                            ▼
                    Tool Observation
                            │
                            ▼
                ╔════════════════════════╗
                ║ Agent Middleware Again ║
                ╚════════════╤═══════════╝
                             │
                             ▼
                            LLM
                             │
                          ...loop...

```

### Execution examples

#### Calculator request

```text
User
  ↓
Agent Middleware
  ↓
LLM
  ↓
Calculator
  ↓
Tool Result
  ↓
Agent Middleware
  ↓
LLM
  ↓
Final Answer

```

#### Weather request

```text
User
  ↓
Agent Middleware
  ↓
LLM
  ↓
Weather Checker
  ↓
Tool Result
  ↓
Agent Middleware
  ↓
LLM
  ↓
Final Answer

```

#### Search request

```text
User
  ↓
Agent Middleware
  ↓
LLM
  ↓
Search Tool
  ↓
Tavily
  ↓
ContextBuilder
  ├── relevance ranking
  ├── token budgeting
  ├── compression
  ├── Lost-in-the-Middle reordering
  ├── quality gate
  └── audit logging
  ↓
Optimized Search Observation
  ↓
Agent Middleware
  ↓
LLM
  ↓
Final Answer

```

The important architectural distinction is:

> **Middleware wraps model calls globally; `ContextBuilder` is a retrieval-specific component owned by the `search` tool.**

---

# Two-Layer Context Engineering

## 1. Agent-Level Context Engineering

The agent-level middleware is the **global control layer**.

It applies to every model call regardless of which tools are used, strictly following this operational sequence:

| Step | Component | Responsibility |
| --- | --- | --- |
| 1 | `adaptive_system_prompt` | Dynamically constructs model instructions from runtime context |
| 2 | `input_guardrail` | Evaluates inputs for security policies, prompt injection, and compliance |
| 3 | `dynamic_model_selection` | Provides context-aware model routing (e.g., routing to `gemini-2.5-pro` vs `gemini-2.5-flash`) |
| 4 | `role_based_tools` | Controls which tools are available to each role |
| 5 | `step_limiter` | Restricts agent execution depth (strips tools when step limit is hit) |
| 6 | `production_telemetry` | Logs latency, token usage, cost estimates, and operational metrics |
| 7 | `ToolOutputCompressionMiddleware` | Prevents unusually large tool outputs from bloating context using `gemini-2.5-flash` |
| 8 | `ToolErrorMiddleware` | Converts tool failures into structured model feedback |
| 9 | `SummarizationMiddleware` | Compresses long-running conversation history |

## 2. Retrieval-Level Context Engineering

The retrieval layer is **conditional**.

It only executes when the agent invokes:

```text
search()

```

The `ContextBuilder` receives the raw Tavily results and transforms them into a bounded, optimized observation before returning that observation to the ReAct loop.

---

# ReAct Execution Model

The agent uses a cyclic model/tool architecture:

```text
                ┌─────────────────────┐
                │ Agent Middleware    │
                └──────────┬──────────┘
                           ↓
                          LLM
                           │
                    ┌──────┴──────┐
                    │             │
                 Tool Call     Final Answer
                    │             │
                    ↓             ↓
                 Execute        Return
                    │
                    ↓
             Tool Observation
                    │
                    ↓
             Agent Middleware
                    │
                    ↓
                   LLM
                    │
                   ...

```

This means a single user request can involve multiple model calls.

For a search request, the search observation is additionally processed by the `ContextBuilder` before the next model call.

---

# ContextBuilder

`ContextBuilder` is the custom retrieval-context optimization engine.

Its purpose is:

*Reduce retrieval noise and context pressure while retaining information relevant to the user's query.*

It does not replace the agent or produce the final answer. It prepares the **evidence** that the agent's LLM will evaluate.

## Pipeline

```text
Raw Retrieval Chunks
        │
        ▼
Cleaning & Term Normalization
        │
        ▼
Lexical Relevance Ranking
        │
        ▼
Token-Budget Selection
        │
        ├── Fits Budget ──────────► Keep
        │
        └── Exceeds Budget
                    │
                    ▼
          Opportunistic Compression (via Gemini)
                    │
                ┌───┴────┐
                │        │
              Fits     Doesn't Fit
                │        │
                ▼        ▼
              Keep     Discard
                │
                ▼
Lost-in-the-Middle Reordering
        │
        ▼
Quality Gate
        │
    ┌───┴────┐
    │        │
  PASS     FAIL
    │        │
    ▼        ▼
 Final Context  Empty Context
        │
        ▼
 Tool Observation

```

## ContextBuilder techniques

### Relevance ranking

Each retrieved chunk is assigned a deterministic lexical relevance score based on meaningful query-term overlap.

```text
relevance =
    matching query terms
    --------------------
    query terms

```

This is intentionally lightweight and deterministic. The implementation can later be replaced by BM25 or embedding similarity without changing the surrounding architecture.

### Token-budget allocation

The builder enforces an explicit retrieval-context budget.

It accounts for both:

```text
chunk tokens
+
separator tokens

```

so the budget reflects the actual assembled context rather than just the raw documents.

### Opportunistic compression

Compression is only attempted when a chunk cannot fit naturally into the remaining budget.

This avoids paying the latency cost of an LLM compression call for chunks that already fit. When needed, `gemini-2.5-flash` compresses the candidate chunk.

### Lost-in-the-Middle reordering

Reordering occurs **after budget selection** so that moving the second-highest-ranked chunk to the end cannot cause it to be discarded by the token allocator.

For a ranked set:

```text
A B C D E

```

the selected set is reordered as:

```text
A C D E B

```

where:

```text
A = highest relevance
B = second-highest relevance

```

This is a heuristic designed to place highly relevant information at the beginning and end of the assembled context.

### Quality gating

The final context receives a quality score:

```text
Quality =
    0.70 × Query-Term Coverage
  + 0.30 × Context-Length Sufficiency

```

If:

```text
quality < quality_threshold

```

the builder rejects the context rather than silently sending a low-quality retrieval payload to the model.

### Auditing

Each retrieval-context build records:

* query
* input chunk count
* selected chunk count
* discarded chunk count
* raw estimated tokens
* final estimated tokens
* configured token budget
* compression count
* average relevance
* maximum relevance
* query-term coverage
* context-length score
* final quality score
* gate result

Audit records are available through:

```python
context_builder.audit_json()

```

---

# Why ContextBuilder Is Search-Only

The ContextBuilder is intentionally not attached to every tool.

## Calculator

Calculator output is deterministic and generally small:

```text
125 * 48
   ↓
AST Calculator
   ↓
6000

```

There is no collection of retrieved documents to rank or compress.

## Weather

The weather tool produces a small, structured observation:

```text
Weather for Accra:
28°C, partly cloudy,
humidity 65%, wind 12 km/h NE.

```

Again, there is no retrieval corpus requiring the ContextBuilder pipeline.

## Search

Search can return multiple heterogeneous documents:

```text
Tavily
  ↓
multiple documents
  ↓
large / redundant / differently relevant chunks
  ↓
ContextBuilder

```

This is where retrieval-specific context engineering provides value.

---

# Component Responsibilities

| Component | Responsibility | Implementation |
| --- | --- | --- |
| `ContextBuilder` | Retrieval-context optimization | Lexical relevance ranking, separator-aware token budgeting, per-chunk Gemini compression, Lost-in-the-Middle reordering, quality gating, audit logging |
| Agent Middleware | Global model-call context control | Dynamic prompting, input guardrails, model routing, tool authorization, step limits, telemetry, summarization, error handling |
| `Search` | External information retrieval | Tavily + `ContextBuilder` |
| `Calculator` | Mathematical computation | AST parser with explicit operator/function allowlists |
| `Weather Checker` | Weather information | Small structured weather observation |
| `ToolOutputCompressionMiddleware` | Generic tool-output protection | Compresses unusually large tool results using `gemini-2.5-flash` |
| `ToolErrorMiddleware` | Tool failure recovery | Converts exceptions into structured feedback |
| `MemorySaver` | Persistent agent state | Stores conversation state by `thread_id` |

---

# Agent Middleware Stack

## `adaptive_system_prompt`

Injects runtime context such as:

```text
user_role
expertise_level
environment

```

into the model's instructions dynamically. It executes around each model invocation.

## `input_guardrail`

Filters and sanitizes inbound prompts before they reach the model. It detects adversarial inputs, prompt injection attempts, or out-of-bounds user instructions, halting or flagging unsafe requests early.

## `dynamic_model_selection`

Provides a context-aware model routing layer. It evaluates context complexity, prompt size, or query difficulty and dynamically routes the request to the appropriate model (e.g., routing to `gemini-2.5-pro` for multi-step reasoning or deep history, and defaulting to `gemini-2.5-flash` for high-speed tasks).

## `role_based_tools`

Applies tool authorization according to runtime role.

Example:

```text
viewer
├── search ✅
├── weather_checker ✅
└── calculator ❌

```

```text
calculator_only
├── calculator ✅
├── search ❌
└── weather_checker ❌

```

## `step_limiter`

Enforces execution depth controls:

```text
MAX_MODEL_STEPS = 6

```

When the limit is reached, tool definitions are stripped from the model request, forcing the agent to produce a final answer with available observations.

## `production_telemetry`

Collects execution metrics across every model pass, including token counts, latency, routing decisions, and system logs, ensuring full operational visibility in production.

## `ToolOutputCompressionMiddleware`

Provides a secondary protection layer for abnormally large tool responses from non-search or arbitrary tools using `gemini-2.5-flash`.

## `ToolErrorMiddleware`

Catches tool execution failures and returns structured feedback to the agent instead of terminating the entire run.

## `SummarizationMiddleware`

Compresses older conversation history when the configured token threshold is reached, preventing long-running conversations from continuously accumulating context.

---

# Safety-Oriented Calculator

The calculator does not rely on unrestricted `eval()`.

Instead it parses expressions using Python's AST and explicitly permits:

* arithmetic operators
* unary operators
* approved mathematical functions
* numeric constants

Unsupported expressions are rejected.

---

# Getting Started

## Prerequisites

* Python environment managed with **Pixi** or standard `venv`
* Google Gemini API key (`GOOGLE_API_KEY`)
* Tavily API key (`TAVILY_API_KEY`)

## Install

```bash
pixi init
pixi add langchain langchain-google-genai langgraph tavily-python python-dotenv

```

Activate the environment:

```bash
pixi shell

```

Configure Environment Variables:

```bash
export GOOGLE_API_KEY="your-google-gemini-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

```

---

# Usage

## Run the agent

```bash
python main_agent.py

```

or:

```bash
pixi run python main_agent.py

```

## Programmatic usage

```python
from main_agent import run

result = run(
    "Who is the current president of Ghana in 2026?",
    thread_id="session_search",
    user_role="user",
    expertise_level="advanced",
    environment="production",
)

```

A calculator request:

```python
result = run(
    "What is 125 * 48?",
    thread_id="session_math",
)

```

A weather request:

```python
result = run(
    "What is the weather in Accra?",
    thread_id="session_weather",
)

```

---

# Inspecting ContextBuilder Audits

Import the shared `ContextBuilder` instance:

```python
from main_agent import context_builder

print(
    context_builder.audit_json()
)

```

Important:

*Audit entries are created only when the `search` tool invokes `ContextBuilder`.*

A calculator-only or weather-only request does not generate a ContextBuilder audit record.

Example:

```json
[
  {
    "timestamp": "2026-08-25T17:21:24+00:00",
    "query": "current president of Ghana in 2026?",
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

---

# Role-Based Access Control Testing

Test role policies by changing the runtime context.

For example:

```python
run(
    "Calculate 125 * 48",
    thread_id="viewer_test",
    user_role="viewer",
)

```

The `role_based_tools` middleware removes the calculator from the model's available tools for the viewer role.

---

# Configuration Reference

| Parameter | Default | Component | Description |
| --- | --- | --- | --- |
| `token_budget` | `1200` | `ContextBuilder` | Maximum estimated tokens for assembled retrieval context |
| `quality_threshold` | `0.60` | `ContextBuilder` | Minimum quality score required to pass retrieval gating |
| `separator` | `"\n\n---\n\n"` | `ContextBuilder` | Delimiter between retrieved chunks |
| `min_quality_context_tokens` | `80` | `ContextBuilder` | Token count at which the length component reaches full score |
| `minimum_remaining_tokens_for_compression` | `40` | `ContextBuilder` | Minimum remaining budget required before attempting compression |
| `MAX_MODEL_STEPS` | `6` | Agent | Maximum model-call steps allowed per request |
| `SUMMARY_TRIGGER_TOKENS` | `4000` | Agent | Conversation size that triggers summarization |
| `MAX_TOOL_OUTPUT_CHARS` | `5000` | Agent | Tool-output size threshold for generic compression |

---

# Token Accounting

The default `TokenCounter` uses an approximation:

```text
estimated tokens ≈ characters / 4

```

This is intended for lightweight local execution and demonstrations.

The `ContextBuilder` accepts a custom tokenizer callable:

```python
ContextBuilder(
    token_counter=my_tokenizer
)

```

This allows a model-specific tokenizer such as Google's native Gemini tokenizer or another provider-specific tokenizer to be used when exact context accounting is required.

The architecture therefore separates **token-budget policy** from **token-counting implementation**.

---

# Design Principles

### Global middleware, conditional retrieval optimization

The system deliberately avoids treating all context as the same.

```text
Agent-level context
        ↓
Always relevant to model execution

Retrieval context
        ↓
Only relevant when search/retrieval occurs

```

### Evidence before reasoning

Search results are optimized before being returned as an observation to the agent.

```text
Raw retrieval
    ↓
ContextBuilder
    ↓
Optimized evidence
    ↓
LLM reasoning

```

### Explicit budgets

Instead of allowing retrieved documents to consume the context window without control, the ContextBuilder maintains an explicit retrieval-context budget.

### Graceful degradation

When compression fails, a chunk does not fit, or the quality gate fails, the system avoids silently pretending that the retrieval context is sufficient.

### Observable context engineering

Every retrieval optimization produces auditable metrics so context decisions can be inspected rather than treated as hidden model behavior.

---

# Project Summary

This project demonstrates a two-layer approach to context engineering:

```text
                    CONTEXT ENGINEERING
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
     AGENT-LEVEL LAYER               RETRIEVAL LAYER
       ALWAYS ACTIVE                   SEARCH ONLY
            │                               │
     dynamic prompting               relevance ranking
     input guardrails                token budgeting
     model routing                   compression
     tool authorization              Lost-in-Middle
     step limits                     quality gating
     telemetry & metrics             audit logging
     summarization                   
            │                               │
            └───────────────┬───────────────┘
                            ▼
                           LLM

```

The resulting architecture combines:

* **LangChain middleware** for global agent-level context control (`adaptive_system_prompt`, `input_guardrail`, `dynamic_model_selection`, `role_based_tools`, `step_limiter`, `production_telemetry`)
* **LangGraph-backed agent execution** for the ReAct loop and state management
* **Custom ContextBuilder** for retrieval-specific context optimization
* **Google Gemini** for model inference, dynamic routing, and per-chunk compression
* **Tavily** for web retrieval
* **AST-based calculation** for safe deterministic math
* **MemorySaver** for persistent conversation state
* **Auditing & Telemetry** for end-to-end observability

The central architectural principle is simple:

*Every model call passes through agent-level middleware. Only search calls additionally pass through ContextBuilder before their results re-enter the agent loop.*
