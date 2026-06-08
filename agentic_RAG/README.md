# Agentic RAG with Advanced LangGraph Routing & Confidence Thresholds

An enterprise-grade, stateful **Corrective RAG (CRAG)** pipeline built using **LangGraph**, **AWS Bedrock (Anthropic Claude & Amazon Titan)**, and a local **ChromaDB** vector store.

This system acts as an automated **Financial Fraud Investigator**. Instead of relying on a naive, single-shot database lookup, this agent evaluates the mathematical similarity of retrieved past cases, routes data flows conditionally, and dynamically optimizes its search queries if the initial historical matches are ambiguous. It is wrapped as a high-performance **FastAPI** microservice and fully instrumented with **LangSmith** for production-grade tracing and observability.

## 🏗️ Architecture & System Data Flow

The system uses a state machine to track retry counters and control path execution. The diagram below reflects the structure generated dynamically by compiling `agentic_rag.py`:

```
           [ START ]
               │
          init_node  (Sets retrieval_attempts = 0)
               │
               ▼
   generate_query_or_respond
               │
     [ route_on_tool_calls ]
               ├──► (No Tool Calls) ──► [ END ]
               └──► (Tool Calls) ────► retrieve_node (ToolNode)
                                           │
                                           ▼
                                  [ grade_documents ]
                                           │
             ┌──────────────────────────────┼──────────────────────────────┐
             ▼                              ▼                              ▼
      (Score ≥ 0.60)                  (Score < 0.60)                 (Score < 0.60)
             │                   [retrieval_attempts < 1]       [retrieval_attempts >= 1]
             ▼                              ▼                              ▼
      generate_answer                rewrite_question        fallback_empty_response
             │                              │                              │
             ▼                              ▼                              ▼
          [ END ]              (Loops back to tool agent)               [ END ]

```

## 🔥 Key Architectural Features

### 1. Vector Search Enforced in Cosine Space

To ensure that database distance metrics scale predictably regardless of text length anomalies, the local collection uses a strict **HNSW Cosine Coordinate Space**:


$$\text{Similarity Score} = 1.0 - \text{Cosine Distance}$$

### 2. State-Driven Memory Loops

Rather than relying on stateless recursion, the workflow extends the base `MessagesState` class to maintain an explicit loop-tracker (`retrieval_attempts`). This protects your system against infinite LLM loop execution and caps retries deterministically at **2 passes**.

### 3. Algorithmic Guardrails over LLM Guesswork

Instead of wasting API tokens asking an LLM if the fetched documentation is relevant, relevance grading is performed programmatically within Python inside the custom tool definition. If the similarity score drops below **0.60**, a conditional gate intercepts execution instantly.

### 4. Asynchronous Production Ingress

The state machine is wrapped using **FastAPI** and **Uvicorn**, providing a production-ready asynchronous HTTP POST endpoint (`/ask`). It implements structured input/output validation powered by **Pydantic** to guard the pipeline against malformed JSON schemas.

## 🛠️ Tech Stack & Dependencies

* **Orchestration / Framework:** `langgraph`, `langchain-core`
* **Large Language Model:** Anthropic Claude (via `ChatBedrockConverse` API on AWS)
* **Embedding Engine:** Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`)
* **Vector Database:** `langchain-chroma` (ChromaDB engine)
* **API Wrapper & Server:** `fastapi`, `uvicorn`, `pydantic`
* **Observability & Infrastructure:** `langsmith`, `python-dotenv`, `boto3`, `jq`

## ⚙️ Configuration & Environment Setup

### 1. Installation

Install all required libraries inside your virtual environment (`venv`):

```bash
pip install -q -U langgraph langchain_community langchain_aws \
langchain_chroma langchain_core python-dotenv boto3 jq \
fastapi uvicorn pydantic

```

### 2. Secret Management & Credentials

The application natively sources keys from the active runtime environment. Create a `.env` file in your root workspace directory:

```ini
# AWS Infrastructure Tokens
AWS_ACCESS_KEY_ID=your_iam_user_access_key
AWS_SECRET_ACCESS_KEY=your_secure_backend_signature_string
AWS_DEFAULT_REGION=us-east-1

# LangSmith Deep Telemetry & Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Week_3_Fraud_Agentic_RAG

```

### 🚀 Quick Start Guide

**Step 1:** Ingest Fraud History into Vector Database
Configure your core storage settings in `agentic_rag.py`:

```python
PERSIST_DIR = "./chromadb_frauddata_agentic"
COLLECTION_NAME = "langgraph_frauddata_agentic"
COLLECTION_METADATA = {"hnsw:space": "cosine"}
INGESTION_LIMIT = 500

```

**Step 2:** Start the FastAPI Endpoint Microservice
Launch the server via your application module file (`fastapi_rag.py`). Uvicorn will spin up a hot-reloading development server listening on port `8000`:

```bash
python3 fastapi_rag.py

```

**Step 3:** Trigger a Live Audit Execution Route
Open a secondary terminal window and submit an incident analysis request via an HTTP POST payload:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Analyze a transaction involving an NGN transfer from Sydney at late night using a new mobile device on a datacenter VPN."}'

```

The API will respond with a clean, validated JSON structure detailing both the node journey and the text analysis:

```json
{
  "node_execution_sequence": [
    "init",
    "generate_query_or_respond",
    "retrieve",
    "rewrite_question",
    "generate_query_or_respond",
    "retrieve",
    "generate_answer"
  ],
  "final_response": "# Fraud Risk Assessment\n\n**Pattern Match:** This transaction exhibits hallmarks consistent with FC-053..."
}

```

## 📊 Live Sample Stream Output Tracing

When evaluated against high-risk inputs, the system demonstrates parallel investigation tactics and accurate routing:

* **Parallel Queries Fired:** Claude processes the risk flags and generates three distinct search queries in parallel to gather maximum historical data (`NGN Transfer...`, `international money transfer...`, and `SIM swapping...`).
* **Programmatic Interception:** The tool logs real-time scores directly to telemetry console displays:
* Query 1 Similarity: `0.5367` ❌ (Fails isolated threshold)
* Query 2 Similarity: `0.6094` ✅ (Passes threshold)


* **Automated Bypassing:** Because Query 2 safely matches past patterns above the 0.60 standard boundary, the conditional edge `grade_documents` prints `✅ Threshold Passed. Routing to answer generation.` and avoids mutating the baseline question.
* **Structured Delivery:** The engine delivers a comprehensive audit answer matching structural indicators (anonymous IP proxies, fresh device fingerprints, off-hours timing mismatch) to confirm vulnerability profiles dynamically.

---

## 👁️ Enterprise Observability & LangSmith Tracing

Because adaptive, self-correcting agent loop architectures exhibit non-deterministic behavior, standard API logging strategies are insufficient. This project completely instruments all LangGraph node transitions, conditional evaluations, and external AWS Bedrock invocations using native **LangSmith Integration**.

### Operational Trace Overview

Below is a visual breakdown of a complete transaction evaluation cycle captured live via the LangSmith monitoring console:

### Key Metrics Monitored in the Trace Engine

1. **State-Machine Path Auditing**
The nested tree view documents the complete runtime lifecycle. If a query underperforms, the trace visualizes the exact pivot point where the system transitions to `rewrite_question` before routing back into the retrieval matrix. This exposes how the agent updates the internal state message history arrays chronologically.

2. **Microsecond Latency Profiling**
The timeline clearly segregates processing latency profiles. It isolates high-speed local mathematical operations (like querying our local ChromaDB index using Titan Text Embeddings V2 vectors) from variable external downstream network hops (like streaming input contextual blocks to the `ChatBedrockConverse` client for Claude processing).

3. **Strict Financial Token Counting**
Every structural node layer explicitly audits input, output, and aggregate token properties. This prevents budget inflation by allowing engineering teams to calculate exact billing metrics per fraud investigation run, check for runaway iterative loops, and optimize prompt block patterns under load.

4. **Concurrent Tool Performance**
When the agent uses tool execution arrays to query multiple historical vectors simultaneously, LangSmith splits the calls into synchronized parallel traces. One can inspect each standalone search string side-by-side, view its specific cosine similarity score output, and evaluate the programmatic grading criteria without interrupting the core asynchronous user thread.