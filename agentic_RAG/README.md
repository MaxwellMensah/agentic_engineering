Markdown
# Agentic RAG with Advanced LangGraph Routing & Confidence Thresholds

An enterprise-grade, stateful **Corrective RAG (CRAG)** pipeline built using **LangGraph**, **AWS Bedrock (Anthropic Claude & Amazon Titan)**, and a local **ChromaDB** vector store. 

This system acts as an automated **Financial Fraud Investigator**. Instead of relying on a naive, single-shot database lookup, this agent evaluates the mathematical similarity of retrieved past cases, routes data flows conditionally, and dynamically optimizes its search queries if the initial historical matches are ambiguous.



## 🏗️ Architecture & System Data Flow
The system uses a state machine to track retry counters and control path execution. The diagram below reflects the structure generated dynamically by compiling `fraud_agenticrag.py`:

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
             │                    [retrieval_attempts < 1]       [retrieval_attempts >= 1]
             ▼                              ▼                              ▼
      generate_answer                rewrite_question           fallback_empty_response
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


## 🛠️ Tech Stack & Dependencies

* **Orchestration / Framework:** `langgraph` (v0.0+), `langchain-core`
* **Large Language Model:** Anthropic Claude (via `ChatBedrockConverse` API on AWS)
* **Embedding Engine:** Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`)
* **Vector Database:** `langchain-chroma` (ChromaDB engine)
* **Dataset Management:** `jq` (JSONL nested structure processing)


## ⚙️ Configuration & Environment Setup

### 1. Installation
Install all required libraries silently:

```bash
pip install -q -U langgraph "langchain[openai]" langchain_community \
langchain-text-splitters bs4 requests langchain_aws langchain_openai \
langchain_chroma langchain_core python-dotenv boto3 jq
```

### 2. Secret Management & Credentials
The application natively targets secure environment retrieval. Set the following environmental keys or populate them into your Google Colab Secrets interface:

* **AWS_ACCESS_KEY_ID:** Your IAM user identifier access token.

* **AWS_SECRET_ACCESS_KEY:** Your secure backend signature string token.

### 🚀 Quick Start Guide
**Step 1:** Initialize Database and Ingest Fraud History (chroma_setup.py)
Run the data parser code block to copy the dataset from your remote drive root into active memory workspace, slice out a capped evaluation index window (500 records), convert objects into text structures, and store them into ChromaDB:

**Python**

From fraud_agenticrag.py configuration section:

PERSIST_DIR = **```"./chromadb_frauddata_agentic"```**

COLLECTION_NAME = **```"langgraph_frauddata_agentic"```**

COLLECTION_METADATA = **```{"hnsw:space": "cosine"}```**

INGESTION_LIMIT = **```500```**



**Step 2:** Running the Graph App Execution
Invoke the graph stream using an analytical scenario query payload:

Python
```
inputs = {
    "messages": [
        {
            "role": "user",
            "content": "Analyze a transaction involving an NGN transfer from Sydney at late night using a new mobile device on a datacenter VPN."
        }
    ]
}

for chunk in graph.stream(inputs):
    for node, update in chunk.items():
        print(f"--- Node Executed: {node} ---")
        update["messages"][-1].pretty_print()
```

### 📊 Live Sample Stream Output Tracing
When evaluated against high-risk inputs, the system demonstrates parallel investigation tactics and accurate routing:

* **Parallel Queries Fired:** Claude processes the risk flags and generates three distinct search queries in parallel to gather maximum historical data (NGN Transfer..., international money transfer..., and SIM swapping...).

* **Programmatic Interception:** The tool logs real-time scores directly to telemetry console displays:

    - Query 1 Similarity: ```0.5367``` ❌ (Fails isolated threshold)

    - Query 2 Similarity: ```0.6804``` ✅ (Passes threshold)

* **Automated Bypassing:** Because Query 2 safely matches past patterns above the 0.60 standard boundary, the conditional edge grade_documents prints ✅ Threshold Passed. Routing to answer generation. and avoids mutating the baseline question.

* **Structured Delivery:** The engine delivers a comprehensive audit answer matching structural indicators (anonymous IP proxies, fresh device fingerprints, off-hours timing mismatch) to confirm vulnerability profiles dynamically.