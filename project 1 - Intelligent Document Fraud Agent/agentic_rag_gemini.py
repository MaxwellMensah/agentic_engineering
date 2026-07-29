import os
import json
import time
import logging
from typing import Literal
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Initial setup for logging and environment variables
logging.basicConfig(level=logging.INFO)

# Load all variables from the local .env file directly into system environment
load_dotenv()

# Configuration Constants
LOCAL_FILE_PATH = "fraud_detection_dataset_V4.jsonl"
PERSIST_DIR = "./chromadb_frauddata_agentic"
COLLECTION_NAME = "langgraph_frauddata_agentic"
COLLECTION_METADATA = {"hnsw:space": "cosine"}
INGESTION_LIMIT = 500

# VERIFY LOCAL DATA ACCESSIBILITY
if os.path.exists(LOCAL_FILE_PATH):
    logging.info(f"📁 Success! '{LOCAL_FILE_PATH}' found in local directory workspace.")
else:
    raise FileNotFoundError(
        f"❌ Error: Could not locate '{LOCAL_FILE_PATH}' in your workspace root directory. "
        f"Please ensure you copied the file here before executing."
    )

# LOAD AND READ THE DATA
with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as file:
    first_line = file.readline()
    first_record = json.loads(first_line.strip())

# parse the JSONL nested structure
print(f"\n⚙️ Extracting exactly {INGESTION_LIMIT} records from conversational JSONL...")
doc_splits = []

with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file, 1):
        if not line.strip():
            continue

        if len(doc_splits) >= INGESTION_LIMIT:
            break

        data = json.loads(line.strip())
        conversations = data.get("conversations", [])

        user_query = ""
        assistant_analysis = ""

        for message in conversations:
            if message["role"] == "user":
                user_query = message["content"]
            elif message["role"] == "assistant":
                assistant_analysis = message["content"]

        combined_text = (
            f"TRANSACTION DETAILS: {user_query}\n"
            f"FRAUD ANALYSIS & REASONING: {assistant_analysis}"
        )

        extracted_user_id = (
            user_query.split(" ")[0] if "attempting" in user_query else f"ROW-{idx}"
        )

        metadata = {
            "case_id": f"FC-{idx:03d}",
            "target_user": extracted_user_id,
            "type": "Multi-Factor Fraud Check",
        }

        doc_splits.append(Document(page_content=combined_text, metadata=metadata))

logging.info(f"📊 Extracted {len(doc_splits)} operational records.")


# CUSTOM EMBEDDINGS CLASS WITH EXPONENTIAL BACKOFF
class RateLimitedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """Custom wrapper to gracefully scale back request volume and retry when hitting
    Gemini's free tier request-per-minute (100 RPM) limits.
    """

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        results = []
        sub_batch_size = 10  # Keeps the request chunk-size small and safe

        for i in range(0, len(texts), sub_batch_size):
            chunk = texts[i : i + sub_batch_size]

            retries = 5
            backoff = 5
            while retries > 0:
                try:
                    chunk_embeddings = super().embed_documents(chunk, **kwargs)
                    results.extend(chunk_embeddings)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        logging.warning(
                            f"⚠️ Rate limited on embedding sub-batch (items {i} to {i + len(chunk)}). "
                            f"Retrying in {backoff}s... ({retries} retries left)"
                        )
                        time.sleep(backoff)
                        backoff *= 2
                        retries -= 1
                    else:
                        raise e
            if retries == 0:
                raise RuntimeError(
                    "Failed to generate embeddings after multiple retries due to rate limits."
                )

            # Brief pause between successful internal sub-batches to pace rate usage
            if i + sub_batch_size < len(texts):
                time.sleep(1.5)

        return results

    def embed_query(self, text: str, **kwargs) -> list[float]:
        retries = 5
        backoff = 2
        while retries > 0:
            try:
                return super().embed_query(text, **kwargs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logging.warning(
                        f"⚠️ Rate limited on query embedding. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    retries -= 1
                else:
                    raise e
        raise RuntimeError("Failed to embed query after multiple retries.")


# Load Rate-Limited Google Gemini Embedding Engine
embeddings = RateLimitedGoogleEmbeddings(model="models/gemini-embedding-001")
logging.info("🚀 Generating embeddings with Google Gemini...gemini-embedding-001")


# initialize or refresh the Chroma Database
if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
    logging.info(
        "📁 Existing Vector Database found on disk. Loading cached collection safely..."
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata=COLLECTION_METADATA,
    )
else:
    logging.info(
        "✨ No local database found. Generating embeddings and initializing new Gemini collection..."
    )

    # 1. Spin up the Vector Store directory
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata=COLLECTION_METADATA,
    )

    # 2. Populate documents step-by-step
    batch_size = 40
    sleep_delay = 5  # Reduced since our inner class also sleeps and regulates backoff
    total_docs = len(doc_splits)

    logging.info(f"⏳ Ingesting {total_docs} documents in batches of {batch_size}...")
    for i in range(0, total_docs, batch_size):
        batch = doc_splits[i : i + batch_size]
        logging.info(
            f"📦 Writing batch {(i // batch_size) + 1}/{(total_docs - 1) // batch_size + 1}..."
        )

        vector_store.add_documents(batch)

        if i + batch_size < total_docs:
            logging.info(f"💤 Cooling off for {sleep_delay}s between batches...")
            time.sleep(sleep_delay)

    logging.info(
        f"✅ Successfully stored {len(doc_splits)} conversational records into local ChromaDB!"
    )


# Custom State to track how many times we've retried a query
class AgenticRAGState(MessagesState):
    retrieval_attempts: int


# Tools interface
@tool
def retrieve_fraud_cases(query: str) -> str:
    """Queries the internal historical vector database to scan for past fraud patterns,
    SIM swapping tricks, phishing signatures, and transaction anomalies.
    """
    results = vector_store.similarity_search_with_score(query, k=3)
    if not results:
        return "[STATUS: EMPTY_DB] No matching records found."

    best_doc, best_distance_score = results[0]
    best_similarity_score = 1.0 - best_distance_score

    print(
        f"📊 [Telemetry] Query: '{query}' | Fraud Match Similarity: {best_similarity_score:.4f}"
    )

    confident_docs = []
    for doc, distance in results:
        similarity = 1.0 - distance
        if similarity >= 0.6:
            confident_docs.append(
                f"Case ID: {doc.metadata.get('case_id')} | Details: {doc.page_content}"
            )

    if best_similarity_score >= 0.6:
        return (
            f"[CONFIDENCE: HIGH] (Top Match Score: {best_similarity_score:.2f})\n\n"
            + "\n\n".join(confident_docs)
        )
    else:
        return (
            f"[CONFIDENCE: LOW] The highest historical fraud match was {best_similarity_score:.2f}, "
            f"which falls below our required 0.6 threshold.\n"
            f"ACTION REQUIRED: The current query did not yield a confident historical match. "
            f"Optimize search terms to find broader fraud techniques or adjacent patterns."
        )


retriever_tool = retrieve_fraud_cases


# JUDGE MODEL SETUP (Gemini Flash via Google Generative AI)
response_model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
)
logging.info("🤖 Gemini Model Loaded Successfully")


# NODE PROMPTS
GENERATE_PROMPT = (
    "You are an expert financial fraud investigator. "
    "Use the following pieces of retrieved historical fraud cases to evaluate the user's inquiry. "
    "If the historical cases don't provide confident overlap, explain that this appears to be a novel exploit pattern. "
    "Keep your analysis concise and structured in three sentences maximum.\n"
    "Inquiry: {question} \n"
    "<historical_cases>\n{context}\n</historical_cases>"
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


# THE STATE INITIALIZATION NODE
def initialize_graph(state: AgenticRAGState):
    return {"retrieval_attempts": 0, "messages": state["messages"]}


# NODE FUNCTIONS
def generate_query_or_respond(state: AgenticRAGState):
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def generate_answer(state: AgenticRAGState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def rewrite_question(state: AgenticRAGState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])

    current_attempts = state.get("retrieval_attempts", 0)
    return {
        "retrieval_attempts": current_attempts + 1,
        "messages": [HumanMessage(content=response.content)],
    }


def grade_documents(
    state: AgenticRAGState,
) -> Literal["generate_answer", "rewrite_question", "fallback_empty_response"]:
    context_string = state["messages"][-1].content
    attempts = state.get("retrieval_attempts", 0)

    if "[CONFIDENCE: LOW]" in context_string:
        print(f"⚠️ Threshold Check Failed (Attempt {attempts + 1}/2).")

        if attempts >= 1:
            print("🛑 Max retrieval attempts reached. Routing to fallback.\n")
            return "fallback_empty_response"

        return "rewrite_question"

    print("✅ Threshold Passed. Routing to answer generation.\n\n")
    return "generate_answer"


def fallback_empty_response(state: AgenticRAGState):
    return {
        "messages": [
            HumanMessage(
                content="I'm sorry, I couldn't find any verified information regarding that topic in my document base."
            )
        ]
    }


# NODE AND GRAPH CONSTRUCTION
workflow = StateGraph(AgenticRAGState)

workflow.add_node("init", initialize_graph)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("fallback_empty_response", fallback_empty_response)

workflow.add_edge(START, "init")
workflow.add_edge("init", "generate_query_or_respond")


def route_on_tool_calls(state: AgenticRAGState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


workflow.add_conditional_edges(
    "generate_query_or_respond",
    route_on_tool_calls,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
        "fallback_empty_response": "fallback_empty_response",
    },
)

workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)
workflow.add_edge("fallback_empty_response", END)

graph = workflow.compile()
logging.info("🎉 Graph compiled successfully with production threshold safeguards!")

output_image_path = "graph_architecture.png"
try:
    with open(output_image_path, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    logging.info(f"💾 Graph visualization successfully saved to: {output_image_path}")
except Exception as e:
    logging.warning(f"Could not generate graph visualization image: {e}")


# STREAMING EXECUTION
if __name__ == "__main__":
    print("\n🔬 Running standalone test query...")
    test_state = {
        "messages": [
            {
                "role": "user",
                "content": "Analyze a transaction involving an NGN transfer from Sydney at late night using a new mobile device on a datacenter VPN.",
            }
        ]
    }
    for chunk in graph.stream(test_state):
        for node, update in chunk.items():
            logging.info(f"Update from node: {node}")
            if "messages" in update and len(update["messages"]) > 0:
                update["messages"][-1].pretty_print()
