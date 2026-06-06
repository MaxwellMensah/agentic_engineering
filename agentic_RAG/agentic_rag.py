import os
import bs4
import json
import boto3
import shutil
import requests
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
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initial setup for logging and environment variables
logging.basicConfig(level=logging.INFO)

# Load all variables from the local .env file directly into system environment
load_dotenv()

# Extract keys cleanly for your AWS Bedrock and LangGraph components
aws_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")

# Configuration Constants
LOCAL_FILE_PATH = "fraud_detection_dataset_V4.jsonl"
PERSIST_DIR = "./chromadb_frauddata_agentic"
COLLECTION_NAME = "langgraph_frauddata_agentic"
COLLECTION_METADATA = {"hnsw:space": "cosine"}
INGESTION_LIMIT = 500

# Initialize the low-level bedrock control plane client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_id,
    aws_secret_access_key=aws_secret,
)

# VERIFY LOCAL DATA ACCESSIBILITY
if os.path.exists(LOCAL_FILE_PATH):
    logging.info(f"📁 Success! '{LOCAL_FILE_PATH}' found in local directory workspace.")
else:
    raise FileNotFoundError(
        f"❌ Error: Could not locate '{LOCAL_FILE_PATH}' in your workspace root directory. "
        f"Please ensure you copied the file here before executing."
    )

# LOAD AND READ THE DATA

# open the locally copied file path
with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as file:
    # read only the first line
    first_line = file.readline()

    # parse the string into a clean Python dictionary
    first_record = json.loads(first_line.strip())

print("🔑 === FIRST RECORD KEY-VALUE PAIRS === 🔑\n")
print(json.dumps(first_record, indent=4))

# parse the JSONL nested structure
print(f"\n⚙️ Extracting exactly {INGESTION_LIMIT} records from conversational JSONL...")
doc_splits = []

with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file, 1):
        if not line.strip():
            continue

        # stop loading as soon as we reach our target slice
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

        # build an information-rich payload for the embedding model
        combined_text = (
            f"TRANSACTION DETAILS: {user_query}\n"
            f"FRAUD ANALYSIS & REASONING: {assistant_analysis}"
        )

        # pull out the unique user tracking key if visible (e.g., U-2604379)
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


# Load the embedding model via Bedrock
embeddings = BedrockEmbeddings(
    client=bedrock_client, model_id="amazon.titan-embed-text-v2:0"
)
logging.info("🚀 Generating embeddings with Bedrock...titan-embed-text-v2")


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
        "✨ No local database found. Generating embeddings and initializing new AWS collection..."
    )
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        collection_metadata=COLLECTION_METADATA,
    )
    logging.info(
        f"✅ Successfully stored {len(doc_splits)} conversational records into local ChromaDB!"
    )


# Custom State to track how many times we've retried a query
class AgenticRAGState(MessagesState):
    retrieval_attempts: int


# Tools are the interface for your graph's nodes to interact with external systems, databases, or APIs. 
# This tool queries our internal vector database for historical fraud cases that match the user's inquiry. 
# The tool's output is carefully formatted to include confidence scores and structured information for the judge model to evaluate. 
@tool
def retrieve_fraud_cases(query: str) -> str:
    """Queries the internal historical vector database to scan for past fraud patterns,
    SIM swapping tricks, phishing signatures, and transaction anomalies.
    """

    # The internal logic remains EXACTLY the same!
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
            # Formatting the output nicely for the fraud investigator agent
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


# pass the updated tool to your graph's ToolNode
retriever_tool = retrieve_fraud_cases


# JUDGE MODEL SETUP (Claude Haiku 4.5 via Bedrock)
response_model = ChatBedrockConverse(
    region_name="us-east-1",
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",  # or claude-3-5-sonnet
    aws_access_key_id=aws_id,
    aws_secret_access_key=aws_secret,
    temperature=0.0,
)
logging.info(" Model Loaded")


# NODE PROMPTS
# Generate query prompt for the retriever tool
GENERATE_PROMPT = (
    "You are an expert financial fraud investigator. "
    "Use the following pieces of retrieved historical fraud cases to evaluate the user's inquiry. "
    "If the historical cases don't provide confident overlap, explain that this appears to be a novel exploit pattern. "
    "Keep your analysis concise and structured in three sentences maximum.\n"
    "Inquiry: {question} \n"
    "<historical_cases>\n{context}\n</historical_cases>"
)

# Rewrite answers prompt for the judge model to improve the question if the initial retrieval confidence is low
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
    """Ensures our retry tracker starts at 0 when a new question enters."""
    return {"retrieval_attempts": 0, "messages": state["messages"]}


# NODE FUNCTIONS
# Main ReAct Router Engine
def generate_query_or_respond(state: AgenticRAGState):
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# Document Generator Node
def generate_answer(state: AgenticRAGState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# Query Optimizer Node
def rewrite_question(state: AgenticRAGState):
    """Rewrites the original user question and increments the attempt counter."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])

    # Increment our retry state counter
    current_attempts = state.get("retrieval_attempts", 0)
    return {
        "retrieval_attempts": current_attempts + 1,
        "messages": [HumanMessage(content=response.content)],
    }


# The systematic grader edge: checks the retrieved document's confidence score and decides the next step in the flow
def grade_documents(
    state: AgenticRAGState,
) -> Literal["generate_answer", "rewrite_question", "fallback_empty_response"]:
    """Determines whether the retrieved documents passed the 0.6 threshold."""

    # grab what our tool just returned
    context_string = state["messages"][-1].content
    attempts = state.get("retrieval_attempts", 0)

    # if the tool flagged a low score, intercept control flow!
    if "[CONFIDENCE: LOW]" in context_string:
        print(f"⚠️ Threshold Check Failed (Attempt {attempts + 1}/2).")

        # Guardrail: if we already retried twice, stop burning tokens and fail gracefully
        if attempts >= 1:
            print("🛑 Max retrieval attempts reached. Routing to fallback.\n")
            return "fallback_empty_response"

        return "rewrite_question"

    print("✅ Threshold Passed. Routing to answer generation.\n\n")
    return "generate_answer"


# Fallback Terminal Node
def fallback_empty_response(state: AgenticRAGState):
    return {
        "messages": [
            HumanMessage(
                content="I'm sorry, I couldn't find any verified information regarding that topic in my document base."
            )
        ]
    }


# NODE AND GRAPH CONSTRUCTION
# re-initialize workflow cleanly using your updated State tracker class
workflow = StateGraph(AgenticRAGState)

# assign your nodes
workflow.add_node("init", initialize_graph)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("fallback_empty_response", fallback_empty_response)

# Setup structural routing flow strings
workflow.add_edge(START, "init")
workflow.add_edge("init", "generate_query_or_respond")


def route_on_tool_calls(state: AgenticRAGState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# decide to call a tool or finish
workflow.add_conditional_edges(
    "generate_query_or_respond",
    route_on_tool_calls,
    {
        "tools": "retrieve",
        END: END,
    },
)

# SYSTEMATIC ACTION: Connect the confidence checker output to its target nodes
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
        "fallback_empty_response": "fallback_empty_response",
    },
)

# direct loops back to retry searching
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# close the execution ends
workflow.add_edge("generate_answer", END)
workflow.add_edge("fallback_empty_response", END)

# compile into an app executable
graph = workflow.compile()
logging.info("🎉 Graph compiled successfully with production threshold safeguards!")


# extract the raw binary image data and write it directly to disk
output_image_path = "graph_architecture.png"

with open(output_image_path, "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

logging.info(f"💾 Graph visualization successfully saved to: {output_image_path}")


# STREAMING EXECUTION
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze a transaction involving an NGN transfer from Sydney at late night using a new mobile device on a datacenter VPN.",
            }
        ]
    }
):
    for node, update in chunk.items():
        logging.info(f"Update from node: {node}")

        # pretty_print as it cleanly outputs to standard out
        update["messages"][-1].pretty_print()
        # use a raw print statement for visual line breaks to avoid logging format pollution
        print("\n\n")
