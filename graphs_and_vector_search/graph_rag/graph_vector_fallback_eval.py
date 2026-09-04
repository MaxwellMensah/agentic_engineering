import csv
import os

import chromadb
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Drivers & Clients
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Mock ChromaDB Vector Store for Fallback
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="fraud_unstructured_docs")
collection.add(
    documents=[
        "Unstructured Report: Synthetic credit fraud frequently uses fake utility bills to pass address verification.",
        "Security Alert: Font inconsistencies in bank statements indicate digital image tampering.",
    ],
    ids=["doc1", "doc2"],
)


def fetch_graph_context(keyword: str):
    """Retrieves subgraph context and computes a heuristic graph confidence score."""
    cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($keyword)
        MATCH path = (n)-[r:LEADS_TO|RELATED_TO*1..2]-(connected)
        RETURN n.name AS Source, 
            type(r[0]) AS Rel, 
            connected.name AS Target, 
            labels(connected)[0] AS TargetType
        LIMIT 20
    """
    with driver.session() as session:
        records = session.run(cypher, keyword=keyword).data()

    if not records:
        return "", 0.0  # Confidence 0 if no nodes match

    facts = [
        f"- [{r['Source']}] --({r['Rel']})--> [{r['Target']}] ({r['TargetType']})"
        for r in records
    ]

    # Confidence threshold calculated based on returned relationship density (scale 0.0 - 1.0)
    confidence = min(len(records) / 5.0, 1.0)
    return "\n".join(facts), confidence


def vector_search_fallback(query: str) -> str:
    """Executes vector similarity search in ChromaDB when graph confidence is low."""
    results = collection.query(query_texts=[query], n_results=1)
    docs = results.get("documents", [[]])[0]
    return docs[0] if docs else "No vector context found."


def run_graph_rag(question: str, keyword: str):
    """Executes Graph RAG with ChromaDB vector fallback if graph confidence < 0.6."""
    graph_context, confidence = fetch_graph_context(keyword)
    retrieval_source = "Graph"

    if confidence < 0.6:
        retrieval_source = "Vector Fallback (ChromaDB)"
        context = vector_search_fallback(question)
    else:
        context = graph_context

    prompt = f"""Answer strictly using the context below.

                Context ({retrieval_source}):
                {context}

                Question: {question}
                Answer:
            """

    response = ai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )

    return response.text.strip(), confidence, retrieval_source


def run_eval_harness():
    """Runs test evaluation harness and appends results to eval_results.csv."""
    test_cases = [
        {
            "question": "What forgery methods lead to Identity Theft?",
            "keyword": "Identity Theft",
        },
        {
            "question": "How do utility bills relate to fraud?",
            "keyword": "Utility Bill",
        },
        {
            "question": "What is wire fraud policy?",
            "keyword": "Wire Fraud",
        },  # Triggers fallback
    ]

    csv_file = "eval_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Question", "Keyword", "ConfidenceScore", "SourceUsed", "Answer"]
            )

        for case in test_cases:
            ans, conf, source = run_graph_rag(case["question"], case["keyword"])
            writer.writerow(
                [case["question"], case["keyword"], f"{conf:.2f}", source, ans]
            )
            print(
                f"Evaluated: '{case['question']}' | Source: {source} | Conf: {conf:.2f}"
            )


if __name__ == "__main__":
    print("--- Executing Evaluation Harness ---")
    run_eval_harness()
    print("\nEvaluation complete. Scores logged to eval_results.csv.")
    driver.close()
