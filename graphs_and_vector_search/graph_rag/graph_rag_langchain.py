import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

# Database & API Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Strict Fail-Fast Validation
required_vars = {
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USER": NEO4J_USER,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing_vars = [key for key, val in required_vars.items() if not val]
if missing_vars:
    raise RuntimeError(f"Missing required .env variables: {', '.join(missing_vars)}")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Gemini Flash Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.0
)


def fetch_subgraph_context(keyword: str) -> str:
    """Queries Neo4j for 2-hop connected entities and relationships."""

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
        return "No relevant context found in Knowledge Graph."

    facts = [
        f"- [{r['Source']}] --({r['Rel']})--> [{r['Target']}] ({r['TargetType']})"
        for r in records
    ]
    return "\n".join(facts)


def ask_graph_rag(question: str, keyword: str) -> str:
    """Retrieves graph context and invokes Gemini using LangChain schema."""
    context = fetch_subgraph_context(keyword)

    system_instruction = (
        "You are a Fraud Risk Analyst AI. Answer the user's question STRICTLY "
        "using the provided Knowledge Graph facts. If the information is missing, "
        "explicitly state that the database lacks sufficient context."
    )

    user_prompt = f"Knowledge Graph Facts:\n{context}\n\nQuestion: {question}\nAnswer:"

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    q = "What document types and forgery methods are associated with Identity Theft?"
    key = "Identity Theft"

    print(f"Question: {q}\n")
    print("--- Grounded Gemini LLM Response ---")
    print(ask_graph_rag(q, key))

    driver.close()
