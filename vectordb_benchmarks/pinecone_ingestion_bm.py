from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import time
import os


load_dotenv()  # Load environment variables from .env file

# Load real fraud cases from JSONL file (same as ChromaDB Part 1)
fraud_cases = []
with open("fraud_detection_dataset_V4.jsonl", "r") as f:
    for line_num, line in enumerate(f, start=1):
        data = json.loads(line.strip())
        # Adapt field names to match your JSON structure
        case_text = data.get(
            "description", data.get("text", data.get("transaction_detail", str(data)))
        )
        severity = data.get("severity", data.get("risk_level", "medium"))
        fraud_cases.append(
            {
                "id": f"case_{line_num}",
                "text": case_text,
                "severity": severity,
                "raw": data,  # keep original if needed (consistent with ChromaDB)
            }
        )
        if line_num >= 100:  # only first 100 cases
            break

print(f"Loaded {len(fraud_cases)} fraud cases from JSONL")

# Load the SAME embedding model as ChromaDB (all-MiniLM-L6-v2)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings manually (same as ChromaDB Part 1)
texts = [case["text"] for case in fraud_cases]
embeddings = model.encode(texts).tolist()  # list of 384‑dim vectors

# Pinecone setup (manual vectors, no auto‑embedding)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "fraud-cases-index"

# Delete existing index if it exists (for clean test)
if pc.has_index(index_name):
    pc.delete_index(index_name)

# Create serverless index with dimension 384 (matching the embedding model)
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(f"Index '{index_name}' created")

# Connect to the index
index = pc.Index(index_name)

# Prepare vectors for upsert (id, vector, metadata)
vectors = []
for i, case in enumerate(fraud_cases):
    vectors.append((case["id"], embeddings[i], {"severity": case["severity"]}))

# Upsert vectors to Pinecone
start_time = time.time()
index.upsert(vectors=vectors, namespace="fraud_cases")
pinecone_ingest_time = time.time() - start_time
print(f"✅ Pinecone ingest time: {pinecone_ingest_time:.2f} seconds")

# Query: embed query with the same model
query_text = "suspicious transaction pattern"
query_embedding = model.encode([query_text]).tolist()[0]

start_time = time.time()
query_results = index.query(
    vector=query_embedding, top_k=5, namespace="fraud_cases", include_metadata=True
)
pinecone_query_time = time.time() - start_time
print(f"⏱️ Pinecone query time: {pinecone_query_time:.4f} seconds")
print(f"🏆 Top 5 similar cases: {[item['id'] for item in query_results['matches']]}")


# Index 'fraud-cases-index' created
# ✅ Pinecone ingest time: 9.03 seconds
# ⏱️ Pinecone query time: 0.2722 seconds
# 🏆 Top 5 similar cases: ['case_16', 'case_52', 'case_79', 'case_32', 'case_51']