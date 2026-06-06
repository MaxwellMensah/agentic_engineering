import chromadb
from sentence_transformers import SentenceTransformer
import json
import time

# Load real fraud cases from JSONL file
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
                "raw": data,  # keep original if needed
            }
        )
        if line_num >= 100:  # only first 100 cases
            break

print(f"Loaded {len(fraud_cases)} fraud cases from JSONL")

# ChromaDB setup
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db_store")
collection = client.get_or_create_collection(name="fraud_cases")

# Generate embeddings manually
texts = [case["text"] for case in fraud_cases]
embeddings = model.encode(texts).tolist()

# Ingest (pass pre‑computed embeddings)
start_time = time.time()
collection.add(
    ids=[case["id"] for case in fraud_cases],
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"severity": case["severity"]} for case in fraud_cases],
)
chroma_ingest_time = time.time() - start_time
print(f"✅ ChromaDB ingest time: {chroma_ingest_time:.2f} seconds")

# Query: embed query manually
query_text = "suspicious transaction pattern"
query_embedding = model.encode([query_text]).tolist()[0]
start_time = time.time()
results = collection.query(query_embeddings=[query_embedding], n_results=5)
chroma_query_time = time.time() - start_time
print(f"⏱️ ChromaDB query time: {chroma_query_time:.4f} seconds")
print(f"🏆 Top 5 similar cases: {results['ids'][0]}")


# ✅ ChromaDB ingest time: 0.09 seconds
# ⏱️ ChromaDB query time: 0.0150 seconds
# 🏆 Top 5 similar cases: ['case_16', 'case_52', 'case_79', 'case_32', 'case_51']