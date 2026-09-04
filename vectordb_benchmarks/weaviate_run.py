import os
import random

import chromadb
import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Generate 100 Synthetic Fraud Cases
fraud_types = [
    "Identity Theft",
    "Synthetic Credit Fraud",
    "Wire Fraud",
    "Account Takeover",
]
docs = ["Passport", "Utility Bill", "Bank Statement", "Drivers License"]
methods = [
    "Photo Replacement",
    "Font Inconsistency",
    "EXIF Metadata Alteration",
    "VPN Proxy Routing",
]

dataset = []
for i in range(1, 101):
    case_id = f"CASE-{1000 + i}"
    ftype = random.choice(fraud_types)
    doc = random.choice(docs)
    method = random.choice(methods)
    account_num = f"ACC-{random.randint(10000, 99999)}"

    desc = f"Investigation for {case_id} involving {ftype}. Targeted document: {doc} using {method}. Associated account: {account_num}."
    dataset.append(
        {
            "case_id": case_id,
            "description": desc,
            "fraud_type": ftype,
            "account_num": account_num,
            "risk_score": round(random.uniform(0.5, 0.99), 2),
        }
    )

print(f"Generated {len(dataset)} synthetic fraud cases.")

# Ingest into ChromaDB & Extract Computed Embeddings
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="fraud_cases")

chroma_collection.add(
    documents=[item["description"] for item in dataset],
    metadatas=[
        {"case_id": item["case_id"], "fraud_type": item["fraud_type"]}
        for item in dataset
    ],
    ids=[item["case_id"] for item in dataset],
)

# Extract computed vectors from ChromaDB for transfer
chroma_data = chroma_collection.get(include=["embeddings"])
embeddings_map = {
    case_id: emb for case_id, emb in zip(chroma_data["ids"], chroma_data["embeddings"])
}
print("Populated ChromaDB and extracted 100 dense vector embeddings.")

# Connect & Batch Ingest into Weaviate
if WEAVIATE_URL and WEAVIATE_API_KEY:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
    )
else:
    client = weaviate.connect_to_embedded()

try:
    if client.collections.exists("FraudCase"):
        client.collections.delete("FraudCase")

    fraud_collection = client.collections.create(
        name="FraudCase",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="case_id", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="fraud_type", data_type=DataType.TEXT),
            Property(name="account_num", data_type=DataType.TEXT),
            Property(name="risk_score", data_type=DataType.NUMBER),
        ],
    )

    # Ingest objects WITH explicit vectors
    with fraud_collection.batch.dynamic() as batch:
        for item in dataset:
            doc_vector = embeddings_map[item["case_id"]]
            batch.add_object(properties=item, vector=doc_vector)

    print("Successfully migrated 100 cases + vectors into Weaviate.")

    # Compare Chroma Dense vs Weaviate Hybrid Search
    target_account = dataset[0]["account_num"]
    query = f"Wire Fraud involving {target_account} Utility Bill"

    print(f"\n--- QUERY: '{query}' ---")

    # A. ChromaDB Dense Search
    chroma_res = chroma_collection.query(query_texts=[query], n_results=2)
    print("\n[ChromaDB Dense Vector Search Results]:")
    for doc in chroma_res["documents"][0]:
        print(f"  - {doc}")

    # B. Weaviate Hybrid Search (BM25 Keyword + Explicit Vector)
    query_vector = chroma_collection._embedding_function([query])[0]

    weaviate_res = fraud_collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=0.5,
        limit=2,
        return_metadata=MetadataQuery(score=True),
    )

    print("\n[Weaviate Hybrid Search (BM25 + Vector) Results]:")
    for obj in weaviate_res.objects:
        score = obj.metadata.score
        print(f"  - [Score: {score:.4f}] {obj.properties['description']}")

finally:
    client.close()
