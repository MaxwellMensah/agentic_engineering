## Python Graph RAG Query Engine

## 1. Architecture Summary
* **Database**: Neo4j Desktop (`neo4j://localhost:7687`)
* **LLM Integration**: Gemini 2.5 Flash via `google-genai` native SDK
* **Retrieval Strategy**: 2-hop graph expansion converting Cypher path records into plain-text facts
* **Prompting Strategy**: Strict system-level grounding to prevent hallucination

## 2. Test Execution Output
* **Query**: "What document types and forgery methods are associated with Identity Theft?"
* **Retrieved Subgraph Facts**:
  - `Photo Replacement` --(LEADS_TO)--> `Identity Theft`
  - `Photo Replacement` --(RELATED_TO)--> `Passport`
  - `Identity Theft` --(RELATED_TO)--> `Synthetic Credit Fraud`
  - `Font Inconsistency` / `EXIF Metadata Alteration` --(LEADS_TO)--> `Synthetic Credit Fraud`
* **Grounded Answer**: Identified `Passport` alongside `Photo Replacement`, `Font Inconsistency`, and `EXIF Metadata Alteration`.