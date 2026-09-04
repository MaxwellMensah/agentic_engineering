import json
from pathlib import Path

RAW_CORPUS = [
    {
        "id": "chunk_wf_01",
        "domain": "wire_fraud",
        "text": "Case #WF-2024-8891: Wire transfer velocity breach detected on account 882190. Multiple high-value transfers totaling $245,000 sent to offshore beneficiary entities in under 12 hours without prior transaction history. Flagged under AML Rule 403.",
    },
    {
        "id": "chunk_wf_02",
        "domain": "wire_fraud",
        "text": "Business Email Compromise (BEC) protocol: Fraudsters execute unauthorized SWIFT wire requests by spoofing C-level executive email domains. Primary verification requires dual-factor out-of-band phone authentication for any transfer exceeding $50,000.",
    },
    {
        "id": "chunk_wf_03",
        "domain": "wire_fraud",
        "text": "AI deepfake voice impersonation fraud: Fraudsters clone executive voices to authorize urgent wire approvals via telephone banking. Requires voice biometric confidence scoring below 85% to trigger mandatory secondary callback verification.",
    },
    {
        "id": "chunk_ato_01",
        "domain": "account_takeover",
        "text": "Account Takeover (ATO) analysis: Incident ID ATO-9912. User credentials compromised via credential stuffing attack from IP range 192.0.2.0/24. Subsequent SIM swap allowed bypass of SMS OTP, leading to unauthorized password reset and email change.",
    },
    {
        "id": "chunk_ato_02",
        "domain": "account_takeover",
        "text": "Device fingerprint anomaly detection: When a logged-in session changes device User-Agent and canvas hash mid-session while performing high-risk actions (e.g., adding a new payout account), freeze outgoing transfers for 24 hours.",
    },
    {
        "id": "chunk_syn_01",
        "domain": "synthetic_identity",
        "text": "Synthetic Identity Fraud (SIF) investigation: Suspect created synthetic profiles using valid SSNs from unallocated pool combined with fabricated names and addresses. Credit header checks revealed zero historical credit bureau footprint prior to 30 days ago.",
    },
    {
        "id": "chunk_syn_02",
        "domain": "synthetic_identity",
        "text": "Piggybacking credit building: Synthetic identity rings apply for authorized user status on established credit card accounts to quickly boost credit scores before executing 'bust-out' card max-outs.",
    },
    {
        "id": "chunk_cb_01",
        "domain": "chargeback",
        "text": "Merchant dispute guidelines: Visa Reason Code 10.4 refers to Fraudulent Transaction - Card-Absent Environment. Compelling evidence requires IP logs, AVS match confirmation, device fingerprint matching, and proof of digital download or physical delivery.",
    },
    {
        "id": "chunk_cb_02",
        "domain": "chargeback",
        "text": "Regulation E electronic fund transfer claims: Consumers must report unauthorized electronic transfers within 60 days of statement issuance. Financial institutions have 10 business days to investigate and issue provisional credit.",
    },
    {
        "id": "chunk_cb_03",
        "domain": "chargeback",
        "text": "Friendly fraud and buyer remorse: First-party dispute abuse occurs when legimate cardholders claim non-receipt of goods despite verified biometric checkout and delivery confirmation at primary billing address.",
    },
    {
        "id": "chunk_aml_01",
        "domain": "anti_money_laundering",
        "text": "Structuring and Smurfing: Case #AML-7703. Customer made 12 consecutive cash deposits of $9,800 across 4 different branch locations over two days to circumvent the $10,000 Bank Secrecy Act (BSA) Currency Transaction Report (CTR) threshold.",
    },
    {
        "id": "chunk_aml_02",
        "domain": "anti_money_laundering",
        "text": "Cryptocurrency off-ramping risk scoring: High-risk indicators include rapid conversion of privacy coins (e.g., Monero) to stablecoins, followed by immediate wire withdrawal to non-cooperative jurisdictions (NCCT).",
    },
    {
        "id": "chunk_aml_03",
        "domain": "anti_money_laundering",
        "text": "Suspicious Activity Report (SAR) filing criteria: Mandatory filing within 30 days for any transaction involving $5,000 or more where the institution knows, suspects, or has reason to suspect potential money laundering or tax evasion.",
    },
    {
        "id": "chunk_app_01",
        "domain": "authorized_push_payment",
        "text": "Authorized Push Payment (APP) fraud: Scam victims are manipulated into initiating irrevocable Real-Time Payments (RTP) or FedNow instant transfers to fraudulent recipient accounts. Bank liability policies vary based on victim authorization.",
    },
    {
        "id": "chunk_app_02",
        "domain": "authorized_push_payment",
        "text": "Mule account routing and rapid dissipation: Case #APP-3310. Fraudulent proceeds from push payments are split into sub-$2,000 transfers across a network of tier-1 mule accounts before instant crypto off-ramping.",
    },
]

# --- 20 Annotated Benchmark Queries ---
EVAL_QUERIES = [
    {
        "query_id": "q_01",
        "query": "What happened in Case #WF-2024-8891 regarding wire transfer velocity?",
        "relevant_ids": ["chunk_wf_01"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_02",
        "query": "How to handle executive email domain spoofing and fraudulent SWIFT payments?",
        "relevant_ids": ["chunk_wf_02"],
        "category": "semantic",
    },
    {
        "query_id": "q_03",
        "query": "What steps are involved when a user account is hijacked via credential stuffing and SIM swapping?",
        "relevant_ids": ["chunk_ato_01", "chunk_ato_02"],
        "category": "multi_doc_semantic",
    },
    {
        "query_id": "q_04",
        "query": "What is the policy for mid-session device fingerprint changes during payout account setup?",
        "relevant_ids": ["chunk_ato_02"],
        "category": "semantic",
    },
    {
        "query_id": "q_05",
        "query": "How do fraudsters use fake identities with unallocated SSNs to pass credit checks?",
        "relevant_ids": ["chunk_syn_01"],
        "category": "semantic",
    },
    {
        "query_id": "q_06",
        "query": "What is credit piggybacking and how does it relate to bust-out card max-outs?",
        "relevant_ids": ["chunk_syn_02"],
        "category": "semantic",
    },
    {
        "query_id": "q_07",
        "query": "What evidence is needed to fight Visa Reason Code 10.4 chargebacks?",
        "relevant_ids": ["chunk_cb_01"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_08",
        "query": "What is the timeline under Regulation E for provisional credit during fraud disputes?",
        "relevant_ids": ["chunk_cb_02"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_09",
        "query": "How do cardholders abuse chargebacks after receiving delivery at their primary address?",
        "relevant_ids": ["chunk_cb_03"],
        "category": "semantic",
    },
    {
        "query_id": "q_10",
        "query": "How to spot deposits structured under $10,000 to avoid CTR reporting in Case #AML-7703?",
        "relevant_ids": ["chunk_aml_01"],
        "category": "hybrid_exact_semantic",
    },
    {
        "query_id": "q_11",
        "query": "AML red flags for Monero and cryptocurrency cash outs to high-risk offshore accounts",
        "relevant_ids": ["chunk_aml_02"],
        "category": "semantic",
    },
    {
        "query_id": "q_12",
        "query": "When is a Suspicious Activity Report (SAR) filing required for transfers over $5,000?",
        "relevant_ids": ["chunk_aml_03"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_13",
        "query": "What are the fraud risk protocols for FedNow and Real-Time Payments (RTP) push transfers?",
        "relevant_ids": ["chunk_app_01"],
        "category": "semantic",
    },
    {
        "query_id": "q_14",
        "query": "How do money mule networks dissipate push payment funds in Case #APP-3310?",
        "relevant_ids": ["chunk_app_02"],
        "category": "hybrid_exact_semantic",
    },
    {
        "query_id": "q_15",
        "query": "What verification is required if AI voice cloning is used to request urgent wire transfers?",
        "relevant_ids": ["chunk_wf_03"],
        "category": "semantic",
    },
    {
        "query_id": "q_16",
        "query": "Incident ID ATO-9912 IP range 192.0.2.0/24 attack log details",
        "relevant_ids": ["chunk_ato_01"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_17",
        "query": "Rule 403 velocity limit breaches on account 882190",
        "relevant_ids": ["chunk_wf_01"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_18",
        "query": "What rules govern dual-factor phone callbacks for wire transfers over $50,000?",
        "relevant_ids": ["chunk_wf_02"],
        "category": "hybrid_exact_semantic",
    },
    {
        "query_id": "q_19",
        "query": "Bank Secrecy Act CTR threshold currency deposit limits",
        "relevant_ids": ["chunk_aml_01"],
        "category": "exact_keyword",
    },
    {
        "query_id": "q_20",
        "query": "How to detect canvas hash changes and new payout account additions in user sessions",
        "relevant_ids": ["chunk_ato_02"],
        "category": "semantic",
    },
]


def generate_dataset(output_dir: str = "."):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    corpus_file = out_path / "fraud_corpus.json"
    eval_file = out_path / "fraud_eval_dataset.json"

    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(RAW_CORPUS, f, indent=2)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(EVAL_QUERIES, f, indent=2)

    print(f"Generated {len(RAW_CORPUS)} corpus chunks -> {corpus_file}")
    print(f"Generated {len(EVAL_QUERIES)} benchmark queries -> {eval_file}")


if __name__ == "__main__":
    generate_dataset()
