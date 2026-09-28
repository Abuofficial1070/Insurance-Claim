# 🏥 AI-Powered Insurance Claim Verification System

An AI-powered insurance claim verification system that combines **Machine Learning, Retrieval-Augmented Generation (RAG), document verification, claim-history analysis, and a deterministic decision engine** to evaluate insurance claims.

The system is designed to reduce manual claim-processing effort while avoiding a major limitation of simple AI-based systems: allowing an LLM to make the final approval/rejection decision.

---

## 🚀 Project Overview

Insurance claim processing usually requires checking multiple sources of information:

- Claim details
- Insurance policy
- Medical reports
- Hospital bills
- Hospital/network eligibility
- Previous claim history
- Fraud risk

This project brings these verification steps into a single application.

The system uses:

- **Random Forest** for fraud-risk prediction
- **RAG + FAISS** for insurance policy verification
- **Document verification** for consistency checking
- **SQLite** for claim-history storage
- **Rule-based Decision Engine** for the final decision
- **Gemini AI** for natural-language explanation

### Important Design Principle

> **Gemini does not make the final insurance decision.**

The final decision is produced by a deterministic Python Decision Engine based on verified evidence.

---

# 🧠 System Architecture

```text
                         USER CLAIM
                             │
                             ▼
                          FLASK
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
        CLAIM DATA       DOCUMENTS       POLICY PDF
             │               │               │
             ▼               ▼               ▼
      Claim History    Document Check       RAG
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                     FEATURE / EVIDENCE
                         PROCESSING
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        RANDOM FOREST             POLICY VERIFICATION
                │                         │
                ▼                         │
        Fraud Probability                  │
                │                         │
                └────────────┬────────────┘
                             ▼
                     DECISION ENGINE
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
             APPROVED   MANUAL REVIEW   REJECTED
                             │
                             ▼
                       GEMINI AI
                       Explanation
                             │
                             ▼
                        RESULT PAGE
