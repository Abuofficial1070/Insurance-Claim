"""
app.py — Insurance Claim Verification (refactored)

Architecture:
  1. Receive form data + uploaded files
  2. Save files
  3. Authoritative claim history from SQLite
  4. Document verification
  5. Build policy RAG + extract policy facts
  6. Random Forest fraud probability (predict_proba)
  7. Deterministic Decision Engine
  8. Gemini explanation only
  9. Persist results to SQLite
 10. Render result.html
"""

import os
import json
import sqlite3

import joblib
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from rag import build_vector_store, search_policy
from claim_history import get_claim_history
from document_verifier import verify_documents
from policy_verifier import verify_policy
from decision_engine import evaluate_claim, risk_level
from insurance_ai import analyze_claim

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = "insurance.db"

# ---------------------------------------------------------------------------
# ML model
# ---------------------------------------------------------------------------
model = joblib.load("fraud_model.pkl")
features = joblib.load("features.pkl")

# ---------------------------------------------------------------------------
# Database initialisation (safe migration — adds columns if absent)
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims(
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        name              TEXT,
        age               INTEGER,
        policy            TEXT,
        gender            TEXT,
        hospital          TEXT,
        disease           TEXT,
        insurance         TEXT,
        previous_claims   INTEGER,
        amount            INTEGER,
        medical_report    TEXT,
        hospital_bill     TEXT,
        policy_pdf        TEXT,
        prediction        TEXT,
        fraud_probability REAL,
        risk_level        TEXT,
        final_decision    TEXT,
        risk_flags        TEXT,
        policy_verified   TEXT,
        document_verified TEXT
    )
    """)

    # Safe migration: add new columns if they don't exist yet
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(claims)")}
    new_columns = {
        "fraud_probability": "REAL",
        "risk_level": "TEXT",
        "final_decision": "TEXT",
        "risk_flags": "TEXT",
        "policy_verified": "TEXT",
        "document_verified": "TEXT",
    }
    for col, col_type in new_columns.items():
        if col not in existing:
            cursor.execute(f"ALTER TABLE claims ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


init_db()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ------------------------------------------------------------------
    # 1. Receive form data
    # ------------------------------------------------------------------
    name             = request.form["name"].strip()
    age              = int(request.form["age"])
    policy           = request.form["policy"].strip()
    gender           = request.form["gender"]
    hospital         = request.form["hospital"].strip()
    disease          = request.form["disease"].strip()
    insurance        = request.form["insurance"]
    previous_claims  = int(request.form.get("previous_claims", 0))
    amount           = int(request.form["amount"])

    # ------------------------------------------------------------------
    # 2. Save uploaded files
    # ------------------------------------------------------------------
    medical_report = request.files["medical_report"]
    hospital_bill  = request.files["hospital_bill"]
    policy_pdf     = request.files["policy_pdf"]

    medical_filename = secure_filename(medical_report.filename)
    bill_filename    = secure_filename(hospital_bill.filename)
    policy_filename  = secure_filename(policy_pdf.filename)

    medical_path = os.path.join(app.config["UPLOAD_FOLDER"], medical_filename)
    bill_path    = os.path.join(app.config["UPLOAD_FOLDER"], bill_filename)
    policy_path  = os.path.join(app.config["UPLOAD_FOLDER"], policy_filename)

    medical_report.save(medical_path)
    hospital_bill.save(bill_path)
    policy_pdf.save(policy_path)

    # ------------------------------------------------------------------
    # 3. Authoritative claim history from SQLite
    # ------------------------------------------------------------------
    history = get_claim_history(
        db_path=DB_PATH,
        name=name,
        policy=policy,
        hospital=hospital,
        disease=disease,
        amount=amount,
        entered_count=previous_claims,
    )

    # ------------------------------------------------------------------
    # 4. Document verification
    # ------------------------------------------------------------------
    documents = verify_documents(
        medical_path=medical_path,
        bill_path=bill_path,
        name=name,
        hospital=hospital,
        disease=disease,
        amount=amount,
    )

    # ------------------------------------------------------------------
    # 5. Policy RAG + fact extraction
    # ------------------------------------------------------------------
    index, chunks = build_vector_store(policy_path)
    policy_snippet = search_policy(disease, index, chunks)
    policy_facts   = verify_policy(policy_path, disease, hospital)

    # ------------------------------------------------------------------
    # 6. Random Forest — use predict_proba for probability
    # ------------------------------------------------------------------
    input_data = {
        "age":                       age,
        "claim_amount":              amount,
        "previous_claims":           history["previous_claim_count"],  # use DB count
        "policy_duration":           5,
        "hospital_type_Private":     1,
        "claim_type_Reimbursement":  0,
    }
    input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)

    fraud_probability = float(model.predict_proba(input_df)[0][1])
    ml_risk_level     = risk_level(fraud_probability)

    # ------------------------------------------------------------------
    # 7. Deterministic Decision Engine
    # ------------------------------------------------------------------
    engine_result = evaluate_claim(
        policy=policy_facts,
        documents=documents,
        history=history,
        fraud_probability=fraud_probability,
        amount=amount,
    )
    final_decision  = engine_result["final_decision"]
    decision_reason = engine_result["reason"]
    risk_flags      = engine_result["risk_flags"]

    # ------------------------------------------------------------------
    # 8. Gemini — explanation only, never decides
    # ------------------------------------------------------------------
    analysis = analyze_claim(
        name=name,
        policy=policy,
        disease=disease,
        hospital=hospital,
        amount=amount,
        age=age,
        claim_history=history,
        policy_facts=policy_facts,
        document_checks=documents,
        fraud_probability=fraud_probability,
        risk_level=ml_risk_level,
        risk_flags=risk_flags,
        final_decision=final_decision,
        decision_reason=decision_reason,
        policy_text_snippet=policy_snippet,
        previous_claims=previous_claims,
    )

    # ------------------------------------------------------------------
    # 9. Persist to SQLite
    # ------------------------------------------------------------------
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO claims(
            name, age, policy, gender, hospital, disease, insurance,
            previous_claims, amount,
            medical_report, hospital_bill, policy_pdf,
            prediction, fraud_probability, risk_level, final_decision,
            risk_flags, policy_verified, document_verified
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        name, age, policy, gender, hospital, disease, insurance,
        previous_claims, amount,
        medical_filename, bill_filename, policy_filename,
        ml_risk_level,                    # legacy "prediction" column
        round(fraud_probability, 4),
        ml_risk_level,
        final_decision,
        json.dumps(risk_flags),
        json.dumps({k: str(v) for k, v in policy_facts.items()}),
        json.dumps({k: str(v) for k, v in documents.items()}),
    ))
    conn.commit()
    conn.close()

    # ------------------------------------------------------------------
    # 10. Render result page
    # ------------------------------------------------------------------
    return render_template(
        "result.html",
        # Claim details
        name=name,
        policy=policy,
        hospital=hospital,
        disease=disease,
        amount=amount,
        age=age,
        gender=gender,
        insurance=insurance,
        # Files
        medical=medical_filename,
        bill=bill_filename,
        policy_file=policy_filename,
        # History
        history=history,
        # Documents
        documents=documents,
        # Policy
        policy_facts=policy_facts,
        policy_snippet=policy_snippet,
        # ML
        fraud_probability=fraud_probability,
        fraud_pct=f"{fraud_probability:.1%}",
        risk_level=ml_risk_level,
        # Decision
        final_decision=final_decision,
        decision_reason=decision_reason,
        risk_flags=risk_flags,
        # Gemini explanation
        analysis=analysis,
    )


@app.route("/claims")
def claims():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims")
    data = cursor.fetchall()
    conn.close()
    return render_template("view_claims.html", claims=data)


if __name__ == "__main__":
    app.run(debug=True)
