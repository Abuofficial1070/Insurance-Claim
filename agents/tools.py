import joblib
import pandas as pd

from rag import build_vector_store, search_policy

model = joblib.load("fraud_model.pkl")
features = joblib.load("features.pkl")


def fraud_prediction_tool(age, amount, previous_claims):

    input_data = {

        "age": age,
        "claim_amount": amount,
        "previous_claims": previous_claims,
        "policy_duration": 5,
        "hospital_type_Private": 1,
        "claim_type_Reimbursement": 0

    }

    df = pd.DataFrame([input_data])

    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)[0]

    if prediction == 1:
        return "HIGH FRAUD RISK"

    return "LOW FRAUD RISK"


def policy_search_tool(pdf_path, disease):

    index, chunks = build_vector_store(pdf_path)

    return search_policy(
        disease,
        index,
        chunks
    )