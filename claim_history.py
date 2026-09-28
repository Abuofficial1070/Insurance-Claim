import sqlite3
from datetime import datetime


def _normalise(value):
    return " ".join(str(value or "").lower().split())


def get_claim_history(db_path, name, policy, hospital, disease, amount, entered_count):
    connection = sqlite3.connect(db_path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(claims)")}
    date_column = "submitted_at" if "submitted_at" in columns else None

    rows = connection.execute(
        """
        SELECT hospital, disease, amount, """ + ("submitted_at" if date_column else "NULL") + """
        FROM claims
        WHERE lower(name) = lower(?) AND policy = ?
        """,
        (name, policy),
    ).fetchall()
    connection.close()

    normal_hospital = _normalise(hospital)
    normal_disease = _normalise(disease)
    amounts = [float(row[2] or 0) for row in rows]
    same_hospital = sum(_normalise(row[0]) == normal_hospital for row in rows)
    same_disease = sum(_normalise(row[1]) == normal_disease for row in rows)
    now = datetime.utcnow()
    recent_30 = 0
    recent_90 = 0
    recent_year = 0
    for row in rows:
        if not row[3]:
            continue
        try:
            age_days = (now - datetime.fromisoformat(row[3])).days
        except ValueError:
            continue
        recent_30 += age_days <= 30
        recent_90 += age_days <= 90
        recent_year += age_days <= 365

    average_amount = sum(amounts) / len(amounts) if amounts else 0
    similar = sum(
        _normalise(row[0]) == normal_hospital
        and _normalise(row[1]) == normal_disease
        and abs(float(row[2] or 0) - amount) <= max(amount * 0.1, 1)
        for row in rows
    )
    actual_count = len(rows)
    return {
        "previous_claim_count": actual_count,
        "claims_last_30_days": recent_30,
        "claims_last_90_days": recent_90,
        "claims_last_1_year": recent_year,
        "average_previous_claim_amount": round(average_amount, 2),
        "total_previous_claim_amount": round(sum(amounts), 2),
        "same_disease_claim_count": same_disease,
        "same_hospital_claim_count": same_hospital,
        "days_since_last_claim": None,
        "current_claim_amount_vs_average": round(amount / average_amount, 2) if average_amount else None,
        "duplicate_or_similar_claim_count": similar,
        "entered_previous_claims": entered_count,
        "claim_history_mismatch": int(entered_count) != actual_count,
        "high_claim_frequency": actual_count >= 5 or recent_90 >= 3,
    }
