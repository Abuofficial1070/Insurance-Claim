RISK_THRESHOLDS = {
    "low_max": 0.20,
    "medium_max": 0.50,
    "high_max": 0.75,
}


def risk_level(probability):
    if probability < RISK_THRESHOLDS["low_max"]:
        return "LOW RISK"
    if probability < RISK_THRESHOLDS["medium_max"]:
        return "MEDIUM RISK"
    if probability < RISK_THRESHOLDS["high_max"]:
        return "HIGH RISK"
    return "CRITICAL RISK"


def evaluate_claim(policy, documents, history, fraud_probability, amount):
    flags = []
    if history["claim_history_mismatch"]:
        flags.append("CLAIM HISTORY MISMATCH")
    if history["high_claim_frequency"]:
        flags.append("HIGH CLAIM FREQUENCY")
    if history["duplicate_or_similar_claim_count"]:
        flags.append("MULTIPLE SIMILAR CLAIMS")
    if history["same_disease_claim_count"] >= 2:
        flags.append("SAME DISEASE CLAIMED MULTIPLE TIMES")
    if history["same_hospital_claim_count"] >= 3:
        flags.append("SAME HOSPITAL CLAIMED MULTIPLE TIMES")
    if amount >= 100000:
        flags.append("HIGH CLAIM AMOUNT")
    if documents["document_mismatch"] or documents["verification_incomplete"]:
        flags.append("DOCUMENT MISMATCH")
    if policy.get("hospital_covered") is False:
        flags.append("HOSPITAL NOT IN NETWORK")
    if policy.get("policy_active") is False:
        flags.append("POLICY INACTIVE")
    if policy.get("waiting_period_satisfied") is False:
        flags.append("WAITING PERIOD NOT SATISFIED")
    if fraud_probability >= RISK_THRESHOLDS["high_max"]:
        flags.append("HIGH ML FRAUD PROBABILITY")

    fatal_policy = any(policy.get(key) is False for key in ("policy_active", "disease_covered", "hospital_covered", "waiting_period_satisfied"))
    if fatal_policy:
        decision = "REJECTED"
        reason = "The policy verification found an explicit eligibility failure."
    elif documents["document_mismatch"] or documents["verification_incomplete"]:
        decision = "MANUAL REVIEW"
        reason = "Documents contain a mismatch or could not be fully verified."
    elif history["claim_history_mismatch"] or history["high_claim_frequency"] or history["duplicate_or_similar_claim_count"]:
        decision = "MANUAL REVIEW"
        reason = "Claim history contains a review signal; previous claims are not treated as proof of fraud."
    elif fraud_probability >= RISK_THRESHOLDS["high_max"]:
        decision = "MANUAL REVIEW"
        reason = "The machine-learning fraud probability is in the critical range."
    elif any(policy.get(key) is None for key in ("policy_active", "disease_covered", "hospital_covered", "waiting_period_satisfied")):
        decision = "MANUAL REVIEW"
        reason = "The policy PDF does not provide enough explicit evidence for automatic approval."
    else:
        decision = "APPROVED"
        reason = "Policy eligibility, documents, claim history, and fraud risk passed the configured checks."

    return {
        "final_decision": decision,
        "reason": reason,
        "risk_level": risk_level(fraud_probability),
        "fraud_probability": fraud_probability,
        "risk_flags": flags,
    }
