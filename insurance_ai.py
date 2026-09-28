"""
insurance_ai.py
---------------
Gemini is used ONLY for natural-language explanation.
The final decision is supplied by the deterministic Decision Engine.
Gemini MUST NOT change, override, or contradict the decision.
"""

import os
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()

client = Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _fmt_bool(value):
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Not specified in policy"


def _fmt_flags(flags):
    if not flags:
        return "None"
    return "\n".join(f"  • {f}" for f in flags)


def analyze_claim(
    *,
    name,
    policy,
    disease,
    hospital,
    amount,
    age,
    # New structured inputs
    claim_history,
    policy_facts,
    document_checks,
    fraud_probability,
    risk_level,
    risk_flags,
    final_decision,
    decision_reason,
    policy_text_snippet="",
    # Legacy fallback (ignored for decision, kept for context)
    previous_claims=0,
):
    """
    Call Gemini to generate a natural-language explanation.

    The final_decision has already been determined by the Decision Engine.
    Gemini must only explain — never override.
    """

    flags_text = _fmt_flags(risk_flags)

    prompt = f"""
You are an Insurance Claim Analysis Assistant providing a detailed explanation report.

CRITICAL INSTRUCTION: The final decision shown below has already been determined by
a deterministic Decision Engine. You MUST NOT change, override, or contradict it.
Your sole purpose is to explain the decision using only the evidence provided.

============================================================
CLAIM DETAILS
============================================================
Name            : {name}
Policy Number   : {policy}
Hospital        : {hospital}
Disease/Procedure: {disease}
Claim Amount    : ₹{amount}
Age             : {age}

============================================================
CLAIM HISTORY (from database — authoritative)
============================================================
Actual Previous Claims   : {claim_history.get('previous_claim_count', 'N/A')}
Entered by User          : {claim_history.get('entered_previous_claims', 'N/A')}
History Mismatch         : {claim_history.get('claim_history_mismatch', False)}
Claims (last 30 days)    : {claim_history.get('claims_last_30_days', 0)}
Claims (last 90 days)    : {claim_history.get('claims_last_90_days', 0)}
Claims (last 1 year)     : {claim_history.get('claims_last_1_year', 0)}
Same Disease Claimed     : {claim_history.get('same_disease_claim_count', 0)} time(s)
Same Hospital Claimed    : {claim_history.get('same_hospital_claim_count', 0)} time(s)
Similar Claims           : {claim_history.get('duplicate_or_similar_claim_count', 0)}
Avg Previous Amount      : ₹{claim_history.get('average_previous_claim_amount', 0)}
High Frequency Flag      : {claim_history.get('high_claim_frequency', False)}

============================================================
POLICY VERIFICATION (facts extracted from PDF only)
============================================================
Policy Active            : {_fmt_bool(policy_facts.get('policy_active'))}
Disease Covered          : {_fmt_bool(policy_facts.get('disease_covered'))}
Hospital In Network      : {_fmt_bool(policy_facts.get('hospital_covered'))}
Waiting Period Satisfied : {_fmt_bool(policy_facts.get('waiting_period_satisfied'))}
Coverage Limit           : {policy_facts.get('coverage_limit') or 'Not specified in policy'}
Exclusions Note          : {policy_facts.get('exclusions_note') or 'None noted'}

Relevant Policy Text:
{policy_text_snippet[:800] if policy_text_snippet else 'Not available'}

============================================================
DOCUMENT VERIFICATION
============================================================
Medical Report Present   : {document_checks.get('medical_report_present', False)}
Hospital Bill Present    : {document_checks.get('hospital_bill_present', False)}
Patient Name Match       : {_fmt_bool(document_checks.get('patient_match'))}
Hospital Match           : {_fmt_bool(document_checks.get('hospital_match'))}
Disease/Diagnosis Match  : {_fmt_bool(document_checks.get('disease_match'))}
Claim Amount Match       : {_fmt_bool(document_checks.get('amount_match'))}
Document Mismatch        : {document_checks.get('document_mismatch', False)}
Verification Incomplete  : {document_checks.get('verification_incomplete', False)}

============================================================
FRAUD RISK ANALYSIS (ML Model)
============================================================
Fraud Probability        : {fraud_probability:.1%}
Risk Level               : {risk_level}

============================================================
RISK FLAGS
============================================================
{flags_text}

============================================================
FINAL DECISION  (set by Decision Engine — DO NOT CHANGE)
============================================================
Decision : {final_decision}
Reason   : {decision_reason}

============================================================
YOUR TASK
============================================================
Write a professional insurance claim analysis report with these sections.
Use only the evidence supplied above. Do not invent facts.
Do not change the Final Decision.

1. CLAIM SUMMARY
   Brief overview of the claim.

2. POLICY COVERAGE
   What the policy explicitly covers or does not cover for this claim.
   If the policy is silent on a point, say "Not specified in policy."

3. HOSPITAL VERIFICATION
   Whether the hospital is in network, or if this could not be determined from the policy.

4. DOCUMENT VERIFICATION
   Whether documents are consistent with the claim. Note any mismatches.

5. CLAIM HISTORY ANALYSIS
   Describe the claimant's claim history. Note any frequency concerns.
   IMPORTANT: High claim count is a risk signal only — not automatic fraud.

6. FRAUD RISK ANALYSIS
   Explain the ML fraud probability and what it means in context.

7. RISK FLAGS
   Explain each flag that was raised and why.

8. FINAL DECISION: {final_decision}
   Confirm and explain the decision using the supplied evidence.

9. REASON
   A concise, plain-English explanation of why this decision was reached.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text