"""
agent.py — DEPRECATED

The old decision logic (`if "covered" in policy → APPROVED`) has been removed
because a policy text containing the word "covered" is NOT sufficient for approval.

The deterministic Decision Engine (decision_engine.py) is now the sole authority
for APPROVED / MANUAL REVIEW / REJECTED decisions.

This file is kept as a placeholder to avoid import errors from any legacy code.
"""


def decision_agent(fraud_prediction, policy_text):
    """
    DEPRECATED.  Do not call this function.

    Returns a placeholder so that any accidental import doesn't crash the app.
    The real decision is made by decision_engine.evaluate_claim().
    """
    raise RuntimeError(
        "decision_agent() is deprecated. "
        "Use decision_engine.evaluate_claim() instead."
    )