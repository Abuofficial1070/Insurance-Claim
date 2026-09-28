import re

from rag import load_pdf, extract_policy_facts


def verify_policy(pdf_path, disease, hospital):
    try:
        text = load_pdf(pdf_path)
    except (OSError, RuntimeError):
        text = ""
    facts = extract_policy_facts(text, disease=disease, hospital=hospital)
    return facts
