import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================
# Read PDF
# ==========================

def load_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


# ==========================
# Split into Chunks
# ==========================

def create_chunks(text, chunk_size=300):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# ==========================
# Build FAISS Vector Store
# ==========================

def build_vector_store(pdf_path):

    text = load_pdf(pdf_path)

    chunks = create_chunks(text)

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

    return index, chunks


# ==========================
# Search Policy
# ==========================

def search_policy(question, index, chunks):

    query_embedding = model.encode([question])

    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, k=3)

    answer = ""

    for idx in I[0]:
        answer += chunks[idx]
        answer += "\n\n"

    return answer


# ==========================
# Policy Fact Extraction
# ==========================

import re as _re


def _find_keywords(text, keywords):
    """Return True if any keyword appears in text (case-insensitive), else None."""
    low = text.lower()
    for kw in keywords:
        if kw.lower() in low:
            return True
    return None


def _explicitly_excluded(text, term):
    """Return True if the text explicitly excludes the term."""
    patterns = [
        rf"(?:not\s+covered|excluded?|exclusion)[^.]*{_re.escape(term)}",
        rf"{_re.escape(term)}[^.]*(?:not\s+covered|excluded?|exclusion)",
    ]
    low = text.lower()
    for pat in patterns:
        if _re.search(pat, low):
            return True
    return False


def extract_policy_facts(text, disease="", hospital=""):
    """
    Extract explicit facts from policy text.

    Returns a dict where:
      - True  = policy explicitly confirms
      - False = policy explicitly denies / excludes
      - None  = policy does not specify (do NOT guess)
    """
    if not text:
        return {
            "policy_active": None,
            "disease_covered": None,
            "hospital_covered": None,
            "waiting_period_satisfied": None,
            "coverage_limit": None,
            "exclusions_note": None,
        }

    low = text.lower()
    d_term = disease.lower().strip()
    h_term = hospital.lower().strip()

    # --- Policy active ---
    if _re.search(r"\bpolicy\s+(?:is\s+)?(?:active|valid|in\s+force)\b", low):
        policy_active = True
    elif _re.search(r"\bpolicy\s+(?:is\s+)?(?:inactive|lapsed|expired|cancelled)\b", low):
        policy_active = False
    else:
        policy_active = None

    # --- Disease covered ---
    if d_term:
        if _explicitly_excluded(low, d_term):
            disease_covered = False
        elif d_term in low and _re.search(
            r"\b(?:covered|coverage|eligible|included)\b", low
        ):
            disease_covered = True
        else:
            disease_covered = None
    else:
        disease_covered = None

    # --- Hospital covered ---
    if h_term:
        if _explicitly_excluded(low, h_term):
            hospital_covered = False
        elif h_term in low and _re.search(
            r"\b(?:network|empanelled|covered|approved)\b", low
        ):
            hospital_covered = True
        else:
            hospital_covered = None
    else:
        hospital_covered = None

    # --- Waiting period ---
    wp_match = _re.search(
        r"waiting\s+period[^.]*?(\d+)\s*(day|month|year)", low
    )
    if wp_match:
        # We found a waiting period clause but cannot determine from text alone
        # whether it has been satisfied without policy start date — return None
        waiting_period_satisfied = None
    elif _re.search(r"\bno\s+waiting\s+period\b", low):
        waiting_period_satisfied = True
    else:
        waiting_period_satisfied = None

    # --- Coverage limit (return raw text, not an invented number) ---
    limit_match = _re.search(
        r"(?:maximum|max\.?|sum\s+insured|coverage\s+limit)[^.\n]*?(?:₹|rs\.?|inr)?\s*([\d,]+)",
        low,
    )
    if limit_match:
        coverage_limit = limit_match.group(1).replace(",", "")
    else:
        coverage_limit = None  # "Not specified in policy"

    # --- Exclusions note ---
    excl_match = _re.search(r"(?:exclusion|not\s+covered)[^\n]{0,200}", low)
    exclusions_note = excl_match.group(0).strip() if excl_match else None

    return {
        "policy_active": policy_active,
        "disease_covered": disease_covered,
        "hospital_covered": hospital_covered,
        "waiting_period_satisfied": waiting_period_satisfied,
        "coverage_limit": coverage_limit,
        "exclusions_note": exclusions_note,
    }