import re
from pathlib import Path

import fitz


def _normalise(value):
    return " ".join(str(value or "").lower().split())


def _read_document(path):
    if not path or not Path(path).exists():
        return ""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        try:
            with fitz.open(path) as document:
                return "\n".join(page.get_text() for page in document)
        except (OSError, RuntimeError):
            return ""
    if suffix in {".txt", ".csv"}:
        try:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
    return ""


def _label_value(text, labels):
    pattern = r"(?:" + "|".join(labels) + r")\s*[:\-]\s*([^\n|]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _amount(text):
    values = re.findall(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)", text, re.IGNORECASE)
    return float(values[-1].replace(",", "")) if values else None


def _matches(expected, actual):
    if not actual:
        return None
    return _normalise(expected) in _normalise(actual) or _normalise(actual) in _normalise(expected)


def verify_documents(medical_path, bill_path, name, hospital, disease, amount):
    medical_text = _read_document(medical_path)
    bill_text = _read_document(bill_path)
    medical_name = _label_value(medical_text, ["patient", "patient name", "name"])
    medical_disease = _label_value(medical_text, ["diagnosis", "disease", "procedure"])
    bill_name = _label_value(bill_text, ["patient", "patient name", "name"])
    bill_hospital = _label_value(bill_text, ["hospital", "hospital name"])
    bill_amount = _amount(bill_text)

    checks = {
        "medical_report_present": bool(medical_text),
        "hospital_bill_present": bool(bill_text),
        "patient_match": _matches(name, medical_name or bill_name),
        "hospital_match": _matches(hospital, bill_hospital),
        "amount_match": None if bill_amount is None else abs(bill_amount - amount) <= max(amount * 0.01, 1),
        "disease_match": _matches(disease, medical_disease),
    }
    mismatch = any(value is False for value in checks.values())
    unavailable = any(value is None for key, value in checks.items() if key not in {"medical_report_present", "hospital_bill_present"})
    verified = checks["medical_report_present"] and checks["hospital_bill_present"] and not mismatch and not unavailable
    return {**checks, "document_mismatch": mismatch, "verification_incomplete": unavailable, "document_verified": verified}
