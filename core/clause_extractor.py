"""
Clause Extractor — regex-based legal clause detection.
"""

import re
from typing import List, Dict

CLAUSE_PATTERNS: Dict[str, str] = {
    "Payment Terms":         r"payment.*?(?:\.|;|\n\n)",
    "Termination":           r"terminat.*?(?:\.|;|\n\n)",
    "Confidentiality":       r"confidential.*?(?:\.|;|\n\n)",
    "Liability":             r"liability.*?(?:\.|;|\n\n)",
    "Indemnification":       r"indemnif.*?(?:\.|;|\n\n)",
    "Governing Law":         r"governing law.*?(?:\.|;|\n\n)",
    "Dispute Resolution":    r"dispute.*?(?:\.|;|\n\n)",
    "Intellectual Property": r"intellectual property.*?(?:\.|;|\n\n)",
    "Force Majeure":         r"force majeure.*?(?:\.|;|\n\n)",
    "Warranty":              r"warrant.*?(?:\.|;|\n\n)",
}


def extract_legal_clauses(text: str) -> List[Dict[str, str]]:
    """Extract legal clauses from document text using regex patterns."""
    text_lower = text.lower()
    found_clauses: List[Dict[str, str]] = []

    for clause_type, pattern in CLAUSE_PATTERNS.items():
        for match in re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            start = match.start()
            context_start = max(0, start - 200)
            context_end = min(len(text), start + 500)
            clause_text = text[context_start:context_end].strip()
            if len(clause_text) > 50:
                found_clauses.append({"type": clause_type, "text": clause_text})

    if not found_clauses:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
        for i, para in enumerate(paragraphs[:5], 1):
            found_clauses.append({"type": f"Section {i}", "text": para[:500]})

    return found_clauses
