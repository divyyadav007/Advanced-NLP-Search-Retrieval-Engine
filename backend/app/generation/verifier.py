import re
from typing import List, Dict, Any


class CitationVerifier:
    """Verifies bracketed citations in LLM responses against provided source chunks."""

    @staticmethod
    def verify_citations(llm_answer: str, top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scan bracketed citation references (e.g. [1], [2]) in the LLM answer and verify
        that they correspond to valid, non-empty context chunk indices.
        """
        citation_pattern = re.compile(r"\[(\d+)\]")
        matches = citation_pattern.findall(llm_answer)

        extracted_indices = sorted(list(set(int(m) for m in matches)))
        flagged_citations = {}
        is_valid = True

        for index in extracted_indices:
            array_slot = index - 1

            # Check if cited index is out of bounds
            if array_slot < 0 or array_slot >= len(top_chunks):
                flagged_citations[f"[{index}]"] = (
                    "MALFORMED_INDEX: Context block index does not exist."
                )
                is_valid = False
                continue

            chunk_text = top_chunks[array_slot]["chunk"].page_content.strip()
            if not chunk_text:
                flagged_citations[f"[{index}]"] = (
                    "EMPTY_CONTEXT: Target chunk contains no text content."
                )
                is_valid = False

        validated_indices = [
            idx for idx in extracted_indices if f"[{idx}]" not in flagged_citations
        ]

        return {
            "is_valid": is_valid,
            "flagged_issues": flagged_citations,
            "validated_indices": validated_indices,
        }
