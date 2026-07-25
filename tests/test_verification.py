import pytest
from src.ingestion.schemas import Chunk, ChunkMetadata
from src.generation.verifier import CitationVerifier


def test_citation_verifier_valid():
    """Test citation verifier with correct, valid bracketed citations."""
    meta = ChunkMetadata(
        source_path="policy.txt", file_type="txt", chunk_index=0, parent_document_id="doc1"
    )
    top_chunks = [
        {
            "chunk": Chunk(
                id="chunk1", page_content="The policy requires 75% attendance.", metadata=meta
            )
        }
    ]

    # Text with valid citation pointing to context block 1
    llm_answer = "Students must maintain 75% attendance [1]."
    report = CitationVerifier.verify_citations(llm_answer, top_chunks)

    assert report["is_valid"] is True
    assert "[1]" not in report["flagged_issues"]


def test_citation_verifier_invalid_index():
    """Test citation verifier when the citation points to a non-existent index."""
    meta = ChunkMetadata(
        source_path="policy.txt", file_type="txt", chunk_index=0, parent_document_id="doc1"
    )
    top_chunks = [
        {
            "chunk": Chunk(
                id="chunk1", page_content="The policy requires 75% attendance.", metadata=meta
            )
        }
    ]

    # Text with invalid citation pointing to [2] (only 1 context block was supplied)
    llm_answer = "Students must maintain 75% attendance [2]."
    report = CitationVerifier.verify_citations(llm_answer, top_chunks)

    assert report["is_valid"] is False
    assert "[2]" in report["flagged_issues"]
    assert "MALFORMED_INDEX" in report["flagged_issues"]["[2]"]
