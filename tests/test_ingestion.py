import pytest
import numpy as np
from src.ingestion.schemas import Document, DocumentMetadata
from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.chunkers import ChunkingEngine
from src.ingestion.deduplicator import ChunkDeduplicator


def test_sanitize_unicode_string():
    """Test unicode sanitization, surrogate stripping, and layout normalization."""
    # Input text with unpaired surrogates, corrupted bullets, and glued words
    raw_text = "Hello\x00world.  Policy document laiddownby administration \ud800."
    sanitized = DocumentParserRouter.sanitize_unicode_string(raw_text)

    # Assert null byte removed
    assert "\x00" not in sanitized
    # Assert unpaired surrogate removed
    assert "\ud800" not in sanitized
    # Assert bullet replaced
    assert "\n - " in sanitized
    # Assert word gluing fixed
    assert "laid down by" in sanitized


def test_fixed_size_chunk():
    """Test sliding window character chunking."""
    doc = Document(
        page_content="A" * 1000, metadata=DocumentMetadata(source_path="test.txt", file_type="txt")
    )
    chunks = ChunkingEngine.fixed_size_chunk(doc, chunk_size=500, chunk_overlap=100)

    # 1000 chars with size 500 and overlap 100 should create 3 chunks:
    # Chunk 0: 0-500
    # Chunk 1: 400-900
    # Chunk 2: 800-1000
    assert len(chunks) == 3
    assert chunks[0].id == f"{doc.id}-chunk-0"
    assert len(chunks[0].page_content) == 500
    assert len(chunks[2].page_content) == 200


def test_deduplicator_cosine_similarity():
    """Test angular distance math between embeddings."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    vec_c = np.array([0.0, 1.0, 0.0])

    # Same vector similarity should be 1.0
    assert pytest.approx(ChunkDeduplicator.calculate_cosine_similarity(vec_a, vec_b)) == 1.0
    # Orthogonal vectors similarity should be 0.0
    assert pytest.approx(ChunkDeduplicator.calculate_cosine_similarity(vec_a, vec_c)) == 0.0
