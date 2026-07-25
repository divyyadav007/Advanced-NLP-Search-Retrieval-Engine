import pytest
from src.ingestion.schemas import Chunk, ChunkMetadata
from src.indexing.hybrid_retriever import HybridRetriever


class MockSparseIndex:
    def search(self, query: str, top_k: int):
        meta = ChunkMetadata(
            source_path="s.txt", file_type="txt", chunk_index=0, parent_document_id="doc1"
        )
        return [
            {
                "chunk": Chunk(id="chunk1", page_content="lexical chunk", metadata=meta),
                "sparse_score": 10.0,
            }
        ]


class MockDenseIndex:
    def search(self, query: str, top_k: int):
        return [
            {
                "chunk_id": "chunk2",
                "text": "dense chunk",
                "metadata": {
                    "source_path": "d.txt",
                    "file_type": "txt",
                    "chunk_index": 0,
                    "parent_document_id": "doc2",
                },
                "dense_score": 0.9,
            }
        ]


def test_hybrid_retriever_fusion():
    """Test parallel search fusion and Reciprocal Rank Fusion ranking updates."""
    sparse = MockSparseIndex()
    dense = MockDenseIndex()

    retriever = HybridRetriever(sparse, dense)
    results = retriever.retrieve("test query", top_k=2)

    # Assert RRF fused results are returned
    assert len(results) == 2

    # Check ids
    chunk_ids = [res["chunk"].id for res in results]
    assert "chunk1" in chunk_ids
    assert "chunk2" in chunk_ids

    # Verify scores are correct
    assert results[0]["score"] > 0.0
