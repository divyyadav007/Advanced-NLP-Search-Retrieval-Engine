import re
from typing import List
from src.config import config
from src.ingestion.schemas import Document, Chunk, ChunkMetadata


class ChunkingEngine:
    """Provides methods for splitting documents into smaller text chunks for indexing."""

    @staticmethod
    def fixed_size_chunk(
        document: Document, chunk_size: int = None, chunk_overlap: int = None
    ) -> List[Chunk]:
        """Splits document text into fixed character windows with sliding overlap."""
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = config.CHUNK_OVERLAP

        text = document.page_content
        chunks = []
        start = 0
        chunk_index = 0

        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size  # Guard against negative or zero stride

        while start < len(text):
            end = start + chunk_size
            slice_text = text[start:end]

            metadata = ChunkMetadata(
                source_path=document.metadata.source_path,
                file_type=document.metadata.file_type,
                chunk_index=chunk_index,
                parent_document_id=document.id,
            )

            chunk_id = f"{document.id}-chunk-{chunk_index}"
            chunks.append(Chunk(id=chunk_id, page_content=slice_text, metadata=metadata))

            start += step
            chunk_index += 1

        return chunks

    @staticmethod
    def structure_aware_markdown_chunk(document: Document) -> List[Chunk]:
        """Splits Markdown documents at header boundaries (# H1, ## H2, ### H3)."""
        text = document.page_content
        sections = re.split(r"(^#+\s+.*$)", text, flags=re.MULTILINE)

        chunks = []
        chunk_index = 0
        current_header = "Introduction"

        for section in sections:
            if not section.strip():
                continue

            if section.startswith("#"):
                current_header = section.strip()
                continue

            contextualized_content = f"{current_header}\n\n{section.strip()}"

            metadata = ChunkMetadata(
                source_path=document.metadata.source_path,
                file_type=document.metadata.file_type,
                chunk_index=chunk_index,
                parent_document_id=document.id,
                custom_attributes={"markdown_header": current_header},
            )

            chunk_id = f"{document.id}-markdown-{chunk_index}"
            chunks.append(
                Chunk(id=chunk_id, page_content=contextualized_content, metadata=metadata)
            )
            chunk_index += 1

        return chunks
