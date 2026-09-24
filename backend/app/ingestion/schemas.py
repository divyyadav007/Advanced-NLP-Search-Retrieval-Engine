import uuid
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata container for parsed source documents."""

    source_path: str = Field(..., description="Path or identifier of the source file.")
    file_type: str = Field(..., description="File extension format (e.g. pdf, md, html, txt).")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Ingestion timestamp."
    )
    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Extensible metadata attributes."
    )


class Document(BaseModel):
    """Container representing a parsed document before chunking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document ID.")
    page_content: str = Field(..., min_length=1, description="Extracted plaintext content.")
    metadata: DocumentMetadata = Field(..., description="Document metadata.")


class ChunkMetadata(DocumentMetadata):
    """Metadata tracking parameters for document chunks."""

    chunk_index: int = Field(
        ..., ge=0, description="Sequential index of chunk inside parent document."
    )
    parent_document_id: str = Field(..., description="Parent document identifier.")


class Chunk(BaseModel):
    """Text slice chunk object used for sparse and dense indexing."""

    id: str = Field(..., description="Unique chunk identifier.")
    page_content: str = Field(..., min_length=1, description="Chunk text content.")
    metadata: ChunkMetadata = Field(..., description="Chunk lineage metadata.")
