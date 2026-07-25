import os
import re
import logging
from pypdf import PdfReader
from bs4 import BeautifulSoup
from src.ingestion.schemas import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentParserRouter:
    """Extracts text from PDF, HTML, Markdown, and TXT files with unicode sanitization."""

    @staticmethod
    def sanitize_unicode_string(raw_text: str) -> str:
        """
        Clean text string to prevent Pydantic unicode validation errors and vector DB crashes.
        - Strips unpaired UTF-16 surrogates (\ud800-\udfff)
        - Removes null bytes (\x00)
        - Replaces corrupted PDF bullet characters
        - Fixes glued words from PDF extraction (e.g., 'laiddownby' -> 'laiddown by')
        """
        if not raw_text:
            return ""

        # 1. Remove unpaired surrogates which cause Pydantic Rust validator errors
        clean_chars = [char for char in raw_text if not ("\ud800" <= char <= "\udfff")]
        text = "".join(clean_chars)

        # 2. Drop non-UTF-8 compliant bytes and null characters
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        text = text.replace("\x00", "")

        # 3. Replace corrupted PDF bullet points with standard dashes
        text = text.replace("", "\n - ")

        # 4. Insert space before glued transition words resulting from PDF line joins
        pattern = r"([a-z])(based|by|down|under|from|to|for|rules|procedures)\b"
        for _ in range(2):
            text = re.sub(pattern, r"\1 \2", text)

        return text

    def _parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF pages."""
        try:
            reader = PdfReader(file_path)
            pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return " \n ".join(pages_text)
        except Exception as err:
            raise ValueError(f"Failed to extract text from PDF file '{file_path}': {err}")

    def _parse_html(self, file_path: str) -> str:
        """Extract text from HTML files using BeautifulSoup."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator=" \n ")
        except Exception as err:
            raise ValueError(f"Failed to parse HTML file '{file_path}': {err}")

    def _parse_txt(self, file_path: str) -> str:
        """Read text from flat text files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as err:
            raise ValueError(f"Failed to read file '{file_path}': {err}")

    def process_file(self, file_path: str) -> Document:
        """Parse file content based on extension and return a sanitized Document object."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found on disk: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if not ext:
            ext = "txt"

        if ext == "pdf":
            raw_text = self._parse_pdf(file_path)
        elif ext in ["html", "htm"]:
            raw_text = self._parse_html(file_path)
        elif ext in ["txt", "md"]:
            raw_text = self._parse_txt(file_path)
        else:
            logger.warning(f"Unknown file extension '.{ext}'. Falling back to plain text parser.")
            raw_text = self._parse_txt(file_path)

        sanitized_content = self.sanitize_unicode_string(raw_text)

        return Document(
            page_content=sanitized_content,
            metadata=DocumentMetadata(source_path=file_path, file_type=ext),
        )
