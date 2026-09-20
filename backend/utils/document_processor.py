import PyPDF2
import pdfplumber
from docx import Document
from pathlib import Path
from typing import List, Dict, Optional
import re
import warnings
import logging
import os

import config

# Suppress PyPDF2 warnings about font descriptors
warnings.filterwarnings('ignore', category=UserWarning, module='PyPDF2')
logging.getLogger('PyPDF2').setLevel(logging.ERROR)


def _is_valid_year(year: str) -> bool:
    """Return True for valid 4-digit publication years."""
    return bool(re.fullmatch(r'(?:19|20)\d{2}', (year or '').strip()))


def extract_publication_year(front_matter_text: str) -> str:
    """Extract publication year from front matter using an LLM, with safe fallback."""
    text = (front_matter_text or '').strip()
    if not text:
        return "0000"

    prompt = (
        "You are a strict metadata extraction pipeline for an academic database. \n"
        "Your only task is to extract the primary publication year from the provided front-matter text of an academic textbook or paper.\n"
        "Instructions:\n"
        "1. Scan the text for copyright dates (©), publication dates, or edition release years.\n"
        "2. If multiple years are present (e.g., previous editions and a current edition), extract the most recent year.\n"
        "3. If no valid year can be found, output '0000'.\n"
        "4. You must output ONLY the 4-digit year. Do not include markdown, JSON formatting, or conversational text.\n"
        "\n"
        "Text to analyze:\n"
        f"{text}"
    )

    # Fast deterministic regex fallback if no API key/client is available.
    def _regex_fallback() -> str:
        candidates = re.findall(r'(?:19|20)\d{2}', text)
        if not candidates:
            return "0000"
        return str(max(int(c) for c in candidates))

    try:
        api_key = os.getenv("OPENAI_API_KEY", "") or getattr(config, "OPENAI_API_KEY", "")
        if not api_key:
            return _regex_fallback()

        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return _regex_fallback()

        model = os.getenv("OPENAI_METADATA_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
        )

        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r'(?:19|20)\d{2}|0000', raw)
        year = match.group(0) if match else "0000"
        return year if _is_valid_year(year) or year == "0000" else "0000"
    except Exception:
        return _regex_fallback()


class DocumentProcessor:
    """Process various document types and extract text content."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def process_file(self, file_path: Path) -> Dict[str, any]:
        """
        Process a single file and extract its content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata and content
        """
        suffix = file_path.suffix.lower()
        publication_year = "0000"
        virtual_filename = file_path.name
        
        if suffix == '.pdf':
            content = self._extract_pdf(file_path)

            # Front matter extraction for year detection.
            front_matter = content[:2000]
            detected_year = extract_publication_year(front_matter)
            publication_year = detected_year if _is_valid_year(detected_year) else "0000"
            virtual_filename = f"[{publication_year}] {file_path.name}"
        elif suffix == '.txt':
            content = self._extract_txt(file_path)
        elif suffix == '.docx':
            content = self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return {
            'filename': file_path.name,
            'virtual_filename': virtual_filename,
            'publication_year': publication_year,
            'path': str(file_path),
            'content': content,
            'format': suffix
        }
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF2 with pdfplumber fallback."""
        text = ""

        # Fast-fail with a clear reason for corrupted/partial uploads.
        try:
            if file_path.stat().st_size == 0:
                raise ValueError(f"PDF is empty (0 bytes): {file_path.name}")
        except OSError:
            pass

        try:
            # Primary: Use PyPDF2 (much faster, lower memory footprint)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception:
                            continue  # Skip problematic pages
        except Exception as e:
            # Fallback: Use pdfplumber (better for complex PDFs, but slower)
            text = "" # Reset text
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                raise ValueError(f"Could not extract text from PDF: {file_path.name}")
        
        return self._clean_text(text)
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        
        return self._clean_text(text)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize line endings first.
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Keep paragraph boundaries; collapse inner spaces/tabs only.
        text = re.sub(r'[ \t]+', ' ', text)

        # Keep common punctuation and line breaks, strip noisy symbols.
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"\n/]+', '', text)

        # Remove excessive blank lines but preserve section structure.
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 700,
        overlap: int = 120,
        semantic: bool = True,
        source_filename: Optional[str] = None,
    ) -> List[str]:
        """
        Split text into chunks using recursive character chunking.
        Prefixes each chunk with the document title to preserve global context.
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""]
            )
            raw_chunks = text_splitter.split_text(text)
        except ImportError:
            # Fallback to simple chunking if langchain is missing
            raw_chunks = self._simple_chunk(text, chunk_size, overlap)
            
        chunks = []
        prefix = f"Source Document: {source_filename}\n---\n" if source_filename else ""
        for rc in raw_chunks:
            chunks.append(prefix + rc.strip())
            
        return chunks
    
    def _simple_chunk(self, text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
        """
        Split text into overlapping chunks (original method).
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # At least 50% through the chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
