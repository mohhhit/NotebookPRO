import PyPDF2
import pdfplumber
from docx import Document
from pathlib import Path
from typing import List, Dict
import re
import warnings
import logging

# Suppress PyPDF2 warnings about font descriptors
warnings.filterwarnings('ignore', category=UserWarning, module='PyPDF2')
logging.getLogger('PyPDF2').setLevel(logging.ERROR)


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
        
        if suffix == '.pdf':
            content = self._extract_pdf(file_path)
        elif suffix == '.txt':
            content = self._extract_txt(file_path)
        elif suffix == '.docx':
            content = self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return {
            'filename': file_path.name,
            'path': str(file_path),
            'content': content,
            'format': suffix
        }
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using pdfplumber with PyPDF2 fallback."""
        text = ""
        try:
            # Primary: Use pdfplumber (better for complex PDFs)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            # Fallback: Use PyPDF2 with warnings suppressed
            try:
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
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50, semantic: bool = True) -> List[str]:
        """
        Split text into chunks using semantic or simple chunking.
        
        Args:
            text: The text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            semantic: Use semantic chunking (by headers/concepts) if True
            
        Returns:
            List of text chunks
        """
        if semantic:
            return self._semantic_chunk(text, chunk_size, overlap)
        else:
            return self._simple_chunk(text, chunk_size, overlap)
    
    def _semantic_chunk(self, text: str, target_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Chunk text by detecting headers and logical sections.
        Perfect for lecture slides and structured documents.
        """
        chunks = []
        
        # Split by common header patterns
        # Pattern 1: Lines that are ALL CAPS or Title Case followed by newline
        # Pattern 2: Lines starting with numbers like "1.", "1.1", etc.
        # Pattern 3: Lines with clear visual separators
        
        # First, split by double newlines (paragraphs)
        sections = text.split('\n\n')
        
        current_chunk = ""
        current_header = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this looks like a header
            is_header = self._is_likely_header(section)
            
            if is_header and len(current_chunk) > 100:
                # Save previous chunk and start new one with this header
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section + "\n\n"
                current_header = section
            else:
                # Add to current chunk
                potential_chunk = current_chunk + section + "\n\n"
                
                # If chunk is getting too large, split it
                if len(potential_chunk) > target_size * 1.5:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section + "\n\n"
                else:
                    current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If semantic chunking produced too few chunks, fall back to simple chunking
        if len(chunks) < len(text) / (target_size * 2):
            return self._simple_chunk(text, target_size, overlap)
        
        return chunks
    
    def _is_likely_header(self, text: str) -> bool:
        """Detect if text is likely a header/title."""
        # Too long to be a header
        if len(text) > 200:
            return False
        
        # Single line headers
        if '\n' not in text:
            # ALL CAPS
            if text.isupper() and len(text.split()) <= 10:
                return True
            
            # Title Case
            if text.istitle() and len(text.split()) <= 10:
                return True
            
            # Numbered sections like "1.", "1.1", "Chapter 1"
            if re.match(r'^(\d+\.)+\s+', text) or re.match(r'^(Chapter|Section|Part)\s+\d+', text, re.IGNORECASE):
                return True
        
        return False
    
    def _simple_chunk(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
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
