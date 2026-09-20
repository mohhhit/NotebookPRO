"""
NotebookLM-style response generator with professional formatting.
"""

from typing import List, Dict
import config
import re


class SimpleGenerator:
    """Lightweight generator with NotebookLM-quality formatting."""
    
    def __init__(self):
        self.ready = True
    
    def _clean_and_format_text(self, text: str) -> str:
        """Clean and format text with proper spacing like NotebookLM."""
        # Fix spacing after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Add proper line breaks after sentences
        text = re.sub(r'([.!?])\s+', r'\1\n\n', text)
        return text.strip()
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms that should be bolded."""
        # Look for capitalized terms, technical terms
        terms = []
        
        # Find terms in quotes
        quoted = re.findall(r'"([^"]+)"', text)
        terms.extend(quoted)
        
        # Find repeated important words (appear 2+ times)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Add words that appear multiple times
        terms.extend([w for w, count in word_count.items() if count >= 2])
        
        return list(set(terms))
    
    def _apply_bold_formatting(self, text: str) -> str:
        """Apply bold formatting to key terms like NotebookLM."""
        key_terms = self._extract_key_terms(text)
        
        # Bold key terms
        for term in key_terms:
            if len(term) > 3:  # Skip very short terms
                text = re.sub(rf'\b({re.escape(term)})\b', r'**\1**', text, count=1)
        
        # Bold specific patterns
        # Numbers with context
        text = re.sub(r'\b(\d+)\s+(observations?|years?|months?|quarters?)', r'**\1 \2**', text)
        
        return text
    
    def _create_structured_response(self, context: str, query: str) -> str:
        """Create a NotebookLM-style structured response."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in context.split('\n\n') if len(p.strip()) > 50]
        
        # Remove duplicates
        unique_paras = []
        seen = set()
        for para in paragraphs:
            para_key = para.lower()[:150]
            if para_key not in seen:
                unique_paras.append(para)
                seen.add(para_key)
                if len(unique_paras) >= 5:
                    break
        
        if not unique_paras:
            return context[:1000]
        
        # Build NotebookLM-style response
        response = ""
        
        # Main explanation (first paragraph - cleaned and formatted)
        main_para = self._clean_and_format_text(unique_paras[0])
        main_para = self._apply_bold_formatting(main_para)
        response += main_para + "\n\n"
        
        # Add structured details if more content available
        if len(unique_paras) > 1:
            response += "### Key Points:\n\n"
            
            for i, para in enumerate(unique_paras[1:4], 1):
                # Extract first 2-3 sentences
                sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 20]
                if sentences:
                    detail = self._clean_and_format_text('. '.join(sentences[:2]) + '.')
                    detail = self._apply_bold_formatting(detail)
                    response += f"{i}. {detail}\n\n"
        
        return response.strip()
    
    def generate_response(
        self,
        prompt: str,
        context: str = "",
        use_case: str = "explanation",
        metadatas: List[Dict] = None,
        **kwargs
    ) -> str:
        """
        Generate a NotebookLM-quality response with strict citations.
        
        Args:
            prompt: User query
            context: Retrieved context from documents
            use_case: Type of response (explanation, summary, qa,notes)
            metadatas: Metadata for each context chunk (for citations)
            
        Returns:
            Professional formatted response with inline citations
        """
        if not context:
            return (
                "I don't have enough information from your uploaded documents to answer this question. "
                "Please upload relevant study materials first, or try rephrasing your question."
            )
        
        # Use specialized prompts based on use case
        if use_case == "summary":
            response = self._create_summary_with_citations(context, prompt, metadatas)
        elif use_case == "notes":
            response = self._create_notes_with_citations(context, prompt, metadatas)
        elif use_case == "qa":
            response = self._create_qa_with_citations(context, prompt, metadatas)
        else:  # Default to explanation
            response = self._create_structured_response_with_citations(context, prompt, metadatas)
        
        return response
    
    def _create_structured_response_with_citations(
        self, 
        context: str, 
        query: str,
        metadatas: List[Dict] = None
    ) -> str:
        """Create NotebookLM-style response with inline citations."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in context.split('\n\n') if len(p.strip()) > 50]
        
        # Remove duplicates
        unique_paras = []
        seen = set()
        for para in paragraphs:
            para_key = para.lower()[:150]
            if para_key not in seen:
                unique_paras.append(para)
                seen.add(para_key)
                if len(unique_paras) >= 5:
                    break
        
        if not unique_paras:
            return context[:1000]
        
        # Build response with citations
        response = ""
        
        # Main explanation (first paragraph - cleaned and formatted)
        main_para = self._clean_and_format_text(unique_paras[0])
        main_para = self._apply_bold_formatting(main_para)
        
        # Add citation to end of main paragraph
        cite_text = self._get_citation(0, metadatas) if metadatas else ""
        response += main_para + cite_text + "\n\n"
        
        # Add structured details if more content available
        if len(unique_paras) > 1:
            response += "### Key Points:\n\n"
            
            for i, para in enumerate(unique_paras[1:4], 1):
                # Extract first 2-3 sentences
                sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 20]
                if sentences:
                    detail = self._clean_and_format_text('. '.join(sentences[:2]) + '.')
                    detail = self._apply_bold_formatting(detail)
                    
                    # Add citation
                    cite_text = self._get_citation(i, metadatas) if metadatas and i < len(metadatas) else ""
                    response += f"{i}. {detail}{cite_text}\n\n"
        
        return response.strip()
    
    def _get_citation(self, index: int, metadatas: List[Dict] = None) -> str:
        """Generate inline citation from metadata."""
        if not metadatas or index >= len(metadatas):
            return ""
        
        meta = metadatas[index]
        filename = meta.get('filename', 'Unknown')
        
        # Remove file extension for cleaner citation
        clean_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
        
        return f" **[{clean_name}]**"
    
    def _create_summary_with_citations(
        self, 
        context: str, 
        query: str,
        metadatas: List[Dict] = None
    ) -> str:
        """Create a summary with citations."""
        sentences = []
        seen = set()
        for s in context.split('.'):
            s_clean = s.strip()
            if len(s_clean) > 40 and s_clean.lower() not in seen:
                sentences.append(s_clean)
                seen.add(s_clean.lower())
                if len(sentences) >= 6:
                    break
        
        if not sentences:
            return context[:800]
        
        response = "## Summary\n\n"
        for i, point in enumerate(sentences, 1):
            cite = self._get_citation(i-1, metadatas) if metadatas else ""
            response += f"{i}. {point}.{cite}\n\n"
        
        return response.strip()
    
    def _create_qa_with_citations(
        self, 
        context: str, 
        query: str,
        metadatas: List[Dict] = None
    ) -> str:
        """Answer with strict source grounding."""
        paragraphs = [p.strip() for p in context.split('\n\n') if len(p.strip()) > 50]
        
        if not paragraphs:
            sentences = [s.strip() + '.' for s in context.split('.') if len(s.strip()) > 30]
            response = ' '.join(sentences[:6])
            cite = self._get_citation(0, metadatas) if metadatas else ""
            return response + cite
        
        # Remove duplicates
        unique_paras = []
        seen = set()
        for para in paragraphs:
            para_key = para.lower()[:150]
            if para_key not in seen:
                unique_paras.append(para)
                seen.add(para_key)
                if len(unique_paras) >= 3:
                    break
        
        # Fix spacing and add citations
        response = unique_paras[0] if unique_paras else context[:800]
        response = re.sub(r'([.!?])([A-Z])', r'\1 \2', response)
        cite = self._get_citation(0, metadatas) if metadatas else ""
        response += cite
        
        # Add supporting details if available
        if len(unique_paras) > 1:
            second_para = re.sub(r'([.!?])([A-Z])', r'\1 \2', unique_paras[1])
            cite2 = self._get_citation(1, metadatas) if metadatas and len(metadatas) > 1 else ""
            response += "\n\n" + second_para + cite2
        
        return response.strip()
    
    def _create_notes_with_citations(
        self, 
        context: str, 
        query: str,
        metadatas: List[Dict] = None
    ) -> str:
        """Create study notes with source attribution."""
        sections = [s.strip() for s in context.split('\n\n') if len(s.strip()) > 40]
        
        # Remove duplicates
        unique_sections = []
        seen = set()
        for section in sections:
            section_key = section.lower()[:100]
            if section_key not in seen:
                unique_sections.append(section)
                seen.add(section_key)
                if len(unique_sections) >= 6:
                    break
        
        if not unique_sections:
            return context[:1000]
        
        response = "## Study Notes\n\n"
        
        for i, section in enumerate(unique_sections, 1):
            sentences = [s.strip() for s in section.split('.') if len(s.strip()) > 20]
            
            if sentences:
                heading = sentences[0]
                cite = self._get_citation(i-1, metadatas) if metadatas else ""
                response += f"### {i}. {heading}{cite}\n\n"
                
                for sent in sentences[1:3]:
                    response += f"- {sent}\n"
                response += "\n"
        
        return response.strip()
    
    def _create_summary(self, context: str, query: str) -> str:
        """Create a clean summary from retrieved context."""
        # Extract key sentences - remove duplicates
        sentences = []
        seen = set()
        for s in context.split('.'):
            s_clean = s.strip()
            # Remove duplicates and filter short/low-quality sentences
            if len(s_clean) > 40 and s_clean.lower() not in seen:
                sentences.append(s_clean)
                seen.add(s_clean.lower())
                if len(sentences) >= 6:
                    break
        
        if not sentences:
            return context[:800]
        
        response = "## Summary\n\n"
        for i, point in enumerate(sentences, 1):
            response += f"{i}. {point}.\n\n"
        
        return response.strip()
    
    def _create_explanation(self, context: str, query: str) -> str:
        """Create a well-formatted explanation from retrieved context."""
        # Remove duplicate paragraphs
        paragraphs = []
        seen = set()
        for para in context.split('\n\n'):
            para_clean = para.strip()
            # Keep unique, substantial paragraphs
            if len(para_clean) > 50:
                para_lower = para_clean.lower()[:200]  # Check first 200 chars for duplicates
                if para_lower not in seen:
                    paragraphs.append(para_clean)
                    seen.add(para_lower)
        
        if not paragraphs:
            # Fallback: split by sentence
            sentences = [s.strip() + '.' for s in context.split('.') if len(s.strip()) > 30]
            return ' '.join(sentences[:8])
        
        # Build clean, formatted response with proper spacing
        response = ""
        
        # Add first paragraph as main explanation (ensure spacing between sentences)
        first_para = paragraphs[0]
        # Add space after punctuation if missing
        import re
        first_para = re.sub(r'([.!?])([A-Z])', r'\1 \2', first_para)
        response += first_para
        
        # Add additional details if available
        if len(paragraphs) > 1:
            response += "\n\n### Key Points:\n\n"
            for i, para in enumerate(paragraphs[1:4], 1):  # Max 3 additional points
                # Extract first sentence as bullet
                sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 20]
                if sentences:
                    response += f"• {sentences[0]}.\n"
                    if len(sentences) > 1 and len(sentences[1]) > 20:
                        response += f"  {sentences[1]}.\n"
                    response += "\n"
        
        return response.strip()
    
    def _create_qa(self, context: str, query: str) -> str:
        """Answer a question with clean formatting."""
        # Find most relevant paragraphs
        paragraphs = [p.strip() for p in context.split('\n\n') if len(p.strip()) > 50]
        
        if not paragraphs:
            sentences = [s.strip() + '.' for s in context.split('.') if len(s.strip()) > 30]
            return ' '.join(sentences[:6])
        
        # Remove duplicates
        unique_paras = []
        seen = set()
        for para in paragraphs:
            para_key = para.lower()[:150]
            if para_key not in seen:
                unique_paras.append(para)
                seen.add(para_key)
                if len(unique_paras) >= 3:
                    break
        
        # Fix spacing in response
        import re
        response = unique_paras[0] if unique_paras else context[:800]
        response = re.sub(r'([.!?])([A-Z])', r'\1 \2', response)
        
        # Add supporting details if available
        if len(unique_paras) > 1:
            second_para = re.sub(r'([.!?])([A-Z])', r'\1 \2', unique_paras[1])
            response += "\n\n" + second_para
        
        return response.strip()
    
    def _create_notes(self, context: str, query: str) -> str:
        """Create well-structured study notes."""
        # Split and clean sections
        sections = [s.strip() for s in context.split('\n\n') if len(s.strip()) > 40]
        
        # Remove duplicates
        unique_sections = []
        seen = set()
        for section in sections:
            section_key = section.lower()[:100]
            if section_key not in seen:
                unique_sections.append(section)
                seen.add(section_key)
                if len(unique_sections) >= 6:
                    break
        
        if not unique_sections:
            return context[:1000]
        
        response = "## Study Notes\n\n"
        
        for i, section in enumerate(unique_sections, 1):
            # Extract key information
            sentences = [s.strip() for s in section.split('.') if len(s.strip()) > 20]
            
            if sentences:
                # Use first sentence as heading
                heading = sentences[0]
                response += f"### {i}. {heading}\n\n"
                
                # Add bullet points for remaining content
                for sent in sentences[1:3]:  # Max 2 additional sentences
                    response += f"- {sent}\n"
                response += "\n"
        
        return response.strip()
