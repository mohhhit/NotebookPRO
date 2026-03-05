"""
Real LLM-based generator using Groq or Google Gemini API.
This ACTUALLY generates responses (unlike SimpleGenerator which just extracts text).
"""

import os
from typing import List, Dict, Optional
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMGenerator:
    """
    Actual LLM-based response generation using Groq (Llama-3-70B) or Gemini.
    This is what NotebookLM uses - real AI generation, not text extraction.
    """
    
    def __init__(self, provider: str = "groq", api_key: Optional[str] = None):
        """
        Initialize LLM generator.
        
        Args:
            provider: "groq" or "gemini"
            api_key: API key (if None, reads from environment or asks user)
        """
        self.provider = provider
        self.client = None
        self.ready = False
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY", "")
        elif provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        else:
            self.api_key = ""
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if not self.api_key:
            return
        
        try:
            if self.provider == "groq" and GROQ_AVAILABLE:
                self.client = Groq(api_key=self.api_key)
                self.ready = True
            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-1.5-flash')
                self.ready = True
        except Exception as e:
            print(f"Failed to initialize {self.provider}: {e}")
            self.ready = False
    
    def set_api_key(self, api_key: str):
        """Update API key and reinitialize."""
        self.api_key = api_key
        self._initialize_client()
    
    def generate_response(
        self,
        prompt: str,
        context: str = "",
        use_case: str = "explanation",
        metadatas: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """
        Generate response using actual LLM (NotebookLM-style).
        
        Args:
            prompt: User's question
            context: Retrieved context from documents
            use_case: Response type (explanation, summary, qa, notes)
            metadatas: Metadata for citations
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum response length
            
        Returns:
            Generated response with inline citations
        """
        if not self.ready:
            return (
                "⚠️ **LLM not configured.** Please add your API key in the sidebar.\n\n"
                "Get a free key:\n"
                "- **Groq** (recommended, very fast): https://console.groq.com/keys\n"
                "- **Gemini** (Google): https://makersuite.google.com/app/apikey"
            )
        
        if not context:
            return (
                "I don't have enough information from your uploaded documents to answer this question. "
                "Please upload relevant study materials first."
            )
        
        # Build NotebookLM-style system prompt with strict source grounding
        system_prompt = self._build_system_prompt(use_case)
        
        # Build user message with context
        user_message = self._build_user_message(prompt, context, metadatas)
        
        try:
            # Generate with LLM
            if self.provider == "groq":
                response = self._generate_groq(system_prompt, user_message, temperature, max_tokens)
            elif self.provider == "gemini":
                response = self._generate_gemini(system_prompt, user_message, temperature, max_tokens)
            else:
                return "Error: Unknown provider"
            
            return response
        
        except Exception as e:
            return f"Error generating response: {str(e)}\n\nPlease check your API key and try again."
    
    def _build_system_prompt(self, use_case: str) -> str:
        """Build specialized system prompt based on use case."""
        base_prompt = (
            "You are an expert academic assistant for Data Science students. "
            "⚠️ CRITICAL RULE: You MUST ONLY use information from the provided context below. "
            "DO NOT use your training knowledge. DO NOT infer beyond what's explicitly stated. "
            "If the context doesn't contain adequate information to answer the question, you MUST respond: "
            "'I cannot find sufficient information about this in the uploaded documents. Please upload materials covering this topic or rephrase your question.'\n\n"
            "⚠️ GROUNDING REQUIREMENT: Every statement must be traceable to the provided context. "
            "If you cannot find it in the context below, DO NOT answer from general knowledge.\n\n"
        )
        
        if use_case == "explanation":
            base_prompt += (
                "**Your task:** Explain the concept in a clear, step-by-step manner suitable for students.\n"
                "- Start with a concise definition\n"
                "- Break down complex ideas into digestible parts\n"
                "- Use examples from the context when available\n"
                "- Add a 'Key Points' section with numbered insights\n"
                "- Use **bold** for important terms and concepts\n"
            )
        elif use_case == "summary":
            base_prompt += (
                "**Your task:** Create a structured summary.\n"
                "- Start with a brief overview (2-3 sentences)\n"
                "- List key points as numbered items\n"
                "- Keep each point concise but informative\n"
                "- Use **bold** for critical terms\n"
            )
        elif use_case == "qa":
            base_prompt += (
                "**Your task:** Answer the question directly and concisely.\n"
                "- Start with a direct answer\n"
                "- Provide supporting details from the context\n"
                "- If there are multiple aspects, use numbered points\n"
                "- Use **bold** for key facts\n"
            )
        elif use_case == "notes":
            base_prompt += (
                "**Your task:** Create structured study notes.\n"
                "- Use clear section headers (###)\n"
                "- Organize information hierarchically\n"
                "- Use bullet points for details\n"
                "- Highlight definitions and formulas\n"
            )
        
        base_prompt += (
            "\n**Citation Rules:**\n"
            "- After making a claim, cite the source document name in brackets like **[DocumentName]**\n"
            "- If information comes from multiple sources, cite all relevant ones\n"
            "- Do NOT make up information - stick strictly to the provided context\n"
        )
        
        return base_prompt
    
    def _build_user_message(self, prompt: str, context: str, metadatas: List[Dict] = None) -> str:
        """Build user message with context and question."""
        # Extract source names from metadata
        sources = []
        if metadatas:
            for meta in metadatas:
                filename = meta.get('filename', 'Unknown')
                clean_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
                if clean_name not in sources:
                    sources.append(clean_name)
        
        message = "**Available Sources (USE ONLY THESE):**\n"
        for source in sources[:5]:  # Show up to 5 sources
            message += f"- {source}\n"
        
        message += f"\n**===== START OF CONTEXT (ANSWER ONLY FROM THIS) =====**\n\n{context}\n\n"
        message += f"**===== END OF CONTEXT =====**\n\n"
        message += f"**Student's Question:** {prompt}\n\n"
        message += "**Instructions:** Answer ONLY using the context between the markers above. If the context doesn't contain the answer, say you don't have that information. Cite sources in brackets."
        
        return message
    
    def _generate_groq(self, system_prompt: str, user_message: str, temperature: float, max_tokens: int) -> str:
        """Generate using Groq API (Llama-3.3-70B)."""
        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Latest 70B model (Dec 2024)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            stream=False
        )
        
        return completion.choices[0].message.content
    
    def _generate_gemini(self, system_prompt: str, user_message: str, temperature: float, max_tokens: int) -> str:
        """Generate using Google Gemini API."""
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95
            )
        )
        
        return response.text
    
    def is_ready(self) -> bool:
        """Check if LLM is ready to generate."""
        return self.ready
    
    def get_provider(self) -> str:
        """Get current provider name."""
        if self.provider == "groq":
            return "Groq (Llama-3.3-70B)"
        elif self.provider == "gemini":
            return "Google Gemini 1.5 Flash"
        return "Unknown"
