import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import config


class ModelInference:
    """Handle model loading and inference for text generation."""
    
    def __init__(self, model_name: str = None, use_4bit: bool = True):
        """
        Initialize the model for inference.
        RAG Mode: Uses pre-trained model directly (no training needed!).
        
        Args:
            model_name: Name or path of the model (uses pre-trained by default)
            use_4bit: Whether to use 4-bit quantization for efficiency
        """
        # Use pre-trained model if specified, otherwise check for fine-tuned model
        if config.USE_PRETRAINED or not Path(config.MODEL_PATH).exists():
            self.model_name = model_name or config.MODEL_NAME
        else:
            self.model_name = model_name or config.MODEL_PATH
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Configure quantization for efficiency
        if use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def generate_response(
        self,
        prompt: str,
        context: str = "",
        use_case: str = "explanation",
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate a response based on the prompt and context.
        
        Args:
            prompt: User query
            context: Retrieved context from documents
            use_case: Type of response (explanation, summary, qa, notes)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        temperature = temperature or config.TEMPERATURE
        max_tokens = max_tokens or config.MAX_TOKENS
        
        # Create system prompt based on use case
        system_prompts = {
            "explanation": "You are an expert tutor. Provide detailed, clear explanations of concepts based on the given context.",
            "summary": "You are a summarization expert. Create concise, well-structured summaries of the provided content.",
            "qa": "You are a knowledgeable assistant. Answer questions accurately based on the given context.",
            "notes": "You are a study notes specialist. Create well-organized, structured study notes from the content."
        }
        
        system_prompt = system_prompts.get(use_case, system_prompts["explanation"])
        
        # Format the full prompt
        full_prompt = self._format_prompt(system_prompt, context, prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = response[len(full_prompt):].strip()
        
        return response
    
    def _format_prompt(self, system_prompt: str, context: str, query: str) -> str:
        """Format the prompt with system instructions, context, and query."""
        prompt = f"{system_prompt}\n\n"
        
        if context:
            prompt += f"Context from your study materials:\n{context}\n\n"
        
        prompt += f"Query: {query}\n\nResponse:"
        
        return prompt
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Additional arguments for generate_response
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
        
        return responses
