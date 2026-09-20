"""
Gemini-based answer generation with NVIDIA NIM follow-up support.
This generates grounded responses from retrieved context.
"""

import os
import re
import time
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    NVIDIA_CLIENT_AVAILABLE = True
except ImportError:
    NVIDIA_CLIENT_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_CLIENT_AVAILABLE = True
except ImportError:
    GEMINI_CLIENT_AVAILABLE = False


class LLMGenerator:
    """Gemini answer generation with NVIDIA follow-up support."""

    def __init__(
        self,
        provider: str = "nvidia",
        api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        del provider

        self.provider = "gemini"
        self.gemini_client = None
        self.nvidia_client = None
        self.groq_client = None
        self.ready = False
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.nvidia_model_name = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-pro")
        self.api_base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.groq_model_name = os.getenv("GROQ_FOLLOWUP_MODEL", "openai/gpt-oss-20b")
        self.groq_api_base_url = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
        self.enable_thinking = os.getenv("NVIDIA_ENABLE_THINKING", "false").lower() == "true"
        self.end_marker = "<<END_OF_ANSWER>>"
        self.last_notice = None
        self._last_finish_reason = None

        self.gemini_api_key = gemini_api_key or api_key or os.getenv("GEMINI_API_KEY", "")
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Gemini and NVIDIA clients."""
        self.gemini_client = None
        self.nvidia_client = None
        self.groq_client = None

        if self.gemini_api_key and GEMINI_CLIENT_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = True
            except Exception as e:
                print(f"Failed to initialize gemini client: {e}")

        if self.nvidia_api_key and NVIDIA_CLIENT_AVAILABLE:
            try:
                self.nvidia_client = OpenAI(
                    base_url=self.api_base_url,
                    api_key=self.nvidia_api_key,
                )
            except Exception as e:
                print(f"Failed to initialize nvidia client: {e}")

        if self.groq_api_key and NVIDIA_CLIENT_AVAILABLE:
            try:
                self.groq_client = OpenAI(
                    base_url=self.groq_api_base_url,
                    api_key=self.groq_api_key,
                )
            except Exception as e:
                print(f"Failed to initialize groq client: {e}")

        self.ready = bool(self.gemini_client or self.nvidia_client)

    def set_api_key(self, api_key: str, provider: str = "gemini"):
        """Update an API key and reinitialize the matching client."""
        if provider == "nvidia":
            self.nvidia_api_key = api_key
        elif provider == "groq":
            self.groq_api_key = api_key
        else:
            self.gemini_api_key = api_key
        self._initialize_clients()

    def _is_quota_error(self, err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "429" in msg
            or "quota" in msg
            or "rate limit" in msg
            or "resource_exhausted" in msg
            or "insufficient_quota" in msg
        )

    def _looks_truncated(self, text: str) -> bool:
        """Heuristic check for cut-off model outputs."""
        if not text:
            return False

        stripped = text.rstrip()
        if len(stripped) < 160:
            return False

        if self.end_marker in stripped:
            return False

        if stripped.endswith((".", "!", "?", "]", ")", '"', "'", "```")):
            return (stripped.count("**") % 2 != 0) or (stripped.count("```") % 2 != 0)

        if stripped[-1].isalnum():
            return True

        if stripped.count("**") % 2 != 0 or stripped.count("```") % 2 != 0:
            return True

        dangling_words = (
            " and", " or", " to", " of", " for", " with", " by", " in", " is", " are", " a", " an", " the"
        )
        lowered = stripped.lower()
        return lowered.endswith(dangling_words) or stripped.endswith((":", "-", "*", ",", ";"))

    def _has_end_marker(self, text: str) -> bool:
        return bool(text) and self.end_marker in text

    def _split_table_cells(self, line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip('|').split('|')]

    def _is_markdown_table_separator(self, line: str) -> bool:
        if '|' not in line:
            return False
        cells = self._split_table_cells(line)
        if len(cells) < 2:
            return False
        return all(bool(re.fullmatch(r':?-{3,}:?', c.replace(' ', ''))) for c in cells)

    def _is_markdown_table_row(self, line: str) -> bool:
        if '|' not in line:
            return False
        cells = self._split_table_cells(line)
        return len(cells) >= 2 and any(c for c in cells)

    def _normalize_table_row(self, cells: List[str], col_count: int) -> str:
        normalized = [c.strip() for c in cells[:col_count]]
        if len(normalized) < col_count:
            normalized.extend([''] * (col_count - len(normalized)))
        return f"| {' | '.join(normalized)} |"

    def _expand_compact_table_line(self, line: str) -> List[str]:
        """Expand single-line compact table blobs into one row per line."""
        if line.count('|') < 8 or '| |' not in line:
            return [line]

        tokens = [t.strip() for t in line.split('|')]
        rows: List[List[str]] = []
        current: List[str] = []

        for token in tokens:
            if token == '':
                if len(current) >= 2:
                    rows.append(current)
                current = []
                continue
            current.append(token)

        if len(current) >= 2:
            rows.append(current)

        if len(rows) < 2:
            return [line]

        col_count = max(len(r) for r in rows)
        return [self._normalize_table_row(r, col_count) for r in rows]

    def _repair_markdown_tables(self, text: str) -> str:
        """Repair malformed markdown tables (missing separator or compact inline rows)."""
        if not text or '|' not in text:
            return text

        expanded_lines: List[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                expanded_lines.append('')
                continue
            expanded_lines.extend(self._expand_compact_table_line(stripped))

        repaired: List[str] = []
        i = 0
        while i < len(expanded_lines):
            line = expanded_lines[i].strip()

            if not self._is_markdown_table_row(line):
                repaired.append(line)
                i += 1
                continue

            block: List[str] = []
            while i < len(expanded_lines):
                probe = expanded_lines[i].strip()
                if not probe:
                    break
                if not (self._is_markdown_table_row(probe) or self._is_markdown_table_separator(probe)):
                    break
                block.append(probe)
                i += 1

            if (
                len(block) >= 2
                and self._is_markdown_table_row(block[0])
                and not self._is_markdown_table_separator(block[1])
                and self._is_markdown_table_row(block[1])
            ):
                col_count = max(2, len(self._split_table_cells(block[0])))
                separator = f"| {' | '.join(['---'] * col_count)} |"
                block.insert(1, separator)

            repaired.extend(block)
            if i < len(expanded_lines) and expanded_lines[i].strip() == '':
                repaired.append('')
                i += 1

        return '\n'.join(repaired)

    def _normalize_math_delimiters(self, text: str) -> str:
        """Normalize model math delimiters for Flutter markdown math parser."""
        if not text:
            return ""

        normalized = re.sub(
            r'\\\[([\s\S]*?)\\\]',
            lambda m: f"$${m.group(1).strip()}$$",
            text,
            flags=re.DOTALL,
        )
        normalized = re.sub(
            r'\\\(([^\n]*?)\\\)',
            lambda m: f"${m.group(1).strip()}$",
            normalized,
        )

        normalized = re.sub(r'\$\$\s+', '$$', normalized)
        normalized = re.sub(r'\s+\$\$', '$$', normalized)
        normalized = re.sub(r'\$\s+', '$', normalized)
        normalized = re.sub(r'\s+\$', '$', normalized)
        return normalized

    def _normalize_formula_equations(self, text: str) -> str:
        """Wrap plain-text metric equations in LaTeX delimiters when missing."""
        if not text:
            return ""

        metric_eq = re.compile(
            r'\b(MAE|RMSE|MAPE|MASE|MSE|R2|R\^2|BLEU|METEOR|BERTScore|ROUGE(?:-[A-Za-z0-9]+)?)\s*=\s*([^$\n]+?)(?=(?:\s*(?:,?\s*where\b|[.;]|$)))',
            flags=re.IGNORECASE,
        )
        formula_label = re.compile(
            r'(?i)(\bformula\s*:\s*)([^$\n]*?=[^$\n]+?)(?=(?:\s*(?:[.;]|$)))'
        )

        out_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line
            if '$' in line or '|' in line:
                out_lines.append(line)
                continue

            # Repair common missing space artifacts around acronyms (e.g., asMAE, whereN).
            line = re.sub(r'([a-z])([A-Z]{2,})', r'\1 \2', line)

            def _wrap_formula_label(match: re.Match) -> str:
                prefix = match.group(1)
                expr = match.group(2).strip()
                return f"{prefix}${expr}$"

            line = formula_label.sub(_wrap_formula_label, line)

            def _wrap_metric_eq(match: re.Match) -> str:
                expr = f"{match.group(1)} = {match.group(2).strip()}"
                return f"${expr}$"

            line = metric_eq.sub(_wrap_metric_eq, line)
            out_lines.append(line)

        return "\n".join(out_lines)

    def _sanitize_final_response(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace(self.end_marker, "").strip()
        cleaned = self._normalize_math_delimiters(cleaned)
        cleaned = self._normalize_formula_equations(cleaned)
        # Strip optional reasoning wrappers if the model emits chain-of-thought blocks.
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>', '', cleaned, flags=re.IGNORECASE)
        cleaned_lines = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue

            # Guardrail against OCR-like numbering artifacts leaking into UI output.
            if re.search(r'(?:\d+\.){8,}\d*', line):
                continue

            compact = re.sub(r'\s+', '', line)
            if len(compact) >= 40 and re.fullmatch(r'[\d\W_]+', compact):
                continue

            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        cleaned = self._repair_markdown_tables(cleaned)
        return cleaned

    def _strip_academic_note_section(self, text: str) -> str:
        """Remove trailing Academic Note sections from assistant output."""
        if not text:
            return ""

        marker = "### Academic Note:"
        idx = text.find(marker)
        if idx == -1:
            return text.strip()
        return text[:idx].strip()

    def _infer_academic_depth_mode(self, prompt: str, use_case: str) -> bool:
        """Detect questions that should receive rigorous, exam-grade treatment."""
        if use_case not in {"qa", "explanation", "notes"}:
            return False

        lower = (prompt or "").lower()
        triggers = (
            "evaluate", "evaluation", "compare", "contrast", "analyze", "analysis",
            "justify", "derive", "mathematical", "formula", "metric", "metrics",
            "bleu", "rouge", "meteor", "bertscore", "nlp", "transformer",
            "limitations", "assumptions", "deep learning", "exam", "academic",
        )
        return any(t in lower for t in triggers)

    def _build_quality_checklist(self, prompt: str, use_case: str) -> str:
        """Return additional instruction block to improve academic completeness."""
        if not self._infer_academic_depth_mode(prompt, use_case):
            return ""

        checklist = (
            "\n**Academic Completeness Checklist (apply when evidence exists in context):**\n"
            "1. Include core mechanism details, not just definitions.\n"
            "2. Include at least one relevant equation, formula component, or scoring term when present.\n"
            "3. State known limitations/failure cases and why they matter.\n"
            "4. Mention important variants/versions (for example, subtypes or related metrics) when supported by context.\n"
            "5. Briefly contrast with modern alternatives if context includes them.\n"
            "6. Synthesize evidence from multiple source snippets when available.\n"
            "7. Keep structure clear: definition -> mechanics -> limitations -> modern view -> takeaway.\n"
            "8. Render all equations in LaTeX ($...$ inline, $$...$$ block), never plain-text formulas.\n"
            "Note: If a checklist item is not supported by the provided context, explicitly say that.\n"
        )

        lower = (prompt or "").lower()
        if "bleu" in lower or "rouge" in lower or "summarization" in lower or "translation" in lower:
            checklist += (
                "\n**NLP Metric Focus (for BLEU/ROUGE-style questions):**\n"
                "- For BLEU: explicitly check for and mention **Brevity Penalty (BP)** when present in context.\n"
                "- For ROUGE: avoid reducing it to recall only; mention precision/recall/F1 behavior when supported.\n"
                "- Mention key ROUGE variants such as **ROUGE-N** and **ROUGE-L (LCS)** if found in context.\n"
                "- If context includes modern semantic metrics (for example **BERTScore**), include them as current practice.\n"
                "- If any of the above are missing in context, explicitly note that limitation instead of guessing.\n"
            )

        return checklist

    def _needs_depth_boost(self, prompt: str, response: str, use_case: str) -> bool:
        """Detect shallow first-pass answers for academically demanding questions."""
        if not self._infer_academic_depth_mode(prompt, use_case):
            return False

        word_count = len((response or "").split())
        if use_case == "qa":
            return word_count < 240
        return word_count < 210

    def _expand_shallow_response(
        self,
        system_prompt: str,
        prompt: str,
        context: str,
        partial_response: str,
        temperature: float,
        max_tokens: int,
        use_case: str,
    ) -> str:
        """Run one targeted expansion pass to improve rigor without changing grounding."""
        quality_checklist = self._build_quality_checklist(prompt, use_case)
        expansion_instruction = (
            "The previous answer is correct but too shallow for an advanced academic response. "
            "Expand it using ONLY the provided context, keeping all citation rules.\n\n"
            f"Question: {prompt}\n\n"
            f"Context:\n{context}\n\n"
            "Current answer:\n"
            f"{partial_response}\n\n"
            "Now produce a richer version with stronger mechanics, limitations, and comparisons where supported by context.\n"
            "Ensure equations are consistently rendered in LaTeX using $...$ and $$...$$.\n"
            "Target depth: around 300-550 words for QA-style prompts when context supports it.\n"
            f"{quality_checklist}\n"
            ""
        )

        expansion_tokens = max(900, min(1800, max_tokens))

        try:
            return self._generate_nvidia(
                system_prompt,
                expansion_instruction,
                temperature=max(0.2, min(temperature, 0.45)),
                max_tokens=expansion_tokens,
                thinking=True,
            )
        except Exception:
            return ""

    def _finish_reason_indicates_cutoff(self) -> bool:
        """Check whether model stop reason indicates truncation due token limits."""
        if self._last_finish_reason is None:
            return False

        reason = str(self._last_finish_reason).lower()
        return (
            "max_tokens" in reason
            or "max token" in reason
            or "length" in reason
            or "token_limit" in reason
        )

    def _continue_truncated_response(
        self,
        system_prompt: str,
        prompt: str,
        context: str,
        partial_response: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Request one continuation pass when the first answer looks incomplete."""
        continuation_instruction = (
            "Your previous answer was cut off. Continue exactly from where it stopped. "
            "Do not repeat any completed sections. Keep the same structure and citation style.\n\n"
            f"Original question: {prompt}\n\n"
            f"Context:\n{context}\n\n"
            "Partial answer already generated:\n"
            f"{partial_response}\n\n"
            "Now provide only the missing continuation and end with the exact marker "
            f"{self.end_marker}."
        )

        continuation_tokens = max(600, min(1400, max_tokens // 2))

        try:
            return self._generate_gemini(
                system_prompt,
                continuation_instruction,
                temperature=max(0.2, min(temperature, 0.5)),
                max_tokens=continuation_tokens,
            )
        except Exception:
            return ""

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
        """Generate response using Gemini with source grounding."""
        del kwargs

        if not self.ready:
            return (
                "LLM not configured. Please add a Gemini or NVIDIA API key in settings.\n\n"
                "Get a Gemini key: https://aistudio.google.com/app/apikey"
            )

        if not context:
            return (
                "I don't have enough information from your uploaded documents to answer this question. "
                "Please upload relevant study materials first."
            )

        system_prompt = self._build_system_prompt(use_case)
        user_message = self._build_user_message(
            prompt,
            context,
            metadatas,
            use_case=use_case,
        )

        self.last_notice = None
        self._last_finish_reason = None

        try:
            if self.gemini_client:
                response = self._generate_gemini(
                    system_prompt,
                    user_message,
                    temperature,
                    max_tokens,
                )
            elif self.nvidia_client:
                response = self._generate_nvidia(
                    system_prompt,
                    user_message,
                    temperature,
                    max_tokens,
                    thinking=False,
                )
            else:
                raise RuntimeError("No configured generation client")

            for _ in range(3):
                if self._has_end_marker(response):
                    break

                if not (
                    self._finish_reason_indicates_cutoff()
                    or self._looks_truncated(response)
                ):
                    break

                continuation = self._continue_truncated_response(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    context=context,
                    partial_response=response,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if continuation:
                    print("[LLM_TIMING] continuation_pass=1")
                    response = response.rstrip() + "\n" + continuation.lstrip()
                else:
                    break

            final_response = self._sanitize_final_response(response)
            final_response = self._strip_academic_note_section(final_response)

            if self._needs_depth_boost(prompt, final_response, use_case):
                clean_partial_response = self._strip_academic_note_section(final_response)

                expanded = self._expand_shallow_response(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    context=context,
                    partial_response=clean_partial_response,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_case=use_case,
                )
                if expanded:
                    print("[LLM_TIMING] depth_boost=1")
                    final_response = self._sanitize_final_response(expanded)
                    final_response = self._strip_academic_note_section(final_response)

            return final_response

        except Exception as e:
            return (
                "Error generating response: Gemini request failed.\n\n"
                f"Details: {str(e)}"
            )

    def generate_follow_up_questions(
        self,
        prompt: str,
        context: str,
        answer: str,
        max_questions: int = 3,
    ) -> List[str]:
        """Generate short, actionable follow-up questions for the chat UI."""
        if not self.nvidia_client or not context or not answer:
            return []

        max_questions = max(1, min(max_questions, 5))

        system_prompt = (
            "You generate concise suggested follow-up questions for an academic chat app. "
            "Return only questions, one per line, no numbering, no bullets, no explanations."
        )
        user_message = (
            f"Original question:\n{prompt}\n\n"
            f"Assistant answer:\n{answer}\n\n"
            f"Generate {max_questions} follow-up questions that the student would logically "
            "ask NEXT, covering concepts NOT already explained in the answer "
            "above. Do NOT rephrase or re-ask anything that was already "
            "addressed in the answer. Focus on: deeper mechanics, related "
            "concepts, or practical implementation details that were mentioned "
            "but not elaborated on.\n\n"
            f"Context:\n{context}"
        )

        def parse_questions(raw_output: str) -> List[str]:
            cleaned = self._sanitize_final_response(raw_output)
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            suggestions: List[str] = []
            seen = set()

            for line in lines:
                q = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line).strip()
                q = q.strip('"').strip("'").strip()
                if not q:
                    continue
                if not q.endswith('?'):
                    q = f"{q}?"
                if len(q) < 12 or len(q) > 180:
                    continue

                key = q.lower()
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(q)

                if len(suggestions) >= max_questions:
                    break

            return suggestions

        def request_questions(client, model_name: str, provider: str) -> List[str]:
            if not client:
                return []
            try:
                raw_output = self._generate_openai_compatible(
                    client=client,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=0.3,
                    max_tokens=220,
                    model_name=model_name,
                    provider=provider,
                )
                return parse_questions(raw_output)
            except Exception as e:
                print(f"Error in {provider} follow-up generation: {e}")
                return []

        suggestions = request_questions(
            self.nvidia_client,
            os.getenv("FOLLOWUP_MODEL", " "),
            "nvidia",
        )
        if suggestions:
            return suggestions

        print("[LLM_TIMING] followup_fallback=groq")
        return request_questions(self.groq_client, self.groq_model_name, "groq")

    def _generate_gemini(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        model_name: Optional[str] = None,
    ) -> str:
        """Generate using Gemini API."""
        if not self.gemini_api_key or not GEMINI_CLIENT_AVAILABLE:
            raise RuntimeError("Gemini client is not configured")

        active_model = model_name or self.gemini_model_name
        t0 = time.perf_counter()
        print(
            f"[LLM_TIMING] request_start model={active_model} provider=gemini "
            f"max_tokens={max_tokens} temperature={temperature} prompt_chars={len(user_message)}"
        )

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(model_name=active_model)
        response = model.generate_content(
            f"{system_prompt}\n\n{user_message}",
            generation_config={
                "temperature": temperature,
                "top_p": 0.7,
                "max_output_tokens": max_tokens,
            },
        )

        content = getattr(response, "text", "") or ""
        print(
            f"[LLM_TIMING] request_done ms={(time.perf_counter() - t0) * 1000:.1f} "
            f"response_chars={len(content or '')}"
        )

        return content or ""

    def generate_stream(
        self,
        prompt: str,
        context: str = "",
        use_case: str = "explanation",
        metadatas: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        **kwargs
    ):
        """Generate response using Gemini or NVIDIA with source grounding via live streaming."""
        del kwargs

        if not self.ready:
            yield (
                "LLM not configured. Please add a Gemini or NVIDIA API key in settings.\n\n"
                "Get a Gemini key: https://aistudio.google.com/app/apikey"
            )
            return

        if not context:
            yield (
                "I don't have enough information from your uploaded documents to answer this question. "
                "Please upload relevant study materials first."
            )
            return

        system_prompt = self._build_system_prompt(use_case)
        user_message = self._build_user_message(
            prompt,
            context,
            metadatas,
            use_case=use_case,
        )

        self.last_notice = None
        self._last_finish_reason = None

        if self.gemini_client:
            yield from self._generate_gemini_stream(
                system_prompt,
                user_message,
                temperature,
                max_tokens,
            )
        elif self.nvidia_client:
            yield from self._generate_nvidia_stream(
                system_prompt,
                user_message,
                temperature,
                max_tokens,
                thinking=False,
            )
        else:
            raise RuntimeError("No configured generation client")

    def _generate_gemini_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        model_name: Optional[str] = None,
    ):
        """Generate using Gemini API with streaming."""
        if not self.gemini_api_key or not GEMINI_CLIENT_AVAILABLE:
            raise RuntimeError("Gemini client is not configured")

        active_model = model_name or self.gemini_model_name
        t0 = time.perf_counter()
        print(
            f"[LLM_TIMING] request_stream_start model={active_model} provider=gemini "
            f"max_tokens={max_tokens} temperature={temperature} prompt_chars={len(user_message)}"
        )

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(model_name=active_model)
        
        try:
            response = model.generate_content(
                f"{system_prompt}\n\n{user_message}",
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.7,
                    "max_output_tokens": max_tokens,
                },
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"[LLM_ERROR] Gemini stream failed: {e}")
            yield f"\n\n[Generation Error: {str(e)}]"
            
        print(f"[LLM_TIMING] stream_done ms={(time.perf_counter() - t0) * 1000:.1f}")

    def _generate_nvidia_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        model_name: Optional[str] = None,
        thinking: bool = True,
    ):
        """Generate using NVIDIA NIM OpenAI-compatible API with streaming."""
        if not self.nvidia_client:
            raise RuntimeError("NVIDIA client is not configured")

        active_model = model_name or self.nvidia_model_name
        t0 = time.perf_counter()
        print(
            f"[LLM_TIMING] request_stream_start model={active_model} thinking={bool(thinking)} "
            f"max_tokens={max_tokens} temperature={temperature} prompt_chars={len(user_message)}"
        )
        
        extra_body = {}
        if "deepseek" in active_model.lower():
            extra_body["chat_template_kwargs"] = {"thinking": bool(thinking)}

        try:
            response = self.nvidia_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.7,
                extra_body=extra_body,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"[LLM_ERROR] NVIDIA stream failed: {e}")
            yield f"\n\n[Generation Error: {str(e)}]"
            
        print(f"[LLM_TIMING] stream_done ms={(time.perf_counter() - t0) * 1000:.1f}")

    def _build_system_prompt(self, use_case: str) -> str:
        """Build specialized system prompt based on use case."""
        base_prompt = (
            "You are the expert generation engine for NotebookPRO, an advanced academic study system for Master-level data science and NLP coursework. "
            "Your goal is to synthesize retrieved textbook excerpts into rigorous, mathematically sound, comprehensive study material.\n\n"
            "STRICT GROUNDING: You MUST ONLY use information present inside the provided context. "
            "Do NOT use outside knowledge and do NOT hallucinate. Every technical claim must be traceable to context.\n\n"
            "TEMPORAL ARBITRATION & SYNTHESIS: Context chunks are labeled as Source [i] [YEAR: YYYY] (filename). "
            "You must analyze year tags. If multiple sources conflict or show conceptual evolution, do NOT blindly merge. "
            "Prioritize the most recent year as modern standard/state-of-the-art, and frame older sources as foundational/historical context.\n\n"
            "ACADEMIC RIGOR: Avoid superficial summaries. "
            "Explain mechanisms, formulas, penalties, architectural details, limitations, and failure modes when present. "
            "Explicitly contrast legacy approaches with modern ones when context supports it.\n\n"
            "REQUIRED OUTPUT STRUCTURE:\n"
            "- Use clear markdown headings with ### for major sections.\n"
            "- Use bulleted breakdowns with bold key terms (for example, * **Mechanism:** ...).\n"
            "- Render formulas in LaTeX using $...$ for inline math and $$...$$ for block math.\n"
            "- Do not use \\(...\\) or \\[...\\] delimiters.\n"
            "- Keep formula notation consistent throughout the answer (do not mix plain-text and LaTeX equations).\n"
            "- For tabular values, always use valid markdown tables: header row, separator row (---), then body rows.\n"
            "- Keep explanations dense, structured, and exam-ready.\n"
            "- Do not add UI-style extras such as 'Suggested Follow-Up Questions' inside the answer body.\n"
            "- NEVER use phrases like 'based on the context', 'the text mentions', 'the provided excerpt states', or 'the sources lack'. Speak with absolute academic authority as if YOU are the textbook.\n"
            "- If advanced requested detail is missing from context, DO NOT mention the limitation. Just answer what you can.\n\n"
        )

        if use_case == "explanation":
            base_prompt += (
                "**Your task:** Explain the concept in a clear, step-by-step manner suitable for students.\n"
                "1. Start with a concise, one-sentence definition.\n"
                "2. Break down the core mechanics or components using bullet points.\n"
                "3. Provide an example (only if found in the text).\n"
                "4. Add a 'Key Takeaway' at the end.\n"
            )
        elif use_case == "summary":
            base_prompt += (
                "**Your task:** Create a highly structured summary.\n"
                "- Start with a brief high-level overview (2 sentences max).\n"
                "- Use '### Key Themes' and list the main points as bulleted items.\n"
                "- Keep each point concise but factually dense.\n"
            )
        elif use_case == "qa":
            base_prompt += (
                "**Your task:** Answer the question directly and comprehensively.\n"
                "- Provide the direct answer immediately in the first sentence.\n"
                "- Build a deep explanation using multiple complementary points from the context, not just one short paragraph.\n"
                "- When the question asks 'why', include mechanisms, business impact, risks, and implementation implications if present in context.\n"
                "- Synthesize across multiple sources and explicitly contrast or connect them when applicable.\n"
                "- Use numbered lists or bullet points to provide supporting details from the context.\n"
                "- Use **bold** for key facts, numbers, and formulas.\n"
                "- Aim for substantial depth (typically 250-500 words) unless the context is very limited.\n"
            )
        elif use_case == "notes":
            base_prompt += (
                "**Your task:** Create comprehensive, structured study notes.\n"
                "- Use clear section headers (###).\n"
                "- Organize information hierarchically (using nested bullet points).\n"
                "- Explicitly highlight **Definitions**, **Formulas**, and **Important Dates/Names**.\n"
            )

        base_prompt += (
            "\n**Citation Rules:**\n"
            "- You MUST cite your source at the end of every major claim or paragraph using numbered brackets like **[1]**, **[2]** based on the Source number provided in the context.\n"
            "- If a claim comes from multiple sources, use **[1, 2]**.\n"
            "- Citations must be numeric only: **[n]** or **[n, m]**.\n"
            "- NEVER include year, filename, or the word 'Source' in citation text.\n"
            "- Do NOT make up information - stick strictly to the provided context.\n"
        )

        return base_prompt

    def _build_user_message(
        self,
        prompt: str,
        context: str,
        metadatas: List[Dict] = None,
        use_case: str = "qa",
    ) -> str:
        """Build user message with context and question."""
        sources = []
        if metadatas:
            for meta in metadatas:
                filename = meta.get("filename", "Unknown")
                clean_name = filename.replace(".pdf", "").replace(".docx", "").replace(".txt", "")
                if clean_name not in sources:
                    sources.append(clean_name)

        message = "**Available Sources (USE ONLY THESE):**\n"
        for source in sources[:5]:
            message += f"- {source}\n"

        quality_checklist = self._build_quality_checklist(prompt, use_case=use_case)

        message += f"\n**===== START OF CONTEXT (ANSWER ONLY FROM THIS) =====**\n\n{context}\n\n"
        message += "**===== END OF CONTEXT =====**\n\n"
        message += f"**Student's Question:** {prompt}\n\n"
        message += (
            "**Instructions:** Answer ONLY using the context between the markers above. "
            "If the context doesn't contain the answer, say you don't have that information. "
            "Cite sources in brackets."
        )
        if quality_checklist:
            message += f"\n{quality_checklist}"

        return message

    def _generate_nvidia(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        model_name: Optional[str] = None,
        thinking: bool = True,
    ) -> str:
        """Generate using NVIDIA NIM OpenAI-compatible API."""
        if not self.nvidia_client:
            raise RuntimeError("NVIDIA client is not configured")

        active_model = model_name or self.nvidia_model_name
        t0 = time.perf_counter()
        print(
            f"[LLM_TIMING] request_start model={active_model} thinking={bool(thinking)} "
            f"max_tokens={max_tokens} temperature={temperature} prompt_chars={len(user_message)}"
        )
        # Only deepseek models on Nvidia NIM support the thinking toggle param natively
        extra_body = {}
        if "deepseek" in active_model.lower():
            extra_body["chat_template_kwargs"] = {"thinking": bool(thinking)}

        response = self.nvidia_client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
            stream=False,
        )

        try:
            self._last_finish_reason = response.choices[0].finish_reason if response.choices else None
        except Exception:
            self._last_finish_reason = None

        if not response.choices:
            return ""

        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, list):
            # Some OpenAI-compatible SDKs can return segmented content.
            chunks = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                    if text:
                        chunks.append(str(text))
            content = "".join(chunks)

        print(
            f"[LLM_TIMING] request_done ms={(time.perf_counter() - t0) * 1000:.1f} "
            f"finish_reason={self._last_finish_reason} response_chars={len(content or '')}"
        )

        return content or ""

    def _generate_openai_compatible(
        self,
        client,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        model_name: str,
        provider: str,
    ) -> str:
        """Generate short outputs through an OpenAI-compatible provider."""
        t0 = time.perf_counter()
        print(
            f"[LLM_TIMING] request_start model={model_name} provider={provider} "
            f"max_tokens={max_tokens} temperature={temperature} prompt_chars={len(user_message)}"
        )
        request_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if provider == "groq":
            request_kwargs["reasoning_effort"] = "low"

        response = client.chat.completions.create(
            **request_kwargs,
        )
        if not response.choices:
            return ""

        content = getattr(response.choices[0].message, "content", "") or ""
        print(
            f"[LLM_TIMING] request_done ms={(time.perf_counter() - t0) * 1000:.1f} "
            f"provider={provider} response_chars={len(content)}"
        )
        return content

    def is_ready(self) -> bool:
        """Check if LLM is ready to generate."""
        return self.ready

    def get_provider(self) -> str:
        """Get current provider name."""
        return f"Gemini {self.gemini_model_name}"

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Simple wrapper for backend compatibility."""
        if not self.ready:
            return (
                "LLM not configured. Please add your Gemini API key.\n\n"
                "Get a key: https://aistudio.google.com/app/apikey"
            )

        try:
            if self.gemini_client:
                return self._generate_gemini(
                    system_prompt="You are a helpful AI assistant.",
                    user_message=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return self._generate_nvidia(
                system_prompt="You are a helpful AI assistant.",
                user_message=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=False,
            )
        except Exception as e:
            return f"Error generating response: {str(e)}"
