import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


OUTPUT_DIR = "training_data"
OUTPUT_FILENAME = "golden_dataset.jsonl"
TEACHER_MODEL = "gemini-3.1-flash-lite-preview"
TARGET_SAMPLES = 500
BACKEND_USE_CASE = "qa"
CHUNK_QUESTION_TEMPERATURE = 0.7
ANSWER_TEMPERATURE = 0.35
ANSWER_END_MARKER = "<<END_OF_ANSWER>>"

ACADEMIC_TOPICS = [
    "Linear Algebra for Data Science",
    "Probability Theory and Distributions",
    "Statistical Inference and Hypothesis Testing",
    "Calculus and Optimization for Machine Learning",
    "Information Theory and Entropy",
    "Numerical Methods and Approximation",
    "Descriptive Statistics and Exploratory Data Analysis",
    "Bayesian Statistics and Bayesian Inference",
    "Regression Analysis and Assumptions",
    "Time Series Analysis and Forecasting",
    "Experimental Design and A/B Testing",
    "Causal Inference and Counterfactuals",
    "Survival Analysis",
    "Dimensionality Reduction Techniques",
    "Anomaly and Outlier Detection",
    "Supervised Learning Fundamentals",
    "Unsupervised Learning and Clustering",
    "Decision Trees and Ensemble Methods",
    "Support Vector Machines",
    "Naive Bayes and Probabilistic Classifiers",
    "k-Nearest Neighbors and Instance-Based Learning",
    "Regularization Techniques (L1, L2, ElasticNet)",
    "Bias-Variance Tradeoff and Model Selection",
    "Cross-Validation and Hyperparameter Tuning",
    "Class Imbalance Handling and Resampling",
    "Feature Engineering and Selection",
    "Model Evaluation Metrics and Confusion Matrix",
    "Feedforward Neural Networks and Backpropagation",
    "Convolutional Neural Networks (CNNs)",
    "Recurrent Neural Networks and LSTMs",
    "Attention Mechanisms and Transformers",
    "Transfer Learning and Fine-Tuning",
    "Generative Adversarial Networks (GANs)",
    "Variational Autoencoders (VAEs)",
    "Diffusion Models",
    "Graph Neural Networks (GNNs)",
    "Optimization Algorithms (SGD, Adam, RMSProp)",
    "Batch Normalization and Dropout Regularization",
    "Loss Functions and Activation Functions",
    "Text Preprocessing and Tokenization",
    "Bag of Words and TF-IDF Representations",
    "Word Embeddings (Word2Vec, GloVe, FastText)",
    "Sequence-to-Sequence Models and Encoder-Decoder",
    "Named Entity Recognition and POS Tagging",
    "Sentiment Analysis and Text Classification",
    "Topic Modeling with LDA and NMF",
    "Machine Translation and BLEU Score",
    "Question Answering and Reading Comprehension",
    "Large Language Models and Prompt Engineering",
    "Retrieval-Augmented Generation (RAG)",
    "BERT, GPT, and Transformer Variants",
    "Image Classification and Object Detection",
    "Semantic Segmentation",
    "Image Augmentation and Data Preprocessing",
    "Object Tracking and Optical Flow",
    "Vision Transformers (ViT)",
    "Markov Decision Processes and Bellman Equations",
    "Q-Learning and Deep Q-Networks (DQN)",
    "Policy Gradient Methods",
    "Multi-Armed Bandit Problems",
    "Model Versioning and Experiment Tracking",
    "ML Pipelines and Workflow Orchestration",
    "Model Serving and REST API Deployment",
    "Containerization with Docker for ML",
    "Continuous Training and CI/CD for ML",
    "Model Monitoring and Data Drift Detection",
    "Feature Stores and Online vs Offline Features",
    "A/B Testing for ML Models in Production",
    "Relational Databases and SQL Fundamentals",
    "Database Normalization and Schema Design",
    "NoSQL Databases (MongoDB, Cassandra, Redis)",
    "Data Warehousing Concepts and Architecture",
    "ETL vs ELT Pipelines",
    "Apache Spark and Distributed Computing",
    "Apache Kafka and Stream Processing",
    "Apache Airflow and Pipeline Orchestration",
    "Data Lake Architecture and Delta Lake",
    "Medallion Architecture (Bronze, Silver, Gold)",
    "Batch Processing vs Stream Processing",
    "Data Partitioning and Indexing Strategies",
    "Data Serialization Formats (Parquet, Avro, ORC)",
    "Cloud Data Platforms (BigQuery, Redshift, Snowflake)",
    "dbt and Data Transformation in the Modern Stack",
    "Data Quality Dimensions and Validation",
    "Data Lineage and Metadata Management",
    "Data Governance Frameworks and Compliance",
    "Data Cataloging and Discovery",
    "GDPR, HIPAA, and Data Privacy Principles",
    "MapReduce Programming Model",
    "Hadoop Ecosystem Overview",
    "CAP Theorem and Distributed Consistency",
    "Sharding, Replication, and Fault Tolerance",
    "Data Mesh Architecture",
    "Principles of Data Visualization",
    "Dashboarding and Business Intelligence Tools",
    "Storytelling with Data",
    "Statistical Charting (Box Plots, Violin Plots, Histograms)",
    "Fairness, Bias, and Algorithmic Accountability",
    "Model Explainability (SHAP, LIME)",
    "Differential Privacy in Machine Learning",
    "AI Safety and Alignment Principles",
]


def ensure_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


def load_api_key_from_env_file() -> None:
    env_path = Path(".env")

    if load_dotenv is not None:
        # Always prefer .env for this script to avoid stale shell/session keys.
        load_dotenv(dotenv_path=env_path, override=True)
        return

    # Fallback parser so .env still works even without python-dotenv installed.
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == "GEMINI_API_KEY" and value:
            os.environ["GEMINI_API_KEY"] = value
            return


def build_gemini_client() -> genai.Client:
    load_api_key_from_env_file()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to .env or export it before running this script."
        )

    print(f"Using GEMINI_API_KEY fingerprint: {mask_api_key(api_key)}")

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def build_generation_prompt(topic: str, sample_index: int) -> str:
    return f"""
You are designing synthetic training data for a Universal Expert Tutor model.
For the academic topic below, complete BOTH tasks in one response.

TOPIC:
{topic}

SAMPLE ID:
{sample_index}

TASK 1 - CHUNK:
Generate a realistic, highly complex textbook chunk of about 200 words for this topic.
The chunk must naturally include data comparisons and mathematical/statistical formulas.

TASK 2 - QUESTION:
Generate one complex student question based strictly on the chunk.

OUTPUT FORMAT:
You must output strictly in valid JSON format like this:
{{
  "chunk": "The generated textbook chunk",
  "question": "The generated question"
}}
""".strip()


def build_backend_context(topic: str, chunk: str) -> str:
    clean_topic = topic.replace("\n", " ").strip()
    return f"Source [1] [YEAR: 2025] ({clean_topic} synthetic textbook):\n{chunk.strip()}\n"


def build_backend_metadata(topic: str) -> list[dict[str, str]]:
    safe_topic = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") or "synthetic_topic"
    return [{"filename": f"{safe_topic}.txt"}]


def infer_academic_depth_mode(prompt: str, use_case: str) -> bool:
    if use_case not in {"qa", "explanation", "notes"}:
        return False

    lower = (prompt or "").lower()
    triggers = (
        "evaluate",
        "evaluation",
        "compare",
        "contrast",
        "analyze",
        "analysis",
        "justify",
        "derive",
        "mathematical",
        "formula",
        "metric",
        "metrics",
        "bleu",
        "rouge",
        "meteor",
        "bertscore",
        "nlp",
        "transformer",
        "limitations",
        "assumptions",
        "deep learning",
        "exam",
        "academic",
    )
    return any(t in lower for t in triggers)


def build_quality_checklist(prompt: str, use_case: str) -> str:
    if not infer_academic_depth_mode(prompt, use_case):
        return ""

    checklist = (
        "\n**Academic Completeness Checklist (apply when evidence exists in context):**\n"
        "1. Include core mechanism details, not just definitions.\n"
        "2. Include at least one relevant equation, formula component, or scoring term when present.\n"
        "3. State known limitations/failure cases and why they matter.\n"
        "4. Mention important variants/versions when supported by context.\n"
        "5. Briefly contrast with modern alternatives if context includes them.\n"
        "6. Synthesize evidence from multiple source snippets when available.\n"
        "7. Keep structure clear: definition -> mechanics -> limitations -> modern view -> takeaway.\n"
        "Note: If a checklist item is not supported by the provided context, explicitly say that.\n"
    )

    lower = (prompt or "").lower()
    if "bleu" in lower or "rouge" in lower or "summarization" in lower or "translation" in lower:
        checklist += (
            "\n**NLP Metric Focus (for BLEU/ROUGE-style questions):**\n"
            "- For BLEU: mention **Brevity Penalty (BP)** when present in context.\n"
            "- For ROUGE: include precision/recall/F1 behavior when supported.\n"
            "- Mention **ROUGE-N** and **ROUGE-L (LCS)** if found in context.\n"
            "- If context includes **BERTScore**, include it as modern practice.\n"
            "- If missing in context, state that limitation explicitly instead of guessing.\n"
        )

    return checklist


def build_backend_system_prompt(use_case: str) -> str:
    base_prompt = (
        "You are the expert generation engine for NotebookPRO, an advanced academic study system for Master-level data science and NLP coursework. "
        "Your goal is to synthesize retrieved textbook excerpts into rigorous, mathematically sound, comprehensive study material.\n\n"
        "STRICT GROUNDING: You MUST ONLY use information present inside the provided context. "
        "Do NOT use outside knowledge and do NOT hallucinate. Every technical claim must be traceable to context.\n\n"
        "TEMPORAL ARBITRATION & SYNTHESIS: Context chunks are labeled as Source [i] [YEAR: YYYY] (filename). "
        "Prioritize the most recent year as modern standard/state-of-the-art, and frame older sources as foundational/historical context.\n\n"
        "ACADEMIC RIGOR: Avoid superficial summaries. "
        "Explain mechanisms, formulas, penalties, architectural details, limitations, and failure modes when present.\n\n"
        "REQUIRED OUTPUT STRUCTURE:\n"
        "- Use clear markdown headings with ### for major sections.\n"
        "- Use bulleted breakdowns with bold key terms (for example, * **Mechanism:** ...).\n"
        "- For tabular values, use valid markdown tables: header row, separator row (---), then body rows.\n"
        "- Keep explanations dense, structured, and exam-ready.\n"
        "- Do not add UI extras like Suggested Follow-Up Questions inside the answer body.\n"
    )

    if use_case == "explanation":
        base_prompt += (
            "\n**Your task:** Explain the concept in a clear, step-by-step manner suitable for students.\n"
            "1. Start with a concise, one-sentence definition.\n"
            "2. Break down the core mechanics using bullet points.\n"
            "3. Provide an example only if found in the text.\n"
            "4. Add a Key Takeaway at the end.\n"
        )
    elif use_case == "summary":
        base_prompt += (
            "\n**Your task:** Create a highly structured summary.\n"
            "- Start with a brief high-level overview (2 sentences max).\n"
            "- Use ### Key Themes and list concise, dense points.\n"
        )
    elif use_case == "qa":
        base_prompt += (
            "\n**Your task:** Answer the question directly and comprehensively.\n"
            "- Provide the direct answer immediately in the first sentence.\n"
            "- Build a deep explanation using multiple complementary points from context.\n"
            "- When question asks why, include mechanisms, risks, impacts, and implementation implications if present.\n"
            "- Use numbered lists or bullet points for supporting details when useful.\n"
            "- Use **bold** for key facts, numbers, and formulas.\n"
            "- Aim for substantial depth (typically 250-500 words) unless context is very limited.\n"
        )
    elif use_case == "notes":
        base_prompt += (
            "\n**Your task:** Create comprehensive, structured study notes.\n"
            "- Use clear section headers (###).\n"
            "- Organize information hierarchically with bullets.\n"
            "- Explicitly highlight **Definitions**, **Formulas**, and **Important Dates/Names**.\n"
        )

    base_prompt += (
        "\n**Citation Rules:**\n"
        "- Cite every major claim or paragraph using **[1]**, **[2]** etc based on Source number.\n"
        "- If claim uses multiple sources, use **[1, 2]**.\n"
        "- Citations must be numeric only: **[n]** or **[n, m]**.\n"
        "- NEVER include year, filename, or the word Source in citation text.\n"
        "- Do NOT make up information.\n"
        f"- End your final response with the exact marker {ANSWER_END_MARKER} on its own line.\n"
    )

    return base_prompt


def build_backend_user_message(
    prompt: str,
    context: str,
    metadatas: list[dict[str, str]] | None,
    use_case: str,
) -> str:
    sources: list[str] = []
    if metadatas:
        for meta in metadatas:
            filename = str(meta.get("filename", "Unknown"))
            clean_name = filename.replace(".pdf", "").replace(".docx", "").replace(".txt", "")
            if clean_name not in sources:
                sources.append(clean_name)

    message = "**Available Sources (USE ONLY THESE):**\n"
    for source in sources[:5]:
        message += f"- {source}\n"

    quality_checklist = build_quality_checklist(prompt, use_case=use_case)

    message += f"\n**===== START OF CONTEXT (ANSWER ONLY FROM THIS) =====**\n\n{context}\n\n"
    message += "**===== END OF CONTEXT =====**\n\n"
    message += f"**Student's Question:** {prompt}\n\n"
    message += (
        "**Instructions:** Answer ONLY using the context between the markers above. "
        "If the context does not contain the answer, say you do not have that information. "
        "Cite sources in brackets."
    )
    if quality_checklist:
        message += f"\n{quality_checklist}"
    return message


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            body = "\n".join(lines[1:-1]).strip()
            return body
    return text


def sanitize_model_json_text(text: str) -> str:
    text = strip_markdown_code_fence(text)
    text = text.replace("\ufeff", "").replace("\x00", "")
    return text


def parse_response_json(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None

    normalized = sanitize_model_json_text(raw_text)
    candidates = [normalized]
    extracted = extract_first_json_object(normalized)
    if extracted and extracted != normalized:
        candidates.append(extracted)

    for candidate in candidates:
        for attempt_fix in range(2):
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
                return None
            except json.JSONDecodeError:
                if attempt_fix == 0:
                    candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
                    continue

    return None


def create_gemini_completion(
    client: genai.Client,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    response_json: bool = False,
    stop_sequences: list[str] | None = None,
) -> str:
    cfg: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if response_json:
        cfg["response_mime_type"] = "application/json"
    if stop_sequences:
        cfg["stop_sequences"] = stop_sequences

    response = client.models.generate_content(
        model=TEACHER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**cfg),
    )
    return response.text or ""


def repair_json_payload(client: genai.Client, malformed_text: str) -> dict[str, Any] | None:
    repair_prompt = f"""
Convert the following model output into ONE valid JSON object.
Rules:
- Output JSON only (no markdown fences, no commentary).
- Must include keys: chunk, question.

MODEL OUTPUT TO REPAIR:
{malformed_text}
""".strip()

    try:
        repaired_text = create_gemini_completion(
            client=client,
            prompt=repair_prompt,
            temperature=0.3,
            max_output_tokens=2048,
            response_json=True,
        )
        return parse_response_json(repaired_text)
    except Exception:
        return None


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_teacher_answer(answer: str) -> str:
    if not answer:
        return ""

    cleaned = answer
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>", "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"^\s*only output the final answer in markdown\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*answer must be in markdown format only\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*keep the answer strictly within the markdown block\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    lines = []
    echo_re = re.compile(
        r"do not use markdown|formatting policy|return continuation text only|preserve the same hybrid format",
        re.IGNORECASE,
    )
    for line in cleaned.splitlines():
        if echo_re.search(line):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def normalize_citation_style(text: str) -> str:
    if not text:
        return ""

    normalized = re.sub(r"\*\*\[(\d+(?:,\s*\d+)*)\]\*\*", r"[\1]", text)
    normalized = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r"**[\1]**", normalized)
    return normalized


def sanitize_final_response(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace(ANSWER_END_MARKER, "").strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>", "", cleaned, flags=re.IGNORECASE)

    cleaned_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        if re.search(r"(?:\d+\.){8,}\d*", line):
            continue

        compact = re.sub(r"\s+", "", line)
        if len(compact) >= 40 and re.fullmatch(r"[\d\W_]+", compact):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = normalize_citation_style(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def strip_academic_note_section(text: str) -> str:
    if not text:
        return ""
    marker = "### Academic Note:"
    idx = text.find(marker)
    if idx == -1:
        return text.strip()
    return text[:idx].strip()


def has_backend_format_signals(answer: str) -> bool:
    if not answer:
        return False
    has_heading = bool(re.search(r"^###\s+", answer, re.MULTILINE))
    has_citation = bool(re.search(r"\[(?:\d+\s*(?:,\s*\d+)*)\]", answer))
    return has_heading and has_citation


def looks_repetitive_answer(answer: str) -> bool:
    if not answer:
        return True

    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if len(sentences) < 6:
        return False

    counts: dict[str, int] = {}
    for sentence in sentences:
        counts[sentence] = counts.get(sentence, 0) + 1

    return max(counts.values()) >= 3


def generate_backend_style_answer(
    client: genai.Client,
    topic: str,
    question: str,
    chunk: str,
    max_transient_retries: int = 1,
) -> str | None:
    context = build_backend_context(topic=topic, chunk=chunk)
    metadata = build_backend_metadata(topic=topic)

    system_prompt = build_backend_system_prompt(BACKEND_USE_CASE)
    user_message = build_backend_user_message(
        prompt=question,
        context=context,
        metadatas=metadata,
        use_case=BACKEND_USE_CASE,
    )

    full_prompt = f"{system_prompt}\n\n{user_message}"

    for attempt in range(max_transient_retries + 1):
        try:
            answer_text = create_gemini_completion(
                client=client,
                prompt=full_prompt,
                temperature=ANSWER_TEMPERATURE,
                max_output_tokens=1600,
                response_json=False,
                stop_sequences=[ANSWER_END_MARKER],
            )

            cleaned_answer = clean_teacher_answer(answer_text)
            cleaned_answer = sanitize_final_response(cleaned_answer)
            cleaned_answer = strip_academic_note_section(cleaned_answer)

            if len(cleaned_answer) < 120:
                print("Warning: backend-style answer too short. Skipping sample.")
                return None
            if looks_repetitive_answer(cleaned_answer):
                print("Warning: repetitive backend-style answer detected. Skipping sample.")
                return None
            if not has_backend_format_signals(cleaned_answer):
                print("Warning: answer missing backend format signals (### headings + citations). Skipping sample.")
                return None

            return cleaned_answer

        except Exception as exc:
            error_msg = str(exc).lower()
            is_transient = (
                "503" in error_msg
                or "high demand" in error_msg
                or "timeout" in error_msg
                or "timed out" in error_msg
            )
            if is_transient:
                print("Warning: transient model error while generating answer. Sleeping for 10 seconds...")
                time.sleep(10)
                if attempt < max_transient_retries:
                    print("Retrying backend-style answer generation...")
                    continue
            print(f"Failed to generate backend-style answer: {exc}")
            return None

    return None


def load_existing_dataset_state(output_path: Path) -> tuple[set[str], int]:
    if not output_path.exists():
        return set(), 0

    seen_fingerprints: set[str] = set()
    processed_slots = 0

    for raw_line in output_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        processed_slots += 1
        try:
            payload = json.loads(line)
            text = payload.get("text") if isinstance(payload, dict) else None
            if isinstance(text, str) and text:
                seen_fingerprints.add(text_fingerprint(text))
        except json.JSONDecodeError:
            continue

    return seen_fingerprints, processed_slots


def generate_triplet(
    client: genai.Client,
    topic: str,
    sample_index: int,
    max_transient_retries: int = 1,
) -> dict[str, str] | None:
    prompt = build_generation_prompt(topic=topic, sample_index=sample_index)

    for attempt in range(max_transient_retries + 1):
        try:
            response_text = create_gemini_completion(
                client=client,
                prompt=prompt,
                temperature=CHUNK_QUESTION_TEMPERATURE,
                max_output_tokens=2048,
                response_json=True,
            )

            payload = parse_response_json(response_text)
            if payload is None:
                if response_text.strip():
                    repaired_payload = repair_json_payload(client, response_text)
                    if repaired_payload is not None:
                        payload = repaired_payload
                        print("Warning: malformed JSON response recovered via repair pass.")

            if payload is None:
                print("Warning: malformed JSON response. Retrying sample...")
                if attempt < max_transient_retries:
                    continue
                print("Skipping sample after malformed JSON response.")
                return None

            chunk = payload.get("chunk")
            question = payload.get("question")

            if isinstance(chunk, str) and isinstance(question, str):
                chunk_clean = chunk.strip()
                question_clean = question.strip()

                if len(chunk_clean) < 160 or len(question_clean) < 20:
                    print("Warning: chunk/question quality too low. Skipping sample.")
                    return None

                answer = generate_backend_style_answer(
                    client=client,
                    topic=topic,
                    question=question_clean,
                    chunk=chunk_clean,
                )
                if not answer:
                    return None

                return {
                    "chunk": chunk_clean,
                    "question": question_clean,
                    "answer": answer,
                }

            print("Warning: response JSON is missing chunk/question. Skipping sample.")
            return None

        except Exception as exc:
            error_msg = str(exc).lower()
            is_transient = (
                "503" in error_msg
                or "high demand" in error_msg
                or "timeout" in error_msg
                or "timed out" in error_msg
            )

            if is_transient:
                print("Warning: transient Gemini error (503/timeout). Sleeping for 10 seconds...")
                time.sleep(10)
                if attempt < max_transient_retries:
                    print("Retrying sample after transient failure...")
                    continue
                print("Skipping sample after transient failure.")
                return None

            print(f"Failed to generate triplet: {exc}")
            return None

    return None


def format_for_unsloth(topic: str, chunk: str, question: str, answer: str) -> dict[str, str]:
    context = build_backend_context(topic=topic, chunk=chunk)
    metadata = build_backend_metadata(topic=topic)
    system_prompt = build_backend_system_prompt(BACKEND_USE_CASE)
    user_message = build_backend_user_message(
        prompt=question,
        context=context,
        metadatas=metadata,
        use_case=BACKEND_USE_CASE,
    )

    chatml_string = (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{answer}<|im_end|>"
    )
    return {"text": chatml_string}


def create_dataset(
    output_dir: str,
    output_filename: str,
) -> None:
    total_samples = TARGET_SAMPLES
    estimated_rounds = math.ceil(total_samples / len(ACADEMIC_TOPICS))
    print("Starting universal tutor dataset generation...")
    print(f"Topics: {len(ACADEMIC_TOPICS)}")
    print(f"Approx rounds: {estimated_rounds}")
    print(f"Target samples: {total_samples}")

    client = build_gemini_client()
    output_path = ensure_output_dir(output_dir) / output_filename

    seen_fingerprints, processed_slots = load_existing_dataset_state(output_path)
    if processed_slots > 0:
        print(
            f"Resuming from existing file: {processed_slots} slot(s) already present, "
            f"{len(seen_fingerprints)} unique sample(s)."
        )

    if processed_slots >= total_samples:
        print(
            "Target slot count already reached in output file. "
            f"Delete the file if you want a fresh {total_samples}-sample run."
        )
        return

    generated_count = 0
    duplicate_regen_limit = 3
    with output_path.open("a", encoding="utf-8") as f:
        for slot_index in range(processed_slots, total_samples):
            round_index = slot_index // len(ACADEMIC_TOPICS)
            topic_index = slot_index % len(ACADEMIC_TOPICS)
            topic = ACADEMIC_TOPICS[topic_index]

            print(
                f"Processing sample {slot_index + 1}/{total_samples} "
                f"(round {round_index + 1}/{estimated_rounds}, topic: {topic})..."
            )

            wrote_unique = False
            for regen_attempt in range(duplicate_regen_limit + 1):
                triplet = generate_triplet(
                    client=client,
                    topic=topic,
                    sample_index=slot_index + 1,
                )

                if not (triplet and "chunk" in triplet and "question" in triplet and "answer" in triplet):
                    break

                line = format_for_unsloth(
                    topic=topic,
                    chunk=triplet["chunk"],
                    question=triplet["question"],
                    answer=triplet["answer"],
                )
                fingerprint = text_fingerprint(line["text"])

                if fingerprint in seen_fingerprints:
                    if regen_attempt < duplicate_regen_limit:
                        print("Duplicate sample detected. Regenerating this slot...")
                        continue
                    print("Duplicate sample persisted after retries. Skipping slot.")
                    break

                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                seen_fingerprints.add(fingerprint)
                generated_count += 1
                wrote_unique = True

                time.sleep(1)
                break

            if not wrote_unique:
                print("Skipping sample due to invalid, failed, or duplicate generation.")

    print(f"Completed. Added {generated_count} new unique sample(s) this run.")
    print(f"Total unique sample fingerprints in file: {len(seen_fingerprints)}")
    print(f"Output file: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Universal Expert Tutor dataset using Gemini."
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory where dataset files are written.",
    )
    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILENAME,
        help="Dataset filename (JSONL).",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    create_dataset(
        output_dir=args.output_dir,
        output_filename=args.output_file,
    )


if __name__ == "__main__":
    main()
