import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = "training_data/qwen_golden_dataset_scratch.jsonl"
DEFAULT_OUTPUT_FILE = "training_data/qwen_golden_dataset_scratch_sanitized.jsonl"
DEFAULT_REPORT_FILE = "training_data/qwen_golden_dataset_scratch_sanitized_report.json"

QWEN_BLOCK_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
    re.DOTALL,
)

ACADEMIC_HEADERS = (
    "### Conceptual Overview",
    "### Mathematical Framework",
    "### Comparative Analysis",
    "### Interpretation",
)

THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>|<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_qwen_chatml(text: str) -> tuple[str, str, str] | None:
    value = normalize_text(text)
    matches = list(QWEN_BLOCK_RE.finditer(value))
    if len(matches) != 3:
        return None

    roles = [m.group(1) for m in matches]
    if roles != ["system", "user", "assistant"]:
        return None

    system = normalize_text(matches[0].group(2))
    user = normalize_text(matches[1].group(2))
    assistant = normalize_text(matches[2].group(2))

    if not (system and user and assistant):
        return None

    return system, user, assistant


def build_qwen_chatml(system: str, user: str, assistant: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{normalize_text(system)}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{normalize_text(user)}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{normalize_text(assistant)}<|im_end|>"
    )


def recover_common_latex_escapes(text: str) -> str:
    # These control chars usually come from JSON-escaped LaTeX sequences like \b and \f.
    return text.replace("\x08", r"\b").replace("\x0c", r"\f")


def remove_private_use_chars(text: str) -> str:
    return re.sub(r"[\ue000-\uf8ff]", "", text)


def remove_disallowed_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)


def collapse_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def strip_meta_training_artifacts(text: str) -> str:
    cleaned = text
    cleaned = THINK_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("/no_think", "")
    cleaned = re.sub(
        r"you\s+must\s+strictly\s+output\s+valid\s+json[\s\S]*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^\s*only output the final answer in markdown\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*answer must be in markdown format only\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*keep the answer strictly within the markdown block\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def sanitize_segment(text: str) -> tuple[str, dict[str, int]]:
    before_private = sum(1 for ch in text if "\ue000" <= ch <= "\uf8ff")
    before_control = sum(1 for ch in text if ord(ch) < 32 and ch not in ("\n", "\r", "\t"))

    cleaned = normalize_text(text)
    cleaned = strip_meta_training_artifacts(cleaned)
    cleaned = recover_common_latex_escapes(cleaned)
    cleaned = remove_private_use_chars(cleaned)
    cleaned = remove_disallowed_control_chars(cleaned)
    cleaned = collapse_whitespace(cleaned)

    after_private = sum(1 for ch in cleaned if "\ue000" <= ch <= "\uf8ff")
    after_control = sum(1 for ch in cleaned if ord(ch) < 32 and ch not in ("\n", "\r", "\t"))

    metrics = {
        "private_removed": max(0, before_private - after_private),
        "control_removed": max(0, before_control - after_control),
        "private_remaining": after_private,
        "control_remaining": after_control,
    }
    return cleaned, metrics


def count_section_hits(text: str) -> int:
    return sum(1 for marker in ACADEMIC_HEADERS if marker in text)


def has_markdown_table(text: str) -> bool:
    for line in text.splitlines():
        if line.count("|") >= 2:
            return True
    return False


def has_unbalanced_dollars(text: str) -> bool:
    count = 0
    escaped = False
    for ch in text:
        if ch == "\\":
            escaped = not escaped
            continue
        if ch == "$" and not escaped:
            count += 1
        escaped = False
    return (count % 2) != 0


def evaluate_quality(
    system: str,
    user: str,
    assistant: str,
    *,
    min_user_chars: int,
    min_assistant_chars: int,
    min_section_hits: int,
    max_private_use_chars: int,
    max_control_chars: int,
    drop_unbalanced_math: bool,
) -> list[str]:
    reasons: list[str] = []

    if len(system) < 40:
        reasons.append("system_too_short")
    if len(user) < min_user_chars:
        reasons.append("user_too_short")
    if len(assistant) < min_assistant_chars:
        reasons.append("assistant_too_short")

    section_hits = count_section_hits(assistant)
    if section_hits < min_section_hits:
        reasons.append("insufficient_academic_sections")

    if "### Comparative Analysis" in assistant and not has_markdown_table(assistant):
        reasons.append("comparative_header_without_table")

    assistant_lower = assistant.lower()
    if "question:" in assistant_lower and "partial answer:" in assistant_lower:
        reasons.append("assistant_prompt_leakage")
    if "formatting policy for this answer" in assistant_lower:
        reasons.append("assistant_instruction_echo")
    if "you must strictly output valid json" in system.lower():
        reasons.append("system_json_schema_leak")
    if "/no_think" in system.lower() or "/no_think" in assistant_lower:
        reasons.append("no_think_marker_present")

    private_count = sum(1 for ch in (system + user + assistant) if "\ue000" <= ch <= "\uf8ff")
    control_count = sum(
        1
        for ch in (system + user + assistant)
        if ord(ch) < 32 and ch not in ("\n", "\r", "\t")
    )
    if private_count > max_private_use_chars:
        reasons.append("private_use_chars_remaining")
    if control_count > max_control_chars:
        reasons.append("control_chars_remaining")

    if "\ufffd" in (system + user + assistant):
        reasons.append("replacement_char_present")

    if drop_unbalanced_math and has_unbalanced_dollars(system + "\n" + user + "\n" + assistant):
        reasons.append("unbalanced_dollar_math")

    return reasons


def sanitize_dataset(
    input_file: Path,
    output_file: Path,
    report_file: Path,
    *,
    min_user_chars: int,
    min_assistant_chars: int,
    min_section_hits: int,
    max_private_use_chars: int,
    max_control_chars: int,
    dedupe: bool,
    drop_unbalanced_math: bool,
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    totals = Counter()
    drop_reasons = Counter()
    drop_reason_examples: dict[str, list[int]] = defaultdict(list)

    with input_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line_no, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue

            totals["lines_seen"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                totals["drop_invalid_json"] += 1
                drop_reasons["invalid_json"] += 1
                if len(drop_reason_examples["invalid_json"]) < 5:
                    drop_reason_examples["invalid_json"].append(line_no)
                continue

            if not isinstance(record, dict):
                totals["drop_non_object"] += 1
                drop_reasons["non_object_record"] += 1
                if len(drop_reason_examples["non_object_record"]) < 5:
                    drop_reason_examples["non_object_record"].append(line_no)
                continue

            raw_text = record.get("text")
            if not isinstance(raw_text, str):
                totals["drop_missing_text"] += 1
                drop_reasons["missing_text_field"] += 1
                if len(drop_reason_examples["missing_text_field"]) < 5:
                    drop_reason_examples["missing_text_field"].append(line_no)
                continue

            triplet = parse_qwen_chatml(raw_text)
            if triplet is None:
                totals["drop_invalid_chatml"] += 1
                drop_reasons["invalid_qwen_chatml"] += 1
                if len(drop_reason_examples["invalid_qwen_chatml"]) < 5:
                    drop_reason_examples["invalid_qwen_chatml"].append(line_no)
                continue

            system, user, assistant = triplet
            system_clean, metrics_system = sanitize_segment(system)
            user_clean, metrics_user = sanitize_segment(user)
            assistant_clean, metrics_assistant = sanitize_segment(assistant)

            totals["private_removed"] += (
                metrics_system["private_removed"]
                + metrics_user["private_removed"]
                + metrics_assistant["private_removed"]
            )
            totals["control_removed"] += (
                metrics_system["control_removed"]
                + metrics_user["control_removed"]
                + metrics_assistant["control_removed"]
            )

            reasons = evaluate_quality(
                system_clean,
                user_clean,
                assistant_clean,
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_section_hits=min_section_hits,
                max_private_use_chars=max_private_use_chars,
                max_control_chars=max_control_chars,
                drop_unbalanced_math=drop_unbalanced_math,
            )

            if reasons:
                totals["drop_quality"] += 1
                for reason in reasons:
                    drop_reasons[reason] += 1
                    if len(drop_reason_examples[reason]) < 5:
                        drop_reason_examples[reason].append(line_no)
                continue

            cleaned_text = build_qwen_chatml(system_clean, user_clean, assistant_clean)
            if dedupe:
                sample_hash = fingerprint(cleaned_text)
                if sample_hash in seen_hashes:
                    totals["drop_duplicate"] += 1
                    drop_reasons["duplicate_after_sanitize"] += 1
                    if len(drop_reason_examples["duplicate_after_sanitize"]) < 5:
                        drop_reason_examples["duplicate_after_sanitize"].append(line_no)
                    continue
                seen_hashes.add(sample_hash)

            dst.write(json.dumps({"text": cleaned_text}, ensure_ascii=False) + "\n")
            totals["kept"] += 1

    report = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "config": {
            "min_user_chars": min_user_chars,
            "min_assistant_chars": min_assistant_chars,
            "min_section_hits": min_section_hits,
            "max_private_use_chars": max_private_use_chars,
            "max_control_chars": max_control_chars,
            "dedupe": dedupe,
            "drop_unbalanced_math": drop_unbalanced_math,
        },
        "totals": dict(totals),
        "drop_reasons": dict(drop_reasons),
        "drop_reason_examples": {k: v for k, v in drop_reason_examples.items()},
    }

    with report_file.open("w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    print("Sanitization complete")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Report file: {report_file}")
    print(f"Rows kept: {totals.get('kept', 0)}")
    print(f"Rows dropped (quality): {totals.get('drop_quality', 0)}")
    print(f"Rows dropped (duplicates): {totals.get('drop_duplicate', 0)}")
    print(f"Rows dropped (invalid): {totals.get('drop_invalid_json', 0) + totals.get('drop_non_object', 0) + totals.get('drop_missing_text', 0) + totals.get('drop_invalid_chatml', 0)}")
    print(f"Control chars removed: {totals.get('control_removed', 0)}")
    print(f"Private-use chars removed: {totals.get('private_removed', 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanitize and quality-filter Qwen ChatML JSONL datasets for fine-tuning."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to source JSONL file.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Path to sanitized JSONL output file.")
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE, help="Path to JSON report output.")

    parser.add_argument("--min-user-chars", type=int, default=120, help="Minimum allowed user content length.")
    parser.add_argument(
        "--min-assistant-chars",
        type=int,
        default=320,
        help="Minimum allowed assistant content length.",
    )
    parser.add_argument(
        "--min-section-hits",
        type=int,
        default=3,
        help="Minimum count of expected academic section headers in assistant output.",
    )
    parser.add_argument(
        "--max-private-use-chars",
        type=int,
        default=0,
        help="Maximum private-use Unicode characters allowed after sanitization.",
    )
    parser.add_argument(
        "--max-control-chars",
        type=int,
        default=0,
        help="Maximum disallowed control characters allowed after sanitization.",
    )

    parser.add_argument("--no-dedupe", action="store_true", help="Disable duplicate removal after sanitization.")
    parser.add_argument(
        "--drop-unbalanced-math",
        action="store_true",
        help="Drop samples with odd counts of unescaped $ delimiters.",
    )

    # Jupyter/Colab injects extra args, so parse known args only.
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    sanitize_dataset(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        report_file=Path(args.report_file),
        min_user_chars=args.min_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        min_section_hits=args.min_section_hits,
        max_private_use_chars=args.max_private_use_chars,
        max_control_chars=args.max_control_chars,
        dedupe=not args.no_dedupe,
        drop_unbalanced_math=args.drop_unbalanced_math,
    )


if __name__ == "__main__":
    main()
