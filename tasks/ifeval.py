"""
IFEval: Instruction Following Evaluation dataset.
https://huggingface.co/datasets/google/IFEval
"""

import json
import re
from datasets import load_dataset

from .common import Task


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_BOLD_RE = re.compile(r"\*\*[^*\n]+?\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)")
_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")
_TITLE_RE = re.compile(r"<<[^<>]+>>")
_BULLET_RE = re.compile(r"^\s*\* ", re.MULTILINE)

_LANG_RANGES = {
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi)
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "ta": (0x0B80, 0x0BFF),  # Tamil
}


def _count_words(text):
    return len(_WORD_RE.findall(text))


def _count_sentences(text):
    parts = re.split(r"[.!?]+", text)
    return sum(1 for part in parts if part.strip())


def _count_paragraphs(text):
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return sum(1 for p in paragraphs if p.strip())


def _split_paragraphs(text):
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _count_highlighted_sections(text):
    bold = len(_BOLD_RE.findall(text))
    italic = len(_ITALIC_RE.findall(text))
    return bold + italic


def _count_bullets(text):
    return len(_BULLET_RE.findall(text))


def _count_placeholders(text):
    return len(_PLACEHOLDER_RE.findall(text))


def _count_letter(text, letter):
    if not letter:
        return 0
    if letter.isalpha():
        target = letter.casefold()
        return sum(1 for c in text.casefold() if c == target)
    return text.count(letter)


def _count_keyword(text, keyword):
    if not keyword:
        return 0
    text_cf = text.casefold()
    keyword_cf = keyword.casefold()
    if re.search(r"\s", keyword_cf):
        return text_cf.count(keyword_cf)
    pattern = re.compile(rf"\b{re.escape(keyword_cf)}\b", re.IGNORECASE)
    return len(pattern.findall(text))


def _check_relation(count, relation, threshold):
    if threshold is None:
        return False
    if relation in (None, "", "at least"):
        return count >= threshold
    if relation in ("less than", "fewer than"):
        return count < threshold
    if relation in ("at most", "no more than"):
        return count <= threshold
    if relation in ("more than",):
        return count > threshold
    if relation in ("equal", "exactly"):
        return count == threshold
    return False


def _check_json_only(text):
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    try:
        decoder = json.JSONDecoder()
        _, idx = decoder.raw_decode(stripped)
    except Exception:
        return False
    return stripped[idx:].strip() == ""


def _check_language(text, language):
    if not language:
        return False
    if language == "en":
        return True
    lang_range = _LANG_RANGES.get(language)
    if lang_range is None:
        return False
    total_letters = 0
    script_letters = 0
    latin_letters = 0
    lo, hi = lang_range
    for ch in text:
        if not ch.isalpha():
            continue
        total_letters += 1
        code = ord(ch)
        if lo <= code <= hi:
            script_letters += 1
        elif "A" <= ch <= "Z" or "a" <= ch <= "z":
            latin_letters += 1
    if total_letters == 0:
        return False
    if latin_letters > 0:
        return False
    return script_letters > 0


def _check_end_phrase(text, end_phrase):
    if not end_phrase:
        return False
    stripped = text.strip()
    if stripped.endswith(end_phrase):
        return True
    if stripped.endswith(end_phrase + '"'):
        return True
    if stripped.endswith(end_phrase + "'"):
        return True
    return False


def _check_repeat_prompt(text, prompt_to_repeat):
    if not prompt_to_repeat:
        return False
    if not text.startswith(prompt_to_repeat):
        return False
    remainder = text[len(prompt_to_repeat):]
    return bool(remainder.strip())


def _check_two_responses(text):
    parts = text.split("******")
    if len(parts) != 2:
        return False
    return all(part.strip() for part in parts)


def _check_multiple_sections(text, section_spliter, num_sections):
    if not section_spliter or num_sections is None:
        return False
    pattern = re.compile(rf"\b{re.escape(section_spliter)}\b\s*\S+", re.IGNORECASE)
    return len(pattern.findall(text)) == num_sections


def _check_postscript(text, marker):
    if not marker:
        return False
    pattern = re.compile(rf"(?im)^\s*{re.escape(marker)}")
    return bool(pattern.search(text))


def _check_nth_paragraph_first_word(text, first_word, num_paragraphs, nth_paragraph):
    if not first_word or not num_paragraphs or not nth_paragraph:
        return False
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) != num_paragraphs:
        return False
    idx = nth_paragraph - 1
    if idx < 0 or idx >= len(paragraphs):
        return False
    words = _WORD_RE.findall(paragraphs[idx])
    if not words:
        return False
    return words[0].casefold() == first_word.casefold()


def _check_capital_word_frequency(text, relation, frequency):
    if frequency is None:
        return False
    words = re.findall(r"\b[A-Z]+\b", text)
    count = len(words)
    return _check_relation(count, relation, frequency)


def _check_english_lowercase(text):
    return bool(re.search(r"[a-z]", text)) and not re.search(r"[A-Z]", text)


def _check_english_capital(text):
    return bool(re.search(r"[A-Z]", text)) and not re.search(r"[a-z]", text)


def _check_keywords_exist(text, keywords):
    if not keywords:
        return False
    return all(_count_keyword(text, kw) > 0 for kw in keywords)


def _check_keywords_forbidden(text, forbidden_words):
    if not forbidden_words:
        return True
    return all(_count_keyword(text, kw) == 0 for kw in forbidden_words)


def _check_keyword_frequency(text, relation, keyword, frequency):
    if not keyword or frequency is None:
        return False
    count = _count_keyword(text, keyword)
    return _check_relation(count, relation, frequency)


def _evaluate_instruction(instruction_id, kwargs, text, prompt):
    if instruction_id == "punctuation:no_comma":
        return "," not in text
    if instruction_id == "detectable_format:number_highlighted_sections":
        num = kwargs.get("num_highlights")
        return _count_highlighted_sections(text) == num
    if instruction_id == "length_constraints:number_words":
        relation = kwargs.get("relation")
        num_words = kwargs.get("num_words")
        return _check_relation(_count_words(text), relation, num_words)
    if instruction_id == "detectable_content:number_placeholders":
        num = kwargs.get("num_placeholders")
        return _count_placeholders(text) >= num if num is not None else False
    if instruction_id == "combination:repeat_prompt":
        prompt_to_repeat = kwargs.get("prompt_to_repeat") or prompt
        return _check_repeat_prompt(text, prompt_to_repeat)
    if instruction_id == "detectable_format:title":
        return bool(_TITLE_RE.search(text))
    if instruction_id == "change_case:english_lowercase":
        return _check_english_lowercase(text)
    if instruction_id == "detectable_format:number_bullet_lists":
        num = kwargs.get("num_bullets")
        return _count_bullets(text) == num
    if instruction_id == "change_case:english_capital":
        return _check_english_capital(text)
    if instruction_id == "detectable_format:multiple_sections":
        section_spliter = kwargs.get("section_spliter")
        num_sections = kwargs.get("num_sections")
        return _check_multiple_sections(text, section_spliter, num_sections)
    if instruction_id == "change_case:capital_word_frequency":
        relation = kwargs.get("capital_relation")
        frequency = kwargs.get("capital_frequency")
        return _check_capital_word_frequency(text, relation, frequency)
    if instruction_id == "startend:quotation":
        stripped = text.strip()
        return len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"'
    if instruction_id == "keywords:existence":
        return _check_keywords_exist(text, kwargs.get("keywords") or [])
    if instruction_id == "length_constraints:number_paragraphs":
        num = kwargs.get("num_paragraphs")
        return _count_paragraphs(text) == num
    if instruction_id == "detectable_format:json_format":
        return _check_json_only(text)
    if instruction_id == "combination:two_responses":
        return _check_two_responses(text)
    if instruction_id == "language:response_language":
        return _check_language(text, kwargs.get("language"))
    if instruction_id == "keywords:letter_frequency":
        relation = kwargs.get("let_relation")
        letter = kwargs.get("letter")
        frequency = kwargs.get("let_frequency")
        return _check_relation(_count_letter(text, letter), relation, frequency)
    if instruction_id == "startend:end_checker":
        return _check_end_phrase(text, kwargs.get("end_phrase"))
    if instruction_id == "keywords:forbidden_words":
        return _check_keywords_forbidden(text, kwargs.get("forbidden_words") or [])
    if instruction_id == "keywords:frequency":
        relation = kwargs.get("relation")
        keyword = kwargs.get("keyword")
        frequency = kwargs.get("frequency")
        return _check_keyword_frequency(text, relation, keyword, frequency)
    if instruction_id == "length_constraints:number_sentences":
        relation = kwargs.get("relation")
        num_sentences = kwargs.get("num_sentences")
        return _check_relation(_count_sentences(text), relation, num_sentences)
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        return _check_nth_paragraph_first_word(
            text,
            kwargs.get("first_word"),
            kwargs.get("num_paragraphs"),
            kwargs.get("nth_paragraph"),
        )
    if instruction_id == "detectable_content:postscript":
        return _check_postscript(text, kwargs.get("postscript_marker"))
    return False


class IFEval(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("google/IFEval", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ],
            "instruction_id_list": row.get("instruction_id_list") or [],
            "kwargs": row.get("kwargs") or [],
            "key": row.get("key"),
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        instruction_ids = conversation.get("instruction_id_list") or []
        kwargs_list = conversation.get("kwargs") or []
        prompt = conversation["messages"][0]["content"] if conversation.get("messages") else ""
        for idx, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[idx] if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict) else {}
            if not _evaluate_instruction(instruction_id, kwargs, assistant_response, prompt):
                return False
        return True
