"""Prompt construction with an exact, maximal shared prefix.

Critical invariant (spec section 4): the query MUST come after the shared
content so that requests in the same prefix group share an exact token-level
prefix up to the query boundary.

Layout of the rendered chat prompt::

    <|im_start|>system
    {FIXED_INSTRUCTION}<|im_end|>
    <|im_start|>user
    <source>
    {SHARED_CONTENT}
    </source>

    <query>
    {QUERY}</query><|im_end|>
    <|im_start|>assistant
    ... (generation prompt, constant across the group)

Everything up to and including "<query>\n" is identical across a prefix group
(the fixed instruction + shared content are constant), so the token-level
shared prefix boundary lands right before {QUERY}.
"""

from __future__ import annotations

from typing import Any


FIXED_INSTRUCTION = (
    "You are given a source document.\n"
    "Answer the query using only the source document.\n"
    "Provide a concise and self-contained summary."
)

# The user turn. The shared content is fully rendered before the query.
# The query marker "<query>\n" terminates the shared region; the query text
# follows immediately.
_USER_TEMPLATE = "<source>\n{shared_content}\n</source>\n\n<query>\n{query}</query>"


def build_messages(shared_content: str, query: str) -> list[dict[str, str]]:
    """Return chat messages with the fixed instruction as the system turn."""
    return [
        {"role": "system", "content": FIXED_INSTRUCTION},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                shared_content=shared_content, query=query
            ),
        },
    ]


def render_prompt(
    tokenizer: Any,
    shared_content: str,
    query: str,
    enable_thinking: bool = False,
) -> str:
    """Render the full prompt string via the model chat template."""
    messages = build_messages(shared_content, query)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def normalize_ids(ids: Any) -> list[int]:
    """Normalize whatever apply_chat_template(tokenize=True) returns to a flat
    list[int]. Handles: list[int], nested list, torch/np tensor, HF
    BatchEncoding/dict, and tokenizers.Encoding (transformers 5.x)."""
    # tokenizers.Encoding exposes .ids
    if hasattr(ids, "ids") and not isinstance(ids, dict):
        return list(ids.ids)
    # BatchEncoding / dict with input_ids
    if isinstance(ids, dict) or hasattr(ids, "get"):
        try:
            inner = ids["input_ids"]
            return normalize_ids(inner)
        except (KeyError, TypeError):
            pass
    # torch / numpy tensor
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    # unwrap a single batch dim
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


QUERY_MARKER = "<query>\n"


def encode_prompt(
    tokenizer: Any,
    shared_content: str,
    query: str,
    enable_thinking: bool = False,
) -> list[int]:
    """Return the tokenized prompt (input_ids) for a request as list[int].

    We tokenize the *rendered chat string* with add_special_tokens=False rather
    than apply_chat_template(tokenize=True). Inline control tokens like
    ``<|im_start|>`` are still recognized as single special tokens, and this
    path lets validation reuse the identical tokenization with offset mapping.
    """
    text = render_prompt(tokenizer, shared_content, query, enable_thinking)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_prompt_with_offsets(
    tokenizer: Any,
    shared_content: str,
    query: str,
    enable_thinking: bool = False,
) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Return (rendered_text, input_ids, char offset per token)."""
    text = render_prompt(tokenizer, shared_content, query, enable_thinking)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc["input_ids"], enc["offset_mapping"]
