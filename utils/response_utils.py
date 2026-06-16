"""Response preprocessing shared by all regression graders.

`strip_reasoning` handles three response shapes produced by eval_sageeval.py /
local_inference.py / vllm_inference.py via renderer.parse_response:

- **str** (Llama, Qwen3 with thinking disabled, or any non-parsed model output).
  May contain `<think>...</think>` blocks (Qwen3 in thinking mode). We drop
  everything up to and including the LAST `</think>`. If `<think>` opened but
  never closed the response was truncated before producing a final answer; we
  return "" so the grader records no_answer (matching lm-eval-harness #3240).

- **list[dict]** (gpt-oss Harmony, Qwen3 when the renderer splits into
  ContentPart objects). Each block is `{"type": "text"|"thinking", ...}`.
  We keep only `{"type": "text"}` blocks.

- **str with Harmony channel-marker residue** (gpt-oss when parse_response
  fell back to raw decode — no `<|return|>` was seen, typically truncation).
  These strings start with the literal word "analysis" fused to content (e.g.
  "analysisWe need to..."), which is `<|channel|>analysis<|message|>...` with
  special tokens stripped by `tokenizer.decode`. If `assistantfinal` is present
  we take everything after it (final channel). If not, the response was
  truncated mid-analysis → return "".

After stripping, GPQA-style graders run their letter-extraction regex on the
result; the IFEval verifier runs its instruction checks on the result.
"""

from __future__ import annotations

import re


_THINK_CLOSE = "</think>"
_THINK_OPEN_RE = re.compile(r"<think\b", re.IGNORECASE)

# Harmony channel-marker residue detector (gpt-oss raw string fallback path).
_HARMONY_ANALYSIS_PREFIX = "analysis"
_HARMONY_FINAL_MARKER = "assistantfinal"


def _strip_harmony_raw(text: str) -> str | None:
    """If `text` looks like a raw-decoded Harmony response, return the final
    channel content ("" if truncated). Otherwise return None (not Harmony).

    Signature we look for: response starts with the literal word "analysis"
    (the `<|channel|>analysis<|message|>` marker with special tokens stripped
    by tokenizer.decode). Normal responses almost never start that way.
    """
    if not text.startswith(_HARMONY_ANALYSIS_PREFIX):
        return None
    # Harmony residue has the message content fused directly to the token name
    # (e.g. "analysisWe need to...") because `<|message|>` was stripped by
    # tokenizer.decode. A legitimate English response starting with the word
    # "analysis" would have whitespace after it. Treat "analysis<space>" or
    # end-of-string as non-Harmony.
    tail = text[len(_HARMONY_ANALYSIS_PREFIX):]
    if not tail or tail[0].isspace():
        return None

    final_idx = text.rfind(_HARMONY_FINAL_MARKER)
    if final_idx < 0:
        # Truncated mid-analysis, no final channel — no answer.
        return ""
    return text[final_idx + len(_HARMONY_FINAL_MARKER):].strip()


def strip_reasoning(response) -> str:
    """Return the final-answer portion of a model response with any reasoning
    trace removed. Returns "" for None / empty / truncated-thinking inputs.
    """
    if response is None:
        return ""

    if isinstance(response, list):
        # Harmony-style list of content parts. Concatenate text parts only.
        pieces: list[str] = []
        for block in response:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    pieces.append(block.get("text", "") or "")
            else:
                # Unknown element type — coerce to str to avoid silently
                # dropping signal, but don't recurse.
                pieces.append(str(block))
        return "\n".join(p for p in pieces if p).strip()

    if isinstance(response, str):
        if _THINK_OPEN_RE.search(response):
            close_idx = response.rfind(_THINK_CLOSE)
            if close_idx < 0:
                # Think opened but never closed → truncated, no final answer.
                return ""
            return response[close_idx + len(_THINK_CLOSE):].strip()

        harmony = _strip_harmony_raw(response)
        if harmony is not None:
            return harmony

        return response.strip()

    # Fallback for exotic types
    return str(response).strip()
