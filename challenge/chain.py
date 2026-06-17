"""GraphCypherQAChain-style helper for the live-LLM Tier 3 path.

Triple-stated Tier 3 scoring methodology (verbatim in
integration-task-spec.md, the published Integration Guide Tier 3
section, and this docstring):

- The 15 canonical eval questions in data/eval_questions.jsonl are scored by exact-result-set equivalence against the deterministic mapper's output on the same fixture graph (the deterministic mapper is the gold).
- A Tier 3 answer is correct iff the executed Cypher returns exactly the same set of result rows as the deterministic mapper for that question; row order matters only for the two ranked questions (#9, #12) where ORDER BY is in the canonical shape.
- A Tier 3 answer that raises UnsupportedCypherError (allowlist rejection) counts as incorrect for that question but is REPORTED SEPARATELY in the autograder summary so learners can distinguish "LLM emitted unsafe Cypher" from "LLM emitted safe-but-wrong Cypher".
- Aggregation: report per-question correctness plus an overall accuracy (correct / 15). No partial credit on rows.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .allowlist import UnsupportedCypherError, validate_query_shape
from .few_shots import EXAMPLE_PAIRS, SCHEMA_PREAMBLE


# Graceful import — langchain_neo4j is optional (not installed in CI).
# When unavailable, set the symbol to None and let callers decide whether
# to skip with a clear reason or use a fake.
try:
    from langchain_neo4j import GraphCypherQAChain  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    GraphCypherQAChain = None  # type: ignore
    LANGCHAIN_AVAILABLE = False


def build_prompt(question: str) -> str:
    """Compose the LLM prompt: schema preamble + few-shots + question.

    Course-helper stub. The exact prompt format is up to you, but at
    minimum:
      - Start with SCHEMA_PREAMBLE.
      - Append each EXAMPLE_PAIRS entry as "Q: ...\\nCypher: ...".
      - End with "Q: {question}\\nCypher:" so the LLM continues with Cypher.
    """
    # assemble the prompt string from SCHEMA_PREAMBLE + EXAMPLE_PAIRS + question.
    parts: list[str] = [SCHEMA_PREAMBLE.strip(), ""]

    for ex in EXAMPLE_PAIRS:
        q = ""
        cypher = ""
        params: dict[str, Any] = {}

        if isinstance(ex, dict):
            q = ex.get("question") or ex.get("query") or ex.get("nl") or ""
            cypher = ex.get("cypher") or ""
            maybe = ex.get("params", {})
            params = maybe if isinstance(maybe, dict) else {}
        elif isinstance(ex, (tuple, list)):
            if len(ex) >= 2:
                q, cypher = ex[0], ex[1]
            if len(ex) >= 3 and isinstance(ex[2], dict):
                params = ex[2]
        else:
            q = getattr(ex, "question", "") or getattr(ex, "query", "") or ""
            cypher = getattr(ex, "cypher", "") or ""
            maybe = getattr(ex, "params", {})
            params = maybe if isinstance(maybe, dict) else {}

        parts.append(f"Q: {q}")
        parts.append(f"Cypher: {cypher}")
        if params:
            parts.append(f"Params: {params}")
        parts.append("")

    parts.append(f"Q: {question}")
    parts.append("Cypher:")
    return "\n".join(parts).strip()


def _coerce_response_text(response: Any) -> str:
    """Normalize provider responses to plain text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response)
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(item.get("text", ""))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
    return str(response)


def _parse_params(raw: str) -> dict[str, Any]:
    """Parse Params: {...} content safely into a dict."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _extract_cypher_and_params(response: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract (cypher, params) from an LLM response."""
    if response is None:
        return None, {}

    if isinstance(response, dict):
        cypher = response.get("cypher") or response.get("query")
        params = response.get("params", {})
        if not isinstance(params, dict):
            params = {}
        return cypher, params

    text = _coerce_response_text(response).strip()
    if not text:
        return None, {}

    fence = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()

    m = re.search(r"Cypher:\s*(.*?)(?:\nParams:\s*(\{.*\}))?\s*$", text, re.S)
    if m:
        cypher = m.group(1).strip()
        raw_params = m.group(2) or ""
        return (cypher or None), _parse_params(raw_params)

    if "\nParams:" in text:
        cypher_part, params_part = text.split("\nParams:", 1)
        cypher = cypher_part.replace("Cypher:", "").strip()
        return (cypher or None), _parse_params(params_part.strip())

    cypher = text.replace("Cypher:", "").strip()
    return (cypher or None), {}


def run_chain(driver, llm_client, question: str) -> dict[str, Any]:
    """Run one question through the chain end-to-end.

    Returns a dict with keys:
      - "question": the input question
      - "cypher":   the LLM-emitted Cypher string (or None if the LLM
                    refused / returned empty)
      - "params":   the params dict the LLM emitted (or {} if none)
      - "rows":     list of result rows from session.run (or [] if
                    the allowlist rejected the Cypher)
      - "rejected": True iff the allowlist raised; False otherwise
      - "rejection_reason": the UnsupportedCypherError message, or None

    Required behaviour:
      1. Build the prompt via build_prompt(question).
      2. Invoke the LLM (llm_client.invoke(prompt) — LangChain Runnable
         convention).
      3. Parse the LLM response to extract a Cypher string and a params
         dict. (The few-shot format is "Cypher: ...\\nParams: {...}".)
      4. Call validate_query_shape(cypher). Catch UnsupportedCypherError
         and return a dict with rejected=True.
      5. If validation passed, run the Cypher via session.run(cypher,
         **params) and return the rows.
    """
    # orchestrate prompt → LLM → parse → allowlist → execute.
    prompt = build_prompt(question)
    response = llm_client.invoke(prompt)
    cypher, params = _extract_cypher_and_params(response)

    if cypher is None:
        return {
            "question":        question,
            "cypher":          None,
            "params":          {},
            "rows":            [],
            "rejected":        False,
            "rejection_reason": None,
        }

    try:
        validate_query_shape(cypher)
    except UnsupportedCypherError as e:
        return {
            "question":        question,
            "cypher":          cypher,
            "params":          params,
            "rows":            [],
            "rejected":        True,
            "rejection_reason": str(e),
        }

    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = [row.data() for row in result]

    return {
        "question":        question,
        "cypher":          cypher,
        "params":          params,
        "rows":            rows,
        "rejected":        False,
        "rejection_reason": None,
    }