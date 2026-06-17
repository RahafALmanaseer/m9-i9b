"""LLM client provider resolution (course-provided dispatch + learner-implemented chain wrapper).

The 5-step resolution order (Ollama-first; hosted-API fallback) and the
Ollama model-presence check are course-provided. Your TODO is the
actual chain wrapper — return an object that exposes a `.invoke(prompt)`
method (LangChain Runnable convention) so `chain.py` can call it
uniformly regardless of provider.
"""

from __future__ import annotations

import os
import shutil
import subprocess


class NoLLMClientAvailableError(RuntimeError):
    """No Ollama and no hosted-provider key is configured."""


class OllamaModelMissingError(RuntimeError):
    """Ollama is installed but the requested model has not been pulled."""


def _ollama_has_model(model: str) -> bool:
    """Return True iff `ollama list` reports `model` as installed.

    Course-provided; do not modify.
    """
    if shutil.which("ollama") is None:
        return False
    try:
        out = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    return any(model.split(":")[0] in line for line in out.stdout.splitlines())


def get_llm_client(model: str = "phi3:mini-4k-instruct-q4_K_M"):
    """Return a LangChain LLM client.

    Resolution order (course-provided dispatch — DO NOT change the order):
      1. OLLAMA_HOST env var → ChatOllama(model=model, base_url=OLLAMA_HOST)
      2. Local Ollama at http://localhost:11434 → ChatOllama(model=model).
         Before returning, run `ollama list` and confirm `model` is present;
         if not, raise OllamaModelMissingError with the exact pull command:
           "Model '<model>' not pulled. Run: ollama pull <model>"
      3. OPENAI_API_KEY in env → ChatOpenAI (fallback; optional per Compute Rule)
      4. ANTHROPIC_API_KEY in env → ChatAnthropic (fallback)
      5. Otherwise → raise NoLLMClientAvailableError with installation guidance

    Steps 1, 2 (the model-presence check), and 5 are course-provided. Your
    TODO is the actual wrapper construction in steps 1, 2, 3, 4 — return
    a configured LangChain client. The model-presence check at step 2 must
    fire BEFORE you return the client so a missing pull surfaces as a
    clean OllamaModelMissingError, not an opaque connection error at
    invoke time.
    """
    # Step 1: OLLAMA_HOST override
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model, base_url=ollama_host)

    # Step 2: local Ollama. Course-provided presence check.
    if shutil.which("ollama") is not None:
        if not _ollama_has_model(model):
            raise OllamaModelMissingError(
                f"Model {model!r} not pulled. Run: ollama pull {model}"
            )
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model)

    # Step 3: hosted OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
        )

    # Step 4: hosted Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    # Step 5: nothing configured — fail-loud with install guidance.
    raise NoLLMClientAvailableError(
        "No LLM provider available. Install Ollama (https://ollama.com) "
        f"and run: ollama pull {model}\n"
        "Or set OPENAI_API_KEY / ANTHROPIC_API_KEY in your .env."
    )