"""Defense hook points. Person 1 calls these inside OfficeBot; Person 3 fills them in (defenses.py).

With no defense every hook is a no-op, so OfficeBot behaves as the vulnerable baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _allow_all(tool: str, args: dict, task: str) -> tuple[bool, str]:
    return True, ""


def _no_sanitize(source: str, text: str) -> str:
    return text


def _no_filter(text: str) -> tuple[str, bool]:
    return text, False


@dataclass
class DefenseHooks:
    # D2: runs on every piece of retrieved text (source is "email", "file" or "note")
    sanitize_retrieved: Callable[[str, str], str] = _no_sanitize
    # D2: appended to the system prompt
    extra_system_prompt: str = ""
    # D1: called before a tool runs. Return (allowed, reason). If not allowed, the call is blocked and traced.
    authorize_tool: Callable[[str, dict, str], tuple[bool, str]] = _allow_all
    # D3: runs on tool arguments and final output. Return (clean_text, flagged)
    egress_filter: Callable[[str], tuple[str, bool]] = _no_filter


def build_hooks(defense: str) -> DefenseHooks:
    """Person 3 replaces this with the real D1/D2/D3 (defenses.py). Stub: no defenses."""
    try:
        from defenses import build_hooks as real_build_hooks  # type: ignore

        return real_build_hooks(defense)
    except ImportError:
        return DefenseHooks()
