"""Safety filter for input and output validation."""

from __future__ import annotations

import re
from typing import Optional


# ── Prompt injection patterns ───────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?prior\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"show\s+(me\s+)?(your\s+)?system\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)", re.IGNORECASE),
    re.compile(r"\bDAN\s+mode\b", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"bypass\s+(your\s+)?(safety|content|moderation)\s+(filter|rules)", re.IGNORECASE),
]

# ── Harmful intent keywords ────────────────────────────────────────────

_HARMFUL_KEYWORDS: list[str] = [
    "write malware",
    "create a virus",
    "build a trojan",
    "hack into",
    "break into",
    "steal credentials",
    "steal passwords",
    "exploit vulnerability",
    "create ransomware",
    "ddos attack",
    "phishing email",
    "create phishing",
    "forge identity",
    "counterfeit",
    "synthesize drugs",
    "make explosives",
    "build a bomb",
    "create a weapon",
    "how to stalk",
    "harm someone",
]

# ── Output safety patterns ──────────────────────────────────────────────

_UNSAFE_OUTPUT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\.", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bopen\s*\(\s*['\"]\/etc\/", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"\bsudo\s+", re.IGNORECASE),
]

_MAX_INPUT_LENGTH = 5000


class SafetyFilter:
    """Rule-based safety checks for agent inputs and outputs.

    All checks are deterministic (no LLM calls), making them fast,
    reliable, and fully testable by the evals framework.
    """

    def __init__(self, system_prompt: str = "") -> None:
        """Initialize with the system prompt to check for leakage."""
        # Store key fragments of the system prompt to detect leakage
        self._system_prompt_fragments: list[str] = []
        if system_prompt:
            # Extract significant phrases (10+ chars) from the system prompt
            words = system_prompt.split()
            for i in range(0, len(words) - 4, 3):
                fragment = " ".join(words[i : i + 5]).lower()
                if len(fragment) >= 10:
                    self._system_prompt_fragments.append(fragment)

    def check_input(self, user_message: str) -> tuple[bool, Optional[str]]:
        """Check if the user input is safe.

        Returns:
            (is_safe, rejection_reason) — is_safe is True if the input passes
            all safety checks. rejection_reason is None when safe.
        """
        # Check 1: Length limit
        if len(user_message) > _MAX_INPUT_LENGTH:
            return False, (
                f"Input too long ({len(user_message)} characters). "
                f"Maximum is {_MAX_INPUT_LENGTH} characters."
            )

        # Check 2: Prompt injection patterns
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(user_message):
                return False, (
                    f"Potential prompt injection detected: "
                    f"matched pattern '{pattern.pattern}'."
                )

        # Check 3: Harmful intent keywords
        msg_lower = user_message.lower()
        for keyword in _HARMFUL_KEYWORDS:
            if keyword in msg_lower:
                return False, (
                    f"Potentially harmful request detected: "
                    f"matched keyword '{keyword}'."
                )

        return True, None

    def check_output(self, agent_response: str) -> tuple[bool, Optional[str]]:
        """Check if the agent output is safe.

        Returns:
            (is_safe, rejection_reason) — is_safe is True if the output passes
            all safety checks. rejection_reason is None when safe.
        """
        # Check 1: System prompt leakage
        response_lower = agent_response.lower()
        for fragment in self._system_prompt_fragments:
            if fragment in response_lower:
                return False, (
                    "Potential system prompt leakage detected in the response."
                )

        # Check 2: Dangerous code patterns
        for pattern in _UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(agent_response):
                return False, (
                    f"Unsafe code pattern detected in response: "
                    f"matched pattern '{pattern.pattern}'."
                )

        return True, None
