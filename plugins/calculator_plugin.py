"""Sample calculator plugin for PluginAgent."""

from __future__ import annotations

import re
from typing import Optional


PLUGIN_NAME = "CalculatorPlugin"
TRIGGERS = [
    "calculate",
    "what is",
    "multiply",
    "plus",
]

_WORD_OPERATORS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "multiplied by": "*",
    "multiply": "*",
    "x": "*",
    "divided by": "/",
    "over": "/",
}

_ALLOWED_EXPR = re.compile(r"^[0-9+\-*/().\s]+$")


def _extract_expression(text: str) -> Optional[str]:
    value = (text or "").lower().strip()
    if not value:
        return None

    value = value.replace("what is", "").replace("calculate", "").strip()

    for phrase, op in _WORD_OPERATORS.items():
        value = value.replace(phrase, f" {op} ")

    value = re.sub(r"\s+", " ", value).strip()

    # Keep only safe expression characters.
    value = re.sub(r"[^0-9+\-*/().\s]", "", value)
    value = value.strip()

    if not value or not _ALLOWED_EXPR.fullmatch(value):
        return None

    return value


async def handle(event) -> str:
    """Handle calculation intent from IntentRecognizedEvent-like object."""
    raw = getattr(event, "raw_text", "") or getattr(event, "intent", "")
    expression = _extract_expression(raw)

    if not expression:
        return "I can only calculate numeric expressions with +, -, *, / and parentheses."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return "Division by zero is not allowed."
    except Exception:
        return "I couldn't evaluate that expression safely."

    return f"The result is {result}"
