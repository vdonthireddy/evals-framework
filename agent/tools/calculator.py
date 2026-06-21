"""Calculator tool with safe math expression evaluation."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from agent.tools.base import BaseTool, ToolResult

# Allowed operators for safe evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return op_func(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are supported")
        func_name = node.func.id
        func = _SAFE_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function: {func_name}")
        if callable(func):
            args = [_safe_eval(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"'{func_name}' is a constant, not a function")

    if isinstance(node, ast.Name):
        # Handle named constants like pi and e
        value = _SAFE_FUNCTIONS.get(node.id)
        if value is not None and not callable(value):
            return value
        raise ValueError(f"Unknown variable: {node.id}")

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def evaluate_expression(expression: str) -> float | int:
    """Safely evaluate a mathematical expression string."""
    # Basic sanity checks
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")
    if len(expression) > 500:
        raise ValueError("Expression too long (max 500 characters)")

    # Block dangerous patterns
    dangerous = ["import", "exec", "eval", "compile", "__", "open", "os.", "sys."]
    expr_lower = expression.lower()
    for pattern in dangerous:
        if pattern in expr_lower:
            raise ValueError(f"Unsafe expression: contains '{pattern}'")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    return _safe_eval(tree)


class CalculatorTool(BaseTool):
    """Perform mathematical calculations safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Perform mathematical calculations. Supports arithmetic (+, -, *, /, **, %), "
            "and functions (sqrt, abs, round, ceil, floor, log, sin, cos, tan). "
            "Constants: pi, e."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '(25 * 37) + 12' or 'sqrt(144)'.",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        expression: str = kwargs.get("expression", "")

        if not expression.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Expression cannot be empty.",
            )

        try:
            result = evaluate_expression(expression)
            # Clean up float representation
            if isinstance(result, float) and result == int(result) and not math.isinf(result):
                result = int(result)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={"expression": expression, "result": result},
            )
        except (ValueError, ZeroDivisionError, TypeError, OverflowError) as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )
