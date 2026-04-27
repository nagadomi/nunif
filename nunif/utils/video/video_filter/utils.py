import io
import math
import tokenize
from typing import Any

from simpleeval import SimpleEval


def ffmpeg_expr_to_python(expr: str) -> str:
    """
    Safely convert FFmpeg expression to Python-compatible expression.
    Uses tokenize to replace reserved keywords used as functions in FFmpeg.
    """
    # Mapping from FFmpeg keyword to internal safe function name
    REPLACEMENT_MAP = {
        "if": "__if_func",
        "while": "__while_func",
        "and": "__and_func",
        "or": "__or_func",
        "not": "__not_func",
    }

    try:
        # Generate tokens from the expression
        tokens = tokenize.generate_tokens(io.StringIO(expr).readline)
        modified_tokens = []

        for tok in tokens:
            # Replace conflicting names to avoid Python SyntaxError
            if tok.type == tokenize.NAME and tok.string in REPLACEMENT_MAP:
                new_name = REPLACEMENT_MAP[tok.string]
                modified_tokens.append((tok.type, new_name, tok.start, tok.end, tok.line))
            else:
                modified_tokens.append(tok)

        # Reconstruct the string from tokens
        return tokenize.untokenize(modified_tokens).strip()
    except (tokenize.TokenError, IndentationError):
        # Fallback to original if tokenization fails
        return expr


class FFmpegEvaluator(SimpleEval):
    """
    A wrapper around SimpleEval that automatically handles FFmpeg expression
    to Python expression conversion (e.g. handling reserved keywords like 'if').
    """

    def eval(self, expr: str) -> Any:
        return super().eval(ffmpeg_expr_to_python(expr))


def _clip(x: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(x, max_val))


def _bitand(x: float, y: float) -> int:
    return int(x) & int(y)


def _bitor(x: float, y: float) -> int:
    return int(x) | int(y)


def _gt(x: float, y: float) -> float:
    return 1.0 if x > y else 0.0


def _lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def _eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def _if_func(cond: Any, true_val: Any, false_val: Any) -> Any:
    return true_val if cond else false_val


def _and_func(x: Any, y: Any) -> float:
    return 1.0 if x and y else 0.0


def _or_func(x: Any, y: Any) -> float:
    return 1.0 if x or y else 0.0


def _not_func(x: Any) -> float:
    return 1.0 if not x else 0.0


def _while_func(cond: Any, expr: Any) -> Any:
    # Note: Full while(cond, expr) support with lazy evaluation is complex
    # in simpleeval. This provides a basic placeholder.
    # In FFmpeg, cond is evaluated every time. Here it's already evaluated.
    return expr if cond else 0.0


def get_evaluator() -> FFmpegEvaluator:
    evaluator = FFmpegEvaluator()

    # Constants
    evaluator.names.update(
        {
            "PI": math.pi,
            "PHI": (1 + 5**0.5) / 2,  # Golden ratio
            "E": math.e,
        }
    )

    # Functions
    evaluator.functions.update(
        {
            "min": min,
            "max": max,
            "clip": _clip,
            "bitand": _bitand,
            "bitor": _bitor,
            "gt": _gt,
            "lt": _lt,
            "eq": _eq,
            "__if_func": _if_func,
            "__and_func": _and_func,
            "__or_func": _or_func,
            "__not_func": _not_func,
            "__while_func": _while_func,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "abs": abs,
            "exp": math.exp,
            "log": math.log,
            "pow": pow,
        }
    )
    return evaluator
