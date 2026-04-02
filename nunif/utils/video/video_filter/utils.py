import math
from typing import Any
from simpleeval import SimpleEval


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


def get_evaluator() -> Any:
    evaluator = SimpleEval()

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
            "if": _if_func,
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
