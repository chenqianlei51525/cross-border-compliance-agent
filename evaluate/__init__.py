"""评测脚本入口：跑批量评测。

典型用法：
    python -m evaluate.run --dataset evaluate/eval_set.json
"""

from .pipeline import EvalItem, run_evaluation

__all__ = ["EvalItem", "run_evaluation"]
