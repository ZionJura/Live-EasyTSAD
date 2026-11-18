"""Run a TUI action sequence independently of the dashboard's lifetime."""
from __future__ import annotations

import json
import sys

from .cli import main


def execute(commands: list[list[str]]) -> int:
    for args in commands:
        print(f"[TUI] Starting {args[0]}", flush=True)
        result = main(args)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(execute(json.loads(sys.argv[1])))
