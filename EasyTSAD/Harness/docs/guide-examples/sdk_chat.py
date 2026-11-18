"""DeepSeek Harness 0.2.2-rc.1 source SDK example. Makes a model call when run."""
from __future__ import annotations

import argparse
from pathlib import Path
from deepseek_harness import DeepSeekHarness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--session-id", default="guide-conversation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Credentials are inherited from the environment; do not put them in this file.
    with DeepSeekHarness(
        dsh_home=str(args.home.resolve()), cwd=str(args.workspace.resolve()),
        profile="sdk", provider="deepseek-official", model="deepseek-v4.1-flash",
    ) as harness:
        session = harness.resume_session(args.session_id) if args.resume else harness.start_session(args.session_id)
        result = session.run(args.prompt, on_notification=lambda item: print(item.method, item.params, flush=True))
        print("session:", result.session_id)
        print("finish:", result.finish_reason)
        print(result.final_response)


if __name__ == "__main__":
    main()
