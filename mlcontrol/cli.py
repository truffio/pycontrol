"""Headless command-line runner: ``python -m mlcontrol train dc_motor``."""

from __future__ import annotations

import argparse
import json
import sys

from mlcontrol.examples import EXAMPLES


def _parse_overrides(pairs):
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        out[key] = int(value) if value.lstrip("-").isdigit() else float(value)
    return out


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="mlcontrol", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list available examples")
    train = sub.add_parser("train", help="train a controller for one example")
    train.add_argument("example", choices=sorted(EXAMPLES))
    train.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values")
    train.add_argument("--out", help="write the result as JSON to this file")
    args = parser.parse_args(argv)

    if args.command == "list":
        for key, ex in EXAMPLES.items():
            print(f"{key:10s} [{ex.level}] {ex.title}")
        return 0

    example = EXAMPLES[args.example]

    def progress(info):
        print(f"[{info.stage}] {info.iteration:4d}/{info.n_iter}  "
              f"best={info.best_cost:.4g}  mean={info.mean_cost:.4g}", flush=True)

    result = example.train(_parse_overrides(args.set), callback=progress)
    evaluation = example.evaluate(result)
    print(f"\nTrained in {result.elapsed_s:.1f} s\n")
    print(f"{'metric':32s} {'learned':>12s} {'baseline':>12s}")
    for row in evaluation.metrics:
        print(f"{row['metric']:32s} {_fmt(row['learned']):>12s} {_fmt(row['baseline']):>12s}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=1)
        print(f"\nResult written to {args.out}")
    return 0


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)


if __name__ == "__main__":
    sys.exit(main())
