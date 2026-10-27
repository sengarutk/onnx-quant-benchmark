#!/usr/bin/env python3
"""
Injects generated result tables into README.md between markers of the form

    <!-- BEGIN:table2_latency_throughput --> ... <!-- END:table2_latency_throughput -->

The content is copied verbatim from results/tables/<name>.md (or results/<name> for non-table files), so the README
never contains hand-copied numbers. Use --check to fail if the README is out of date (CI).
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"(<!-- BEGIN:(?P<name>[\w.-]+) -->)(.*?)(<!-- END:(?P=name) -->)", re.S)


def render(readme: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group("name")
        src = ROOT / "results" / "tables" / f"{name}.md"
        if not src.is_file():
            src = ROOT / "results" / name
        body = src.read_text(encoding="utf-8").strip()
        if src.parent.name == "tables" and body.startswith("# "):
            body = body.split("\n", 1)[1].strip()  # drop the file's own H1 title
        return f"{m.group(1)}\n{body}\n{m.group(4)}"

    return PATTERN.sub(repl, readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
    current = path.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md is out of date with results/; run scripts/update_readme.py")
            sys.exit(1)
        return
    path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
