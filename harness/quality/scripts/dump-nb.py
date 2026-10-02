#!/usr/bin/env python3
"""Dump a notebook as compact text: markdown verbatim, code verbatim, outputs truncated.

    python3 dump-nb.py <path.ipynb> [--out-chars N] [--no-out]

Written for reading notebooks in an agent context, so it favours signal density:
cell indices are stable and quotable, outputs are clipped, images are named not
inlined.
"""
import json
import sys

OUT_CHARS = 900


def text_of(out) -> str:
    t = out.get("output_type")
    if t == "stream":
        return "".join(out.get("text", []))
    if t in ("execute_result", "display_data"):
        data = out.get("data", {})
        if "text/plain" in data:
            return "".join(data["text/plain"])
        return f"<{'/'.join(data)}>"
    if t == "error":
        return f"!!! {out.get('ename')}: {out.get('evalue')}"
    return f"<{t}>"


def main() -> None:
    path = sys.argv[1]
    limit = OUT_CHARS
    show_out = "--no-out" not in sys.argv
    if "--out-chars" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--out-chars") + 1])

    nb = json.load(open(path))
    print(f"##### FILE {path}  ({len(nb['cells'])} cells)")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]).rstrip()
        if c["cell_type"] == "markdown":
            print(f"\n[{i}] MD\n{src}")
            continue
        exe = c.get("execution_count")
        print(f"\n[{i}] CODE exec={exe}\n{src}")
        if not show_out:
            continue
        for out in c.get("outputs", []):
            body = text_of(out).rstrip()
            if len(body) > limit:
                body = body[:limit] + f"\n... [+{len(body) - limit} chars]"
            if body:
                print(f"  OUT> " + body.replace("\n", "\n  OUT> "))


if __name__ == "__main__":
    main()
