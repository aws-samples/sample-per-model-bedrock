#!/usr/bin/env python3
"""Show specific cells of a notebook with full output: cell.py <nb> <i> [<i>...]"""
import json
import sys

nb = json.load(open(sys.argv[1]))
for arg in sys.argv[2:]:
    i = int(arg)
    c = nb["cells"][i]
    print(f"\n===== [{i}] {c['cell_type']} exec={c.get('execution_count')} =====")
    print("".join(c["source"]))
    for out in c.get("outputs", []):
        t = out.get("output_type")
        if t == "stream":
            body = "".join(out.get("text", []))
        elif t in ("execute_result", "display_data"):
            body = "".join(out.get("data", {}).get("text/plain", ["<rich>"]))
        elif t == "error":
            body = f"!!! {out.get('ename')}: {out.get('evalue')}"
        else:
            body = f"<{t}>"
        print("--- OUT ---")
        print(body)
