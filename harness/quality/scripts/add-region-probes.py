#!/usr/bin/env python3
"""Replace two bare region assertions with probes, matching the established pattern.

The audit of the 26 non-OpenAI notebooks found four that claim a model is absent from
a Region. Two of them (mistral, grok) back the claim with a `list_models(region)` loop,
so the claim re-verifies every time the notebook runs. Two (deepseek, kimi) assert it
in prose with nothing behind it, so it will age silently — the volatile-assertion
class we already fixed once in the OpenAI notebook.

Both claims are currently TRUE: verified live, deepseek.v3.2 and
moonshotai.kimi-k2-thinking both 404 in eu-central-1 and 200 in us-east-1. This is not
a correctness fix; it converts a fact with a shelf life into one that refreshes itself,
and it uses grok's existing pattern rather than a new one.

Usage: add-region-probes.py <repo-root>
"""

import sys

import nbformat

TARGETS = {
    "05-deepseek/01-deepseek-v3-reasoning.ipynb": {
        "models": '(("deepseek.v3.2", V32), ("deepseek.v3.1", V31))',
        "row_old": "| Region | Absent from eu-central-1 |",
    },
    "08-moonshot-kimi/01-kimi-k2.ipynb": {
        "models": '(("kimi-k2.5", K25), ("kimi-k2-thinking", THINKING))',
        "row_old": "| Region | Absent from eu-central-1 |",
    },
}

ROW_NEW = ("| Region | Per model, and it moves — the probe above asks the catalogue. "
           "The model card's regional table is authoritative |")

SECTION_MD = """\
## Regional footprint — check before you deploy

Which Regions carry a model is a per-model fact that changes as launches land, so this
asks the live catalogue rather than stating an answer that will quietly go stale.
"""

SECTION_CODE = '''\
regions = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
WATCH = __MODELS__

header = f"{'region':14}" + "".join(f"{label:>22}" for label, _ in WATCH)
print(header)
print("-" * len(header))
for region in regions:
    try:
        catalogue = list_models(region)
    except (RuntimeError, OSError) as exc:
        print(f"{region:14} {type(exc).__name__}")
        continue
    print(f"{region:14}" + "".join(
        f"{('yes' if model_id in catalogue else '-'):>22}" for _, model_id in WATCH))
'''


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for rel, spec in TARGETS.items():
        path = f"{root.rstrip('/')}/{rel}"
        nb = nbformat.read(path, as_version=4)

        # 1. list_models needs importing.
        setup = nb.cells[2]
        old_import = "from bedrock import err, parse_json_lenient, post, safe_print, ttft"
        new_import = "from bedrock import err, list_models, parse_json_lenient, post, safe_print, ttft"
        assert old_import in setup.source, f"{rel}: import line not as expected"
        setup.source = setup.source.replace(old_import, new_import, 1)

        # 2. Probe goes immediately before the Gotchas table.
        got = next(i for i, c in enumerate(nb.cells)
                   if c.cell_type == "markdown" and c.source.lstrip().startswith("## Gotchas"))
        code_src = SECTION_CODE.replace("__MODELS__", spec["models"])
        pair = [nbformat.v4.new_markdown_cell(SECTION_MD),
                nbformat.v4.new_code_cell(code_src)]
        for cell in pair:
            cell.pop("id", None)
        nb.cells[got:got] = pair

        # 3. The row now points at the probe.
        row = next(c for c in nb.cells
                   if c.cell_type == "markdown" and spec["row_old"] in c.source)
        row.source = row.source.replace(spec["row_old"], ROW_NEW, 1)

        nbformat.write(nb, path)
        print(f"{rel}: probe inserted at cell {got}, Region row repointed")


if __name__ == "__main__":
    main()
