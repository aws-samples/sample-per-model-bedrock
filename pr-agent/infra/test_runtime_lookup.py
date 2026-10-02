#!/usr/bin/env python3
"""Tests that the infra scripts find the runtime when it is not on the first page.

ListAgentRuntimes returns 10 runtimes per page, and the account has more than 10. A
lookup that reads only the first page misses the runtime, and deploy.py --step runtime
then calls create_agent_runtime for a name that already exists. Each test plants the
runtime on page 2, so a first-page-only lookup goes red.

Run directly (`python infra/test_runtime_lookup.py`) or under pytest. No network, no AWS:
deploy.py calls STS when imported, so its helper is loaded from source instead.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest import mock

INFRA = Path(__file__).resolve().parent
NAME = "bedrock_samples_pr_agent"


class FakeControl:
    """A bedrock-agentcore-control client with 12 runtimes over two pages."""

    def __init__(self) -> None:
        others = [{"agentRuntimeName": f"other_{i}", "agentRuntimeId": f"o{i}",
                   "agentRuntimeArn": f"arn:other:{i}"} for i in range(11)]
        target = {"agentRuntimeName": NAME, "agentRuntimeId": "target",
                  "agentRuntimeArn": "arn:target"}
        self.pages = [{"agentRuntimes": others[:10]},
                      {"agentRuntimes": others[10:] + [target]}]

    def list_agent_runtimes(self, **_):  # first page only, as the service does
        return self.pages[0]

    def get_paginator(self, op: str):
        assert op == "list_agent_runtimes", op
        return mock.Mock(paginate=lambda **_: iter(self.pages))


def _load_function(path: Path, name: str, **namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_deploy_finds_runtime_on_page_two():
    find = _load_function(INFRA / "deploy.py", "_find_runtime", RUNTIME_NAME=NAME)
    assert find(FakeControl())["agentRuntimeId"] == "target"


def test_deploy_returns_none_when_absent():
    find = _load_function(INFRA / "deploy.py", "_find_runtime", RUNTIME_NAME="missing")
    assert find(FakeControl()) is None


def test_invoke_finds_runtime_on_page_two():
    sys.path.insert(0, str(INFRA))
    with mock.patch("boto3.Session"):
        import invoke
    invoke.session = mock.Mock(client=lambda *_: FakeControl())
    assert invoke.runtime_arn() == "arn:target"


def test_no_script_reads_only_the_first_page():
    # teardown.py's lookup sits inside a function that deletes resources, so it is
    # guarded here: no infra script may call list_agent_runtimes() directly.
    offenders = [p.name for p in INFRA.glob("*.py")
                 if p.name != Path(__file__).name
                 and re.search(r"\.list_agent_runtimes\(", p.read_text())]
    assert not offenders, offenders


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
