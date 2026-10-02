"""Tools the PR agent is allowed to have side effects through.

The agent runs with `permission_mode="bypassPermissions"`, so it can also use the
CLI's own Bash tool. These exist anyway, for three reasons:

1. **Standardisation.** A branch is always named the same way, an EC2 test host is
   always tagged and always terminated, a notification always carries the same fields.
   Freehand shell would get that right most days.
2. **Auditability.** A tool call is a named span in AgentCore Observability. `bash`
   with a 400-character command line is not.
3. **Refusal.** `github_open_pr` cannot push to main and cannot merge. That is
   enforced here, in code, rather than asked for in a prompt.

Each tool returns a dict the model can read, and each one also exposes `.raw()` for
`app.py` to call directly during setup and teardown.
"""
from __future__ import annotations

from claude_agent_sdk import create_sdk_mcp_server

from . import (
    bedrock_tools,
    checkpoint_tools,
    ec2_tools,
    github_tools,
    notify_tools,
    scan_tools,
    source_tools,
)

checkpoint_read = checkpoint_tools.checkpoint_read
checkpoint_write = checkpoint_tools.checkpoint_write


def build_tool_servers() -> dict:
    """In-process MCP servers, grouped so the model sees a coherent toolbox."""
    return {
        "repo": create_sdk_mcp_server(
            name="repo",
            version="1.0.0",
            tools=[
                github_tools.github_clone,
                github_tools.github_recent_changes,
                github_tools.github_list_open_prs,
                github_tools.github_create_branch,
                github_tools.github_commit,
                github_tools.github_push,
                github_tools.github_open_pr,
            ],
        ),
        "bedrock": create_sdk_mcp_server(
            name="bedrock",
            version="1.0.0",
            tools=[
                bedrock_tools.bedrock_list_models,
                bedrock_tools.bedrock_list_runtime_models,
                bedrock_tools.bedrock_invoke,
            ],
        ),
        "testhost": create_sdk_mcp_server(
            name="testhost",
            version="1.0.0",
            tools=[
                ec2_tools.testhost_launch,
                ec2_tools.testhost_run,
                ec2_tools.testhost_terminate,
                ec2_tools.testhost_list_orphans,
            ],
        ),
        "sources": create_sdk_mcp_server(
            name="sources",
            version="1.0.0",
            tools=[source_tools.sources_feed, source_tools.sources_fetch],
        ),
        "quality": create_sdk_mcp_server(
            name="quality",
            version="1.0.0",
            tools=[scan_tools.scan_run, scan_tools.scan_all],
        ),
        "ops": create_sdk_mcp_server(
            name="ops",
            version="1.0.0",
            tools=[
                checkpoint_tools.checkpoint_read,
                checkpoint_tools.checkpoint_write,
                notify_tools.notify_email,
            ],
        ),
    }
