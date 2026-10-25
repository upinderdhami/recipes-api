"""
Stage 2–3 — Multi-agent PR review: ContextAgent + CommentorAgent + AgentWorkflow.
"""

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import UnknownObjectException
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

load_dotenv()

# --- GitHub client -----------------------------------------------------------
# PyGithub talks to GitHub's API using your Personal Access Token.
token = os.getenv("GITHUB_TOKEN")
if not token:
    raise RuntimeError("GITHUB_TOKEN is missing from .env")

git = Github(auth=Auth.Token(token))

# Which repo the agent will inspect. Override with GITHUB_REPO_URL in .env if needed.
repo_url = os.getenv(
    "GITHUB_REPO_URL",
    "https://github.com/upinderdhami/recipes-api.git",
)
repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
username = repo_url.rstrip("/").split("/")[-2]
full_repo_name = f"{username}/{repo_name}"

repo = git.get_repo(full_repo_name)


# --- Tool function 1: PR details ---------------------------------------------
def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Fetch pull request details: author, title, body, diff_url, state, commits."""
    pull_request = repo.get_pull(pr_number)

    commit_shas: list[str] = []
    for c in pull_request.get_commits():
        commit_shas.append(c.sha)

    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_shas": commit_shas,
        "head_sha": pull_request.head.sha,
    }


# --- Tool function 2: file contents ------------------------------------------
def get_file_contents(file_path: str) -> str:
    """Fetch the text contents of a file from the repository given its path."""
    try:
        content_file = repo.get_contents(file_path)
    except UnknownObjectException:
        return f"File not found: {file_path}"

    # get_contents can return a list if file_path is a directory — we only want files.
    if isinstance(content_file, list):
        return f"Path is a directory, not a file: {file_path}"

    return content_file.decoded_content.decode("utf-8")


# --- Tool function 3: commit / changed-file details --------------------------
def get_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Fetch changed files for a commit: filename, status, additions, deletions, patch."""
    commit = repo.get_commit(commit_sha)

    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append(
            {
                "filename": f.filename,
                "status": f.status,  # e.g. added, modified, removed
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,  # the unified diff text (may be None for binaries)
            }
        )
    return changed_files


# --- Stage 3 Step A: shared state tools --------------------------------------
# Multi-agent systems need a shared "notebook" (state). Agents write here so the
# next agent can read what they produced. LlamaIndex injects `ctx` automatically
# when the first parameter is typed as Context.


async def add_context_to_state(ctx: Context, context: str) -> str:
    """Save gathered PR context into workflow state for other agents to use."""
    # Newer LlamaIndex: state lives on ctx.store (not ctx.get/set from older docs).
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = context
    await ctx.store.set("state", current_state)
    return "Context saved to state."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save a draft PR review comment into workflow state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft comment saved to state."


# --- Wrap functions as tools the LLM can call --------------------------------
# FunctionTool reads: function name + docstring + type hints → JSON schema for the LLM.
get_pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
get_file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
get_commit_details_tool = FunctionTool.from_defaults(fn=get_commit_details)
add_context_to_state_tool = FunctionTool.from_defaults(fn=add_context_to_state)
add_comment_to_state_tool = FunctionTool.from_defaults(fn=add_comment_to_state)

# GitHub tools (ContextAgent) + state tools (split per agent below)
github_tools = [
    get_pr_details_tool,
    get_file_contents_tool,
    get_commit_details_tool,
]
# Keep old name for smoke-test helper
tools = github_tools


# --- Stage 3 Step B: ContextAgent as FunctionAgent ---------------------------
# FunctionAgent (vs ReActAgent): still uses tools, but fits AgentWorkflow handoffs.
# can_handoff_to = names of agents it may pass control to (CommentorAgent in Step C).
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

CONTEXT_AGENT_SYSTEM_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather
:
 - The details: author, title, body, diff_url, state, and head_sha;
 - Changed files;
 - Any requested for files;
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
"""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description=(
        "Gathers all needed PR context from GitHub (details, commits, files), "
        "saves it to state, then hands off to CommentorAgent."
    ),
    system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
    tools=[*github_tools, add_context_to_state_tool],
    can_handoff_to=["CommentorAgent"],
)


# --- Stage 3 Step C: CommentorAgent ------------------------------------------
# This agent does NOT call GitHub directly. It asks ContextAgent for data
# (handoff), then writes a human-style review and saves it with add_comment_to_state.
COMMENTOR_AGENT_SYSTEM_PROMPT = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
 - What is good about the PR?
 - Did the author follow ALL contribution rules? What is missing?
 - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
 - Are new endpoints documented? - use the diff to determine this.
 - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the ContextAgent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
"""

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description=(
        "Uses the context gathered by the ContextAgent to draft a pull request "
        "review comment."
    ),
    system_prompt=COMMENTOR_AGENT_SYSTEM_PROMPT,
    tools=[add_comment_to_state_tool],
    can_handoff_to=["ContextAgent"],
)


# --- Stage 3 Step D: AgentWorkflow (orchestrator) ----------------------------
# Workflow owns both agents, starts at CommentorAgent, and seeds shared state.
# Commentor asks for data → handoff → Context gathers → handoff back → draft.
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent],
    root_agent=commentor_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
    },
)


# --- Stage 3 Step E: stream workflow events ----------------------------------
async def main() -> None:
    """Read one question, run the multi-agent workflow, stream handoffs/tools."""
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print(
                    "Selected tools: ",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(
                f"Calling selected tool: {event.tool_name}, "
                f"with arguments: {event.tool_kwargs}"
            )


def _smoke_test() -> None:
    """Optional helpers from earlier steps (not used by Hyperskill's main flow)."""
    command = sys.argv[1]
    print(f"Repo: {full_repo_name}")

    if command == "tools":
        for t in tools:
            meta = t.metadata
            print(f"- name: {meta.name}")
            print(f"  description: {meta.description}")
            print()
    elif command == "agent-info":
        print(f"LLM model: {os.getenv('OPENAI_MODEL')}")
        for agent in (context_agent, commentor_agent):
            print(f"\nAgent: {agent.name}")
            print(f"  Description: {agent.description}")
            print(f"  Tools: {[t.metadata.name for t in agent.tools]}")
            print(f"  Can hand off to: {agent.can_handoff_to}")
    elif command == "pr":
        print(get_pr_details(int(sys.argv[2])))
    elif command == "file":
        print(get_file_contents(sys.argv[2]))
    elif command == "commit":
        print(get_commit_details(sys.argv[2]))
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    # Default: multi-agent workflow (stage 3). Pass a smoke command to skip chat.
    #   poetry run python agent.py
    #   A new PR number 1 has been created. Write a review for it.
    if len(sys.argv) > 1 and sys.argv[1] in {
        "pr",
        "file",
        "commit",
        "tools",
        "agent-info",
    }:
        _smoke_test()
    else:
        asyncio.run(main())
    git.close()
