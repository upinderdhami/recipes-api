"""
Stage 2 — ContextAgent: GitHub tools + ReActAgent + streaming main.
"""

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import UnknownObjectException
from llama_index.core.agent.workflow import AgentOutput, ReActAgent, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
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


# --- Step D: wrap functions as tools the LLM can call ------------------------
# FunctionTool reads: function name + docstring + type hints → JSON schema for the LLM.
# The agent picks a tool by name; informative docstrings matter a lot.
get_pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
get_file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
get_commit_details_tool = FunctionTool.from_defaults(fn=get_commit_details)

# Handy list to hand to the agent
tools = [
    get_pr_details_tool,
    get_file_contents_tool,
    get_commit_details_tool,
]


# --- Step E: LLM + ContextAgent (ReAct) --------------------------------------
# The LLM is the "brain". ReActAgent = Thought → Action (tool) → Observation → Answer.
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
"""

context_agent = ReActAgent(
    name="ContextAgent",
    description="Gathers pull request context from GitHub for later review agents.",
    system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
    tools=tools,
    llm=llm,
)


# --- Step F: run the agent with event streaming (Hyperskill pattern) ---------
async def main() -> None:
    """Read one question from stdin, run ContextAgent, stream thoughts/tools/answer."""
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = context_agent.run(prompt.format())

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
                print(event.response.content)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")


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
        print(f"Agent name: {context_agent.name}")
        print(f"Description: {context_agent.description}")
        print(f"LLM model: {os.getenv('OPENAI_MODEL')}")
        print(f"Tools: {[t.metadata.name for t in tools]}")
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
    # Default: chat with the agent (stage 2 check). Pass a smoke command to skip chat.
    #   poetry run python agent.py
    #   What's the title of PR number 1?
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
