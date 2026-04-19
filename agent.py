import os
import asyncio
from typing import Any

import dotenv
from github import Github
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentOutput, ToolCallResult, ToolCall
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate

dotenv.load_dotenv()

git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")
repo = git.get_repo(full_repo_name) if git and full_repo_name else None

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Fetch pull request details given the PR number. Returns author, title, body, diff_url, state, and commit SHAs."""
    pr = repo.get_pull(pr_number)
    commit_SHAs = [c.sha for c in pr.get_commits()]
    return {
        "user": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_SHAs": commit_SHAs,
    }


def get_pr_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Fetch commit details given a commit SHA. Returns changed files with filename, status, additions, deletions, changes, and patch."""
    commit = repo.get_commit(head_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


def get_file_content(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its file path."""
    return repo.get_contents(file_path).decoded_content.decode('utf-8')


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post a review comment to a GitHub pull request given the PR number and the review comment body."""
    pr = repo.get_pull(pr_number)
    pr.create_review(body=comment, event="COMMENT")
    return f"Review posted to PR #{pr_number}."


async def add_context_to_state(ctx: Context, context: str) -> str:
    """Useful for adding the gathered context to the state."""
    await ctx.store.set("gathered_contexts", context)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for adding the draft comment to the state."""
    await ctx.store.set("review_comment", draft_comment)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final review to the state."""
    await ctx.store.set("final_review", final_review)
    return "State updated with final review."


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
file_content_tool = FunctionTool.from_defaults(get_file_content)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository including PR details, changed files, and file contents.",
    tools=[pr_details_tool, commit_details_tool, file_content_tool, add_context_to_state],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather \n: "
        "\n    - The details: author, title, body, diff_url, state, and head_sha; \n"
        "\n    - Changed files; \n"
        "\n    - Any requested for files; \n"
        "Once you gather the requested info, you MUST hand control back to the Commentor Agent. "
    ),
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n "
        "Ensure to do the following for a thorough review: "
        "\n - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. "
        "\n - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n"
        "\n    - What is good about the PR? \n"
        "\n    - Did the author follow ALL contribution rules? What is missing? \n"
        "\n    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n"
        "\n    - Are new endpoints documented? - use the diff to determine this. \n "
        "\n    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n"
        "\n - If you need any additional details, you must hand off to the ContextAgent. \n"
        "\n - You should directly address the author. So your comments should sound like: \n"
        ' "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"'
        "\n\nCRITICAL INSTRUCTIONS - you MUST follow these steps in order:"
        "\n1. First, call add_comment_to_state with your draft review."
        "\n2. Then, IMMEDIATELY hand off to ReviewAndPostingAgent."
        "\nNEVER respond directly with the review. ALWAYS save it to state first, then hand off."
    ),
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment generated by the CommentorAgent, checks quality, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt=(
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. \n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n"
        "   - The review must: \n"
        "   - Be a ~200-300 word review in markdown format. \n"
        "   - Specify what is good about the PR: \n"
        "   - Did the author follow ALL contribution rules? What is missing? \n"
        "   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n"
        "   - Are there notes on whether new endpoints were documented? \n"
        "   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
        " If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
        " When you are satisfied, post the review to GitHub. "
        "\n - IMPORTANT: After posting the review to GitHub using post_review_to_github, respond with a final confirmation message. Do NOT call post_review_to_github more than once."
    ),
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
