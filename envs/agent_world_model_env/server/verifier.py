"""
Verifier execution for AWM environments.

Two verification modes:
- "sql":  Verifier extracts DB state, then an LLM judges the trajectory.
- "code": Verifier deterministically returns {"result": "complete"|"others"}.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# Wall-clock backstop for the verifier subprocess
VERIFIER_WALL_TIMEOUT_S = float(os.environ.get("OPENENV_AWM_VERIFIER_TIMEOUT", "60.0"))

# ---------------------------------------------------------------------------
# Shared judge runtime
#
# Every `verify` step in SQL mode needs one LLM-judge call. Instead of spinning
# up a fresh event loop + AsyncOpenAI client per call and blocking on it (one by
# one), all judge calls share a single background event loop and a single async
# client, and run concurrently up to MAX_INFLIGHT. Callers stay synchronous:
# they submit a coroutine and block on the future, but many such futures can be
# in flight against the judge endpoint at once.
# ---------------------------------------------------------------------------

# Max concurrent in-flight judge requests across all sessions in this process.
JUDGE_MAX_INFLIGHT = int(os.environ.get("OPENENV_AWM_JUDGE_MAX_INFLIGHT", "512"))
# Per-call wall-clock budget for a judge request (LLM latency, not the subprocess).
JUDGE_TIMEOUT_S = float(os.environ.get("OPENENV_AWM_JUDGE_TIMEOUT", "300.0"))

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_semaphore: asyncio.Semaphore | None = None
_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _get_shared_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide judge event loop, starting it on first use."""
    global _loop
    if _loop is not None:
        return _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="awm-judge-loop", daemon=True
            ).start()
            _loop = loop
    return _loop


async def _get_semaphore() -> asyncio.Semaphore:
    """Concurrency gate, created lazily on the shared loop.

    Safe without a lock: this only runs on the single shared loop, and there is
    no await between the check and the assignment, so it cannot interleave.
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(JUDGE_MAX_INFLIGHT)
    return _semaphore


def _get_client(base_url: str, api_key: str | None) -> AsyncOpenAI:
    """Reuse one AsyncOpenAI client per (base_url, api_key) on the shared loop."""
    key = (base_url, api_key or "EMPTY")
    client = _clients.get(key)
    if client is None:
        client = AsyncOpenAI(base_url=base_url, api_key=key[1])
        _clients[key] = client
    return client


def submit_judge(**kwargs: Any) -> tuple[str, dict]:
    """Run one LLM judge on the shared loop and block until it returns.

    Concurrent callers (different sessions) all dispatch onto the same loop, so
    their requests overlap up to JUDGE_MAX_INFLIGHT instead of running serially.
    Never raises — dispatch/timeout failures come back as ("llm_judge_error", ...).
    """
    loop = _get_shared_loop()
    future = asyncio.run_coroutine_threadsafe(run_llm_judge(**kwargs), loop)
    try:
        return future.result(timeout=JUDGE_TIMEOUT_S)
    except TimeoutError:
        future.cancel()
        return "llm_judge_error", {"error": f"Judge timed out after {JUDGE_TIMEOUT_S}s"}
    except Exception as e:  # noqa: BLE001 — scoring must never crash the step
        return "llm_judge_error", {"error": f"Judge dispatch failed: {e}"}

# isolate runner script
_RUNNER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_verifier_runner.py"
)


def _run_verifier_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    """Invoke the runner. Returns a dict; never raises."""
    try:
        proc = subprocess.run(
            [sys.executable, _RUNNER_PATH],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=VERIFIER_WALL_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "execution_status": "error",
            "error_message": (f"Verifier timed out after {VERIFIER_WALL_TIMEOUT_S}s"),
        }
    except (OSError, ValueError) as e:
        return {
            "execution_status": "error",
            "error_message": f"Failed to launch verifier subprocess: {e}",
        }

    if proc.returncode != 0 and not proc.stdout.strip():
        stderr = (proc.stderr or "").strip()
        return {
            "execution_status": "error",
            "error_message": (
                f"Verifier subprocess exited {proc.returncode}: {stderr[:500]}"
            ),
        }

    if not proc.stdout.strip():
        return {
            "execution_status": "error",
            "error_message": "Verifier subprocess produced no output",
        }

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {
            "execution_status": "error",
            "error_message": (
                f"Bad JSON from verifier subprocess ({e}): {proc.stdout[:500]}"
            ),
        }


def execute_sql_verifier(
    verifier_code: str,
    function_name: str,
    initial_db_path: str,
    final_db_path: str,
) -> dict[str, Any]:
    """Execute SQL-mode verifier code that compares initial vs final DB state.

    Returns:
        Verifier result dict, or error dict with "execution_status": "error".
    """
    if not os.path.exists(initial_db_path):
        return {
            "execution_status": "error",
            "error_message": f"Initial DB not found: {initial_db_path}",
        }
    if not os.path.exists(final_db_path):
        return {
            "execution_status": "error",
            "error_message": f"Final DB not found: {final_db_path}",
        }

    original_modes = {}
    for path in [initial_db_path, final_db_path]:
        try:
            original_modes[path] = os.stat(path).st_mode
            os.chmod(path, 0o444)
        except Exception:
            pass

    try:
        return _run_verifier_subprocess(
            {
                "verifier_code": verifier_code,
                "function_name": function_name,
                "initial_db_path": initial_db_path,
                "final_db_path": final_db_path,
                "final_answer": None,
                "mode": "sql",
            }
        )
    finally:
        for path, mode in original_modes.items():
            try:
                os.chmod(path, mode)
            except Exception:
                pass


def execute_code_verifier(
    verifier_code: str,
    function_name: str,
    initial_db_path: str,
    final_db_path: str,
    final_answer: str | None = None,
) -> dict[str, Any]:
    """Execute code-mode verifier that returns {"result": "complete"|"others"}.

    Returns:
        Dict with "result" key ("complete" or "others") and "execution_status".
    """
    if not os.path.exists(initial_db_path):
        return {
            "result": "others",
            "execution_status": "error",
            "error_message": f"Initial DB not found: {initial_db_path}",
        }
    if not os.path.exists(final_db_path):
        return {
            "result": "others",
            "execution_status": "error",
            "error_message": f"Final DB not found: {final_db_path}",
        }

    original_modes = {}
    for path in [initial_db_path, final_db_path]:
        try:
            original_modes[path] = os.stat(path).st_mode
            os.chmod(path, 0o444)
        except Exception:
            pass

    try:
        result = _run_verifier_subprocess(
            {
                "verifier_code": verifier_code,
                "function_name": function_name,
                "initial_db_path": initial_db_path,
                "final_db_path": final_db_path,
                "final_answer": final_answer,
                "mode": "code",
            }
        )
    finally:
        for path, mode in original_modes.items():
            try:
                os.chmod(path, mode)
            except Exception:
                pass

    # Normalize: code-mode callers expect "result" key on every return.
    if "result" not in result:
        result.setdefault("execution_status", "error")
        result["result"] = "others"
    return result


def run_verifier(
    verifier_entry: dict,
    verifier_mode: str,
    initial_db_path: str,
    final_db_path: str,
    final_answer: str | None = None,
) -> tuple[str, dict]:
    """
    Run the appropriate verifier and return (reward_type, details).

    Returns:
        (reward_type, verify_result_dict)
    """
    verification = verifier_entry.get("verification", {})
    code = verification.get("code", "")

    if not code or not isinstance(code, str) or len(code.strip()) < 10:
        return "code_verify_error", {"error": "No valid verifier code found"}

    default_func = (
        "verify_task_completion" if verifier_mode == "code" else "verify_task"
    )
    func_name = default_func
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def verify_") and "(" in line:
            func_name = line.split("(")[0].replace("def ", "").strip()
            break

    if verifier_mode == "code":
        result = execute_code_verifier(
            code, func_name, initial_db_path, final_db_path, final_answer
        )
        if result.get("execution_status") == "error":
            return "code_verify_error", result
        return result.get("result", "others"), result

    result = execute_sql_verifier(code, func_name, initial_db_path, final_db_path)
    if isinstance(result, dict) and result.get("execution_status") == "error":
        return "code_verify_error", result

    return "incomplete", result


def _normalize_azure_url(base_url: str) -> str:
    """Normalize Azure OpenAI base URLs to include /openai/v1 suffix.

    The Azure OpenAI v1 API uses the standard OpenAI client with
    base_url set to https://RESOURCE.openai.azure.com/openai/v1/.
    This helper auto-appends /openai/v1 when it detects an Azure endpoint.
    """
    if "openai.azure.com" in base_url or "services.ai.azure.com" in base_url:
        stripped = base_url.rstrip("/")
        if not stripped.endswith("/openai/v1"):
            return stripped + "/openai/v1"
    return base_url


async def run_llm_judge(
    task: str,
    verifier_result: dict,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    trajectory: list | None = None,
    verifier_reasoning: str = "",
    success_criteria: str = "",
    failure_criteria: str = "",
) -> tuple[str, dict]:
    """Run LLM-as-a-judge combining agent trajectory AND SQL verification results.

    Aligned with llm_judge_with_sql in agent-world-model-rl-rewards/task_reward.py.

    Works with both OpenAI and Azure OpenAI (v1 API). For Azure, set
    llm_base_url to https://YOUR-RESOURCE.openai.azure.com and the URL
    will be auto-normalized to append /openai/v1.

    Returns:
        (classification, judge_result_dict)
    """
    if not llm_base_url or not llm_model:
        return "llm_judge_error", {
            "error": "LLM endpoint not configured for sql verifier mode"
        }

    llm_base_url = _normalize_azure_url(llm_base_url)

    try:
        client = _get_client(llm_base_url, llm_api_key)

        system_prompt = textwrap.dedent("""\
            You are an impartial evaluator of automated agent task results with access to database verification. Based on the provided JSON trajectory AND the Python verification results from querying the database, decide the task outcome. This trajectory is generated by an MCP agent on a simulated simplified environment. The environment provides a set of MCP tools to help the agent complete the task.

            Input:
                task_json: dict containing task fields such as user task and agent execution_history.
                verification_json: dict containing Python code, reasoning, success_criteria, failure_criteria, and execution results that verified the database state from the Python function

            Output:
                You only output UTF-8 encoded string, avoid any emoji or special characters. You only output English text.

            Classification categories:
                - complete: all required steps and closure actions were successfully executed, AND the database state confirms the task was completed
                - incomplete: partial progress or the database state shows the task is not fully completed
                - server_error: the agent is blocked by MCP server/environment error, e.g., 5xx errors such as "Internal Server Error". Or the MCP server cannot process the valid tool call and return valid results. This can block the agent from completing the task.
                - agent_error: the agent made mistakes, invalid parameters, or missing required data without recovery, failed to complete the user's instruction.

            Priority order for classification:
                1) complete (trajectory shows success AND database confirms it)
                2) server_error (due to the MCP server/environment error)
                3) agent_error (agent-side issue, e.g., invalid mcp_tool_call arguments, hallucination, agent mistakes)
                4) incomplete (everything else unfinished or database state doesn't match expected outcome)

            Key considerations:
            - The verification_json contains checks performed on the database states. You can read the verification code to understand what checks were performed on the database states.
            - The verification_json contains the execution results of the verification code. You can use the execution results to help you judge the task completion.
            - The verification_json contains guidance for how to utilize the code execution results to help you judge the task completion, also including success_criteria and failure_criteria.
            - The verification results may be empty or error, or even the verification code itself is inaccurate. You should not fully rely on the verification results. You need to comprehensively consider the trajectory information to help you judge the task completion.

            Output format (must be valid JSON, no markdown fences, no additional commentary):
                {
                  "reasoning": "<concise explanation considering both trajectory and verification code execution results, the confidence score and considerations for each classification category>",
                  "confidence_score": [0-100, 0-100, 0-100, 0-100] for complete, incomplete, server_error, agent_error respectively,
                  "classification": "<one_of_[complete, incomplete, server_error, agent_error]>",
                  "evidence": {
                    "status": "<original result.status>",
                    "iterations": <int>,
                    "error_signals": ["<important error messages or codes>"],
                    "last_actions": ["<summaries of last few actions>"],
                    "last_status_codes": [<recent status codes>],
                    "database_verification": "<summary of what the database state changed based on code execution results of verification>"
                  }
                }""")

        task_payload = {
            "user_task": task,
            "actual_execution_steps": len(trajectory) if trajectory else 0,
            "trajectory": trajectory or [],
        }

        verification_json = _sanitize_for_json(
            {
                "reasoning": verifier_reasoning,
                "success_criteria": success_criteria,
                "failure_criteria": failure_criteria,
                "code_execution_result": verifier_result,
            }
        )
        try:
            verification_json_str = json.dumps(
                verification_json, ensure_ascii=False, indent=2, default=str
            )
        except Exception:
            verification_json_str = str(verification_json)

        user_prompt = (
            f"task_json:\n"
            f"{json.dumps(task_payload, ensure_ascii=False, indent=2, default=str)}\n\n"
            f"verification_json:\n"
            f"{verification_json_str}"
        )

        semaphore = await _get_semaphore()
        async with semaphore:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_completion_tokens=4096,
            )

        content = response.choices[0].message.content or ""
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    return "llm_judge_error", {
                        "error": f"Failed to parse LLM response: {content}"
                    }
            else:
                return "llm_judge_error", {
                    "error": f"Failed to parse LLM response: {content}"
                }

        classification = result.get("classification", "llm_judge_error").lower().strip()
        valid = {
            "complete",
            "incomplete",
            "server_error",
            "agent_error",
            "llm_judge_error",
        }
        if classification not in valid:
            classification = "llm_judge_error"

        return classification, result

    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return "llm_judge_error", {"error": str(e)}


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, tuple) else k: _sanitize_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
