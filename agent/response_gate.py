"""Opt-in, per-turn release gate for assistant text.

The validator is an administrator-configured local Python script.  Hermes supplies
turn identity, a prompt digest, the fresh manifest path and the exact candidate on
stdin; only an explicit, identity-bound PASS releases that candidate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading
from typing import Any, Optional

from agent.message_content import flatten_message_text
from hermes_constants import get_hermes_home, get_scratch_dir, mkdir_under_hermes_home


logger = logging.getLogger(__name__)

_MAX_TRIGGER_CHARS = 32_000
_MAX_VALIDATOR_OUTPUT_BYTES = 64 * 1024
_MIN_TIMEOUT_SECONDS = 0.1
_MAX_TIMEOUT_SECONDS = 30.0
_BLOCKED_MESSAGES = {
    "MISSING_CURRENT_EVIDENCE": "BLOCKED：找不到本回合的證據清單，草稿未送出。",
    "STALE_IDENTITY_OR_ANSWER": "BLOCKED：證據與本回合身分或待送答案不一致，草稿未送出。",
    "INVALID_EVIDENCE": "BLOCKED：本回合證據未通過結構檢查，草稿未送出。",
    "VALIDATOR_FAILURE": "BLOCKED：自動檢查器無法使用、執行錯誤或逾時，草稿未送出。",
}
_REJECTED_DRAFT_SIDECARS = (
    "api_content",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "anthropic_content_blocks",
    "bedrock_content_blocks",
    "codex_reasoning_items",
    "codex_message_items",
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _block(state: dict[str, Any], reason_code: str) -> str:
    reason = reason_code if reason_code in _BLOCKED_MESSAGES else "VALIDATOR_FAILURE"
    state["verdict"] = "BLOCKED"
    state["blocked_reason"] = reason
    state.pop("approved_candidate_sha256", None)
    return _BLOCKED_MESSAGES[reason]


def _blocked_message(state: dict[str, Any]) -> str:
    return _BLOCKED_MESSAGES.get(
        state.get("blocked_reason"), _BLOCKED_MESSAGES["VALIDATOR_FAILURE"]
    )


def _strip_rejected_draft_sidecars(message: dict[str, Any]) -> None:
    for key in _REJECTED_DRAFT_SIDECARS:
        message.pop(key, None)


def _turn_state(agent: Any) -> Optional[dict[str, Any]]:
    state = getattr(agent, "_response_gate_turn", None)
    if not isinstance(state, dict):
        return None
    return state if state.get("turn_id") == getattr(agent, "_current_turn_id", None) else None


def response_gate_active(agent: Any) -> bool:
    """Whether the current turn, rather than merely the process, is gated."""
    state = _turn_state(agent)
    return bool(state and state.get("active"))


def _resolve_validator(home: Path, configured: Any) -> Path:
    raw = str(configured or "").strip()
    if not raw:
        raise ValueError("validator_script is empty")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (home / candidate).resolve()
    if not resolved.is_relative_to(home.resolve()):
        raise ValueError("relative validator_script escapes the active profile home")
    return resolved


def _timeout(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout_seconds is not numeric")
    parsed = float(value)
    if not (_MIN_TIMEOUT_SECONDS <= parsed <= _MAX_TIMEOUT_SECONDS):
        raise ValueError("timeout_seconds is outside the supported bounds")
    return parsed


def prepare_response_gate(agent: Any, original_user_message: Any, *, turn_id: str) -> str:
    """Bind one fresh gate to ``turn_id`` and return its model-facing instruction.

    Disabled and non-matching turns clear the prior state and return an empty string.
    Regex matching is intentionally limited to visible user text; image/audio payloads
    are ignored and the match input is capped.
    """
    agent._response_gate_turn = None
    try:
        from hermes_cli.config import load_config

        cfg = ((load_config() or {}).get("agent") or {}).get("response_gate") or {}
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            return ""
        pattern = str(cfg.get("trigger_pattern") or "")
        visible_text = flatten_message_text(original_user_message)
        if not pattern or re.search(pattern, visible_text[:_MAX_TRIGGER_CHARS]) is None:
            return ""

        home = get_hermes_home()
        nonce = secrets.token_urlsafe(18)
        gate_dir = mkdir_under_hermes_home(get_scratch_dir(home, prune=False) / "response-gate")
        manifest_path = gate_dir / f"{nonce}.json"
        state = {
            "active": True,
            "session_id": str(getattr(agent, "session_id", "") or ""),
            "turn_id": str(turn_id),
            "nonce": nonce,
            "question_sha256": _digest(visible_text),
            "manifest_path": str(manifest_path),
            "validator_script": str(_resolve_validator(home, cfg.get("validator_script"))),
            "timeout_seconds": _timeout(cfg.get("timeout_seconds", 10)),
            "verdict": None,
            "pending_assistant_message": None,
        }
    except (OSError, TypeError, ValueError, re.error) as exc:
        # An enabled gate whose matching/configuration phase fails is fail-closed for
        # this turn.  Do not include the regex, prompt or path in the user-facing error.
        logger.warning("Response gate setup failed: %s", type(exc).__name__)
        state = {
            "active": True,
            "session_id": str(getattr(agent, "session_id", "") or ""),
            "turn_id": str(turn_id),
            "nonce": secrets.token_urlsafe(18),
            "question_sha256": "",
            "manifest_path": "",
            "validator_script": "",
            "timeout_seconds": 10.0,
            "setup_error": True,
            "verdict": None,
            "pending_assistant_message": None,
        }
    agent._response_gate_turn = state
    return (
        "[Automatic response gate]\n"
        "Before your final answer, write the validator-defined evidence manifest to exactly:\n"
        f"{state['manifest_path']}\n"
        "The trusted validator is at:\n"
        f"{state['validator_script']}\n"
        "Consult the configured skill's documentation for this validator's manifest schema. "
        "Bind it to these exact values: "
        f"session_id={state['session_id']!r}, turn_id={state['turn_id']!r}, "
        f"nonce={state['nonce']!r}, question_sha256={state['question_sha256']!r}. "
        "Include draft_response as the exact final answer text (no hash or paraphrase). "
        "Do not run the validator yourself; Hermes runs the final guard automatically."
    )


def mark_unsupported_runtime(agent: Any) -> None:
    state = _turn_state(agent)
    if state:
        state["setup_error"] = True


def defer_assistant_message(agent: Any, message: dict[str, Any]) -> bool:
    """Hold the final assistant row in memory until its candidate is checked."""
    state = _turn_state(agent)
    if not state or not state.get("active"):
        return False
    state["pending_assistant_message"] = dict(message)
    return True


def append_deferred_assistant_message(agent: Any, messages: list[dict[str, Any]], text: str) -> None:
    """Append the held row with only the checked outbound text."""
    state = _turn_state(agent)
    if not state:
        return
    pending = state.pop("pending_assistant_message", None)
    if not isinstance(pending, dict):
        return
    pending["content"] = text
    if state.get("verdict") == "BLOCKED":
        _strip_rejected_draft_sidecars(pending)
    else:
        pending.pop("api_content", None)
    messages.append(pending)


def _run_validator(script: str, payload: bytes, timeout: float) -> tuple[int, bytes, bool]:
    """Run the trusted script with a bounded stdout collector.

    The reader keeps draining after the cap so a noisy child cannot deadlock on a full
    pipe, but only the first bounded prefix is retained.
    """
    output = bytearray()
    too_large = False
    with tempfile.TemporaryFile() as stdin_file:
        stdin_file.write(payload)
        stdin_file.seek(0)
        proc = subprocess.Popen(  # noqa: S603 -- administrator-configured Python script, no shell
            [sys.executable, script],
            stdin=stdin_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        def _read_stdout() -> None:
            nonlocal too_large
            assert proc.stdout is not None
            try:
                while chunk := proc.stdout.read(8192):
                    remaining = _MAX_VALIDATOR_OUTPUT_BYTES + 1 - len(output)
                    if remaining > 0:
                        output.extend(chunk[:remaining])
                    if len(output) > _MAX_VALIDATOR_OUTPUT_BYTES or len(chunk) > remaining:
                        too_large = True
            except (OSError, ValueError):
                too_large = True

        reader = threading.Thread(target=_read_stdout, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            reader.join(timeout=0.5)
            if reader.is_alive():
                too_large = True
                if proc.stdout is not None:
                    proc.stdout.close()
                reader.join(timeout=0.5)
    return int(proc.returncode or 0), bytes(output), too_large or reader.is_alive()


def validate_response(agent: Any, candidate: Any, *, turn_id: str) -> str:
    """Return the unchanged candidate only for a fresh, explicit validator PASS."""
    state = _turn_state(agent)
    if not state or not state.get("active") or state.get("turn_id") != turn_id:
        return candidate
    if state.get("verdict") == "PASS":
        if isinstance(candidate, str) and state.get("approved_candidate_sha256") == _digest(candidate):
            return candidate
        return _block(state, "STALE_IDENTITY_OR_ANSWER")
    if state.get("verdict") == "BLOCKED":
        return _blocked_message(state)
    if state.get("setup_error") or not isinstance(candidate, str):
        return _block(state, "VALIDATOR_FAILURE")

    identity = {
        key: state[key]
        for key in ("session_id", "turn_id", "nonce", "question_sha256")
    }
    envelope = {
        "schema_version": 1,
        **identity,
        "evidence_manifest_path": state["manifest_path"],
        "candidate_response": candidate,
    }
    expected = {
        "status": "PASS",
        **identity,
        "candidate_sha256": _digest(candidate),
    }
    try:
        script = Path(state["validator_script"])
        manifest = Path(state["manifest_path"])
        if not script.is_file():
            return _block(state, "VALIDATOR_FAILURE")
        if not manifest.is_file():
            return _block(state, "MISSING_CURRENT_EVIDENCE")
        returncode, stdout, oversized = _run_validator(
            str(script),
            json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
            float(state["timeout_seconds"]),
        )
        if oversized:
            return _block(state, "VALIDATOR_FAILURE")
        receipt = json.loads(stdout.decode("utf-8"))
        if isinstance(receipt, dict) and receipt.get("status") == "BLOCKED":
            return _block(state, receipt.get("reason_code"))
        if returncode != 0:
            return _block(state, "VALIDATOR_FAILURE")
        if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
            return _block(state, "STALE_IDENTITY_OR_ANSWER")
    except (OSError, UnicodeError, ValueError, TypeError, subprocess.TimeoutExpired):
        return _block(state, "VALIDATOR_FAILURE")
    state["verdict"] = "PASS"
    state["approved_candidate_sha256"] = expected["candidate_sha256"]
    return candidate


def block_unchecked_result(agent: Any, result: Any) -> Any:
    """Last-resort guard for early exits that bypass the normal finalizer."""
    state = _turn_state(agent)
    if not state or not state.get("active"):
        return result
    if not isinstance(result, dict):
        return result
    # Interrupt/error metadata remains intact for the surface's normal notice path;
    # only potentially model-authored final text is withheld.
    if result.get("interrupted"):
        result["final_response"] = ""
        return result

    candidate = result.get("final_response")
    if (
        state.get("verdict") == "PASS"
        and isinstance(candidate, str)
        and state.get("approved_candidate_sha256") == _digest(candidate)
    ):
        return result
    if state.get("verdict") == "BLOCKED":
        blocked = _blocked_message(state)
        result["final_response"] = blocked
    elif candidate:
        reason = "STALE_IDENTITY_OR_ANSWER" if state.get("verdict") == "PASS" else "VALIDATOR_FAILURE"
        blocked = _block(state, reason)
        result["final_response"] = blocked
    else:
        _block(state, "VALIDATOR_FAILURE")
        return result

    result["pre_transform_response"] = None
    result["response_transformed"] = False
    messages = result.get("messages")
    message = messages[-1] if isinstance(messages, list) and messages else None
    if isinstance(message, dict) and message.get("role") == "assistant":
        if message.get("content") != blocked:
            message["content"] = blocked
        _strip_rejected_draft_sidecars(message)
    return result


__all__ = [
    "append_deferred_assistant_message",
    "block_unchecked_result",
    "defer_assistant_message",
    "mark_unsupported_runtime",
    "prepare_response_gate",
    "response_gate_active",
    "validate_response",
]
