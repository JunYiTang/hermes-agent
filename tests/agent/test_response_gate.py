"""Invariant tests for the opt-in, per-turn assistant response gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


MISSING_EVIDENCE = "BLOCKED：找不到本回合的證據清單，草稿未送出。"
STALE_BINDING = "BLOCKED：證據與本回合身分或待送答案不一致，草稿未送出。"
INVALID_EVIDENCE = "BLOCKED：本回合證據未通過結構檢查，草稿未送出。"
VALIDATOR_FAILURE = "BLOCKED：自動檢查器無法使用、執行錯誤或逾時，草稿未送出。"


def _validator_script(path: Path, *, mode: str = "pass", marker: Path | None = None) -> Path:
    marker_stmt = f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')" if marker else "pass"
    behavior = {
        "pass": """
manifest = json.loads(Path(envelope['evidence_manifest_path']).read_text(encoding='utf-8'))
identity = manifest['identity']
keys = ('session_id', 'turn_id', 'nonce', 'question_sha256')
if any(identity.get(key) != envelope.get(key) for key in keys):
    print(json.dumps({'status': 'BLOCKED', 'reason_code': 'STALE_IDENTITY_OR_ANSWER'}))
    raise SystemExit(2)
if manifest.get('draft_response') != envelope.get('candidate_response'):
    print(json.dumps({'status': 'BLOCKED', 'reason_code': 'STALE_IDENTITY_OR_ANSWER'}))
    raise SystemExit(2)
receipt = {'status': 'PASS', **{key: envelope[key] for key in keys},
           'candidate_sha256': hashlib.sha256(envelope['candidate_response'].encode()).hexdigest()}
print(json.dumps(receipt))
""",
        "invalid": "print('not-json')",
        "oversized": "print('x' * 70000)",
        "error": "raise SystemExit(3)",
        "timeout": "time.sleep(1)",
        "blocked": (
            "print(json.dumps({'status': 'BLOCKED', 'reason_code': 'INVALID_EVIDENCE'})); "
            "raise SystemExit(2)"
        ),
    }[mode]
    path.write_text(
        "import hashlib, json, sys, time\nfrom pathlib import Path\n"
        "envelope = json.load(sys.stdin)\n"
        + marker_stmt + "\n" + behavior + "\n",
        encoding="utf-8",
    )
    return path


def _write_config(home: Path, script: Path, *, enabled: bool = True, pattern: str = "^GATE", timeout=1) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps({
        "agent": {"response_gate": {
            "enabled": enabled,
            "trigger_pattern": pattern,
            "validator_script": str(script),
            "timeout_seconds": timeout,
        }},
        "auxiliary": {"title_generation": {"enabled": False}},
    }), encoding="utf-8")


def _completion(text: str, **sidecars):
    fields = {
        "content": text, "tool_calls": None, "reasoning": None,
        "reasoning_content": None, "reasoning_details": None,
    }
    fields.update(sidecars)
    msg = SimpleNamespace(**fields)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="fake/model",
    )


def _agent(home: Path, *, stream=None, interim=None, reasoning=None, progress=None):
    db = SessionDB(db_path=home / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            model="fake/model", quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="cli", session_id="response-gate-session",
            session_db=db, stream_delta_callback=stream,
            interim_assistant_callback=interim, reasoning_callback=reasoning,
            tool_progress_callback=progress,
        )
    agent.client = MagicMock()
    return agent, db


def _write_manifest(agent, draft: str, *, identity_from: dict | None = None) -> dict:
    state = agent._response_gate_turn
    identity = identity_from or {
        key: state[key] for key in ("session_id", "turn_id", "nonce", "question_sha256")
    }
    manifest = {"identity": identity, "draft_response": draft}
    Path(state["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_validator_timeout_does_not_block_on_large_unread_stdin(tmp_path):
    from agent.response_gate import _run_validator

    script = tmp_path / "does-not-read.py"
    script.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")

    with pytest.raises(subprocess.TimeoutExpired):
        _run_validator(str(script), b"x" * 200_000, 0.2)


def test_cached_pass_and_outer_guard_are_bound_to_approved_candidate():
    from agent.response_gate import block_unchecked_result, validate_response

    approved = "APPROVED ANSWER"
    state = {
        "active": True,
        "turn_id": "test-turn",
        "verdict": "PASS",
        "approved_candidate_sha256": hashlib.sha256(approved.encode()).hexdigest(),
    }
    agent = SimpleNamespace(_current_turn_id="test-turn", _response_gate_turn=state)

    assert validate_response(agent, approved, turn_id="test-turn") == approved
    assert validate_response(agent, "UNVERIFIED CHANGED ANSWER", turn_id="test-turn") == STALE_BINDING

    state.update(
        verdict="PASS",
        approved_candidate_sha256=hashlib.sha256(approved.encode()).hexdigest(),
    )
    result = {
        "final_response": "UNVERIFIED CHANGED ANSWER",
        "messages": [{
            "role": "assistant", "content": "UNVERIFIED CHANGED ANSWER",
            "reasoning": "REJECTED REASONING", "reasoning_details": [{"text": "REJECTED DETAIL"}],
        }],
        "pre_transform_response": "REJECTED PRE-TRANSFORM",
        "response_transformed": True,
        "interrupted": False,
    }

    guarded = block_unchecked_result(agent, result)

    assert guarded["final_response"] == STALE_BINDING
    assert guarded["messages"] == [{"role": "assistant", "content": STALE_BINDING}]
    assert guarded["pre_transform_response"] is None
    assert guarded["response_transformed"] is False

    state.update(verdict="BLOCKED", blocked_reason="MISSING_CURRENT_EVIDENCE")
    state.pop("approved_candidate_sha256", None)
    already_blocked = block_unchecked_result(agent, {
        "final_response": "ANOTHER UNCHECKED ANSWER",
        "messages": [{"role": "assistant", "content": MISSING_EVIDENCE}],
        "interrupted": False,
    })
    assert already_blocked["final_response"] == MISSING_EVIDENCE

    state.update(
        verdict="PASS",
        approved_candidate_sha256=hashlib.sha256(approved.encode()).hexdigest(),
    )
    interrupted = block_unchecked_result(agent, {
        "final_response": "PARTIAL MODEL TEXT", "messages": [], "interrupted": True,
    })
    assert interrupted["final_response"] == ""
    assert state["verdict"] == "PASS"


def test_matched_missing_evidence_withholds_every_text_path_and_persists_block(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    streamed, interim, reasoning, progress = [], [], [], []
    agent, db = _agent(
        home, stream=streamed.append, interim=lambda text, **_: interim.append(text),
        reasoning=reasoning.append, progress=lambda *args, **kwargs: progress.append((args, kwargs)),
    )

    def create(**_kwargs):
        agent._fire_stream_delta("DRAFT SECRET")
        agent._fire_reasoning_delta("REASONING SECRET")
        agent._emit_interim_assistant_message({"role": "assistant", "content": "INTERIM SECRET"})
        return _completion("DRAFT SECRET")

    agent.client.chat.completions.create = create
    result = agent.run_conversation("GATE question", stream_callback=streamed.append)

    stored = [row["content"] for row in db.get_messages(agent.session_id) if row["role"] == "assistant"]
    assert result["final_response"] == MISSING_EVIDENCE
    assert streamed == []
    assert interim == []
    assert reasoning == []
    assert "DRAFT SECRET" not in json.dumps(progress, ensure_ascii=False)
    assert stored == [MISSING_EVIDENCE]
    assert "DRAFT SECRET" not in json.dumps(result["messages"], ensure_ascii=False)
    assert "INTERIM SECRET" not in json.dumps(result["messages"], ensure_ascii=False)


def test_existing_verify_stop_continues_without_releasing_gated_draft(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    streamed, interim, requests = [], [], []
    agent, db = _agent(
        home, stream=streamed.append,
        interim=lambda text, **_: interim.append(text),
    )
    answers = iter(["UNCHECKED DRAFT", "APPROVED RESPONSE"])

    def create(**kwargs):
        requests.append(json.dumps(kwargs.get("messages"), ensure_ascii=False, default=str))
        answer = next(answers)
        agent._fire_stream_delta(answer)
        if answer == "APPROVED RESPONSE":
            _write_manifest(agent, answer)
        return _completion(answer)

    agent.client.chat.completions.create = create
    with (
        patch("agent.verification_stop.verify_on_stop_enabled", return_value=True),
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=["run the existing verification", None],
        ),
    ):
        result = agent.run_conversation("GATE question")

    assert len(requests) == 2
    assert "[Automatic response gate withheld the unverified draft.]" in requests[1]
    assert "run the existing verification" in requests[1]
    assert "UNCHECKED DRAFT" not in requests[1]
    assert result["final_response"] == "APPROVED RESPONSE"
    assert [message["role"] for message in result["messages"]] == ["user", "assistant"]
    assert streamed == []
    assert interim == []
    visible = json.dumps(result["messages"], ensure_ascii=False, default=str)
    stored = json.dumps(db.get_messages(agent.session_id), ensure_ascii=False, default=str)
    assert "UNCHECKED DRAFT" not in visible
    assert "UNCHECKED DRAFT" not in stored
    assert "[Automatic response gate withheld the unverified draft.]" not in stored


def test_runtime_automatically_runs_validator_and_releases_exact_candidate(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    marker = tmp_path / "validator-ran"
    script = _validator_script(tmp_path / "validator.py", marker=marker)
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, db = _agent(home)
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        _write_manifest(agent, "APPROVED RESPONSE")
        return _completion(
            "APPROVED RESPONSE", reasoning="APPROVED REASONING",
            reasoning_details=[{"type": "reasoning.text", "text": "APPROVED DETAIL"}],
        )

    agent.client.chat.completions.create = create
    result = agent.run_conversation("GATE question")

    assert marker.read_text(encoding="utf-8") == "ran"
    assert agent._response_gate_turn["manifest_path"] in json.dumps(captured, default=str)
    assert result["final_response"] == "APPROVED RESPONSE"
    assert result["messages"][-1]["content"] == "APPROVED RESPONSE"
    assert "APPROVED REASONING" in result["messages"][-1]["reasoning"]
    assert "APPROVED DETAIL" in result["messages"][-1]["reasoning"]
    assert result["messages"][-1]["reasoning_details"][0]["text"] == "APPROVED DETAIL"
    assert str(script) in json.dumps(captured, default=str)
    assert [r["content"] for r in db.get_messages(agent.session_id) if r["role"] == "assistant"] == [
        "APPROVED RESPONSE"
    ]


def test_rejected_final_strips_draft_sidecars_and_transform_metadata(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, db = _agent(home)

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **_: ["TRANSFORMED REJECTED DRAFT"] if name == "transform_llm_output" else [],
    )

    def create(**_kwargs):
        _write_manifest(agent, "RAW REJECTED DRAFT")
        return _completion(
            "RAW REJECTED DRAFT", reasoning="REJECTED REASONING",
            reasoning_content="REJECTED REASONING CONTENT",
            reasoning_details=[{"type": "reasoning.text", "text": "REJECTED DETAIL"}],
        )

    agent.client.chat.completions.create = create
    result = agent.run_conversation("GATE question")
    stored = [row for row in db.get_messages(agent.session_id) if row["role"] == "assistant"]

    assert result["final_response"] == STALE_BINDING
    assert result["pre_transform_response"] is None
    assert result["response_transformed"] is False
    assert result["messages"][-1] == {
        key: value for key, value in result["messages"][-1].items()
        if key not in {"reasoning", "reasoning_content", "reasoning_details"}
    }
    leaked = json.dumps({"result": result, "stored": stored}, ensure_ascii=False, default=str)
    for secret in (
        "RAW REJECTED DRAFT", "TRANSFORMED REJECTED DRAFT", "REJECTED REASONING",
        "REJECTED REASONING CONTENT", "REJECTED DETAIL",
    ):
        assert secret not in leaked
    assert [row["content"] for row in stored] == [STALE_BINDING]


def test_known_adapter_reason_is_mapped_without_exposing_raw_output(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py", mode="blocked")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, _db = _agent(home)

    def create(**_kwargs):
        _write_manifest(agent, "REJECTED")
        return _completion("REJECTED")

    agent.client.chat.completions.create = create
    assert agent.run_conversation("GATE question")["final_response"] == INVALID_EVIDENCE


@pytest.mark.parametrize(
    ("manifest_draft", "expected"),
    [("RAW MODEL TEXT", STALE_BINDING), ("TRANSFORMED TEXT", "TRANSFORMED TEXT")],
)
def test_validator_binds_post_transform_candidate(
    manifest_draft, expected, tmp_path, monkeypatch,
):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, db = _agent(home)

    def invoke_hook(name, **_kwargs):
        return ["TRANSFORMED TEXT"] if name == "transform_llm_output" else []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)

    def create(**_kwargs):
        _write_manifest(agent, manifest_draft)
        return _completion("RAW MODEL TEXT")

    agent.client.chat.completions.create = create
    result = agent.run_conversation("GATE question")

    assert result["final_response"] == expected
    assert result["messages"][-1]["content"] == expected
    assert [r["content"] for r in db.get_messages(agent.session_id) if r["role"] == "assistant"] == [expected]


@pytest.mark.parametrize("mode", ["invalid", "oversized", "error", "timeout"])
def test_validator_failure_modes_block_without_draft_leak(mode, tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py", mode=mode)
    _write_config(home, script, timeout=0.1 if mode == "timeout" else 1)
    monkeypatch.setenv("HERMES_HOME", str(home))
    streamed = []
    agent, db = _agent(home, stream=streamed.append)

    def create(**_kwargs):
        _write_manifest(agent, "REJECTED DRAFT")
        agent._fire_stream_delta("REJECTED DRAFT")
        return _completion("REJECTED DRAFT")

    agent.client.chat.completions.create = create
    result = agent.run_conversation("GATE question")

    assert result["final_response"] == VALIDATOR_FAILURE
    assert streamed == []
    assert "REJECTED DRAFT" not in json.dumps(result["messages"], ensure_ascii=False)
    assert [r["content"] for r in db.get_messages(agent.session_id) if r["role"] == "assistant"] == [VALIDATOR_FAILURE]


def test_missing_validator_script_blocks_without_draft_leak(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_config(home, tmp_path / "missing-validator.py")
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, db = _agent(home)

    def create(**_kwargs):
        _write_manifest(agent, "UNRELEASED")
        return _completion("UNRELEASED")

    agent.client.chat.completions.create = create
    result = agent.run_conversation("GATE question")

    assert result["final_response"] == VALIDATOR_FAILURE
    assert "UNRELEASED" not in json.dumps(result["messages"], ensure_ascii=False)
    assert [r["content"] for r in db.get_messages(agent.session_id) if r["role"] == "assistant"] == [VALIDATOR_FAILURE]


def test_budget_fallback_cannot_release_unchecked_summary(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, db = _agent(home)
    agent.max_iterations = 0
    agent._handle_max_iterations = lambda *_: "BUDGET SECRET"

    result = agent.run_conversation("GATE question")

    assert result["final_response"] == MISSING_EVIDENCE
    assert "BUDGET SECRET" not in json.dumps(result["messages"], ensure_ascii=False)
    assert [r["content"] for r in db.get_messages(agent.session_id) if r["role"] == "assistant"] == [MISSING_EVIDENCE]


def test_codex_app_server_gate_fails_closed_before_model_call(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, db = _agent(home)
    agent.api_mode = "codex_app_server"
    called = []
    agent._run_codex_app_server_turn = lambda **_: called.append(True)

    result = agent.run_conversation("GATE question")

    assert called == []
    assert result["final_response"] == VALIDATOR_FAILURE
    assert [r["content"] for r in db.get_messages(agent.session_id) if r["role"] == "assistant"] == [VALIDATOR_FAILURE]


def test_previous_pass_cannot_be_reused_and_next_normal_turn_restores_streaming(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    script = _validator_script(tmp_path / "validator.py")
    _write_config(home, script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    streamed = []
    agent, _db = _agent(home, stream=streamed.append)
    prior_identity = {}
    calls = 0

    def create(**_kwargs):
        nonlocal calls, prior_identity
        calls += 1
        if calls == 1:
            _write_manifest(agent, "FIRST")
            prior_identity = dict(json.loads(Path(agent._response_gate_turn["manifest_path"]).read_text())["identity"])
            return _completion("FIRST")
        if calls == 2:
            _write_manifest(agent, "SECOND", identity_from=prior_identity)
            agent._fire_stream_delta("SECOND")
            return _completion("SECOND")
        agent._fire_stream_delta("NORMAL")
        return _completion("NORMAL")

    agent.client.chat.completions.create = create
    assert agent.run_conversation("GATE first")["final_response"] == "FIRST"
    assert agent.run_conversation("GATE second")["final_response"] == STALE_BINDING
    assert agent.run_conversation("ordinary follow-up")["final_response"] == "NORMAL"
    assert "SECOND" not in "".join(str(x) for x in streamed)
    assert "NORMAL" in "".join(str(x) for x in streamed)


@pytest.mark.parametrize(
    ("enabled", "pattern", "prompt"),
    [(False, "^GATE", "GATE question"), (True, "^GATE", "ordinary question")],
)
def test_disabled_or_nonmatching_turn_does_not_run_validator(
    enabled, pattern, prompt, tmp_path, monkeypatch,
):
    home = tmp_path / "profile"
    marker = tmp_path / "validator-ran"
    script = _validator_script(tmp_path / "validator.py", marker=marker)
    _write_config(home, script, enabled=enabled, pattern=pattern)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent, _db = _agent(home)
    agent.client.chat.completions.create = lambda **_: _completion("UNCHANGED")

    assert agent.run_conversation(prompt)["final_response"] == "UNCHANGED"
    assert not marker.exists()


def test_profile_scope_isolated_a_b_a(tmp_path, monkeypatch):
    from agent.response_gate import prepare_response_gate, response_gate_active
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    script = _validator_script(tmp_path / "validator.py")
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    _write_config(home_a, script, enabled=True)
    _write_config(home_b, script, enabled=False)
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    agent = SimpleNamespace(session_id="scope-session", _current_turn_id="a1")

    token = set_hermes_home_override(home_a)
    try:
        instruction_a1 = prepare_response_gate(agent, "GATE one", turn_id="a1")
        path_a1 = agent._response_gate_turn["manifest_path"]
    finally:
        reset_hermes_home_override(token)
    agent._current_turn_id = "b1"
    token = set_hermes_home_override(home_b)
    try:
        assert prepare_response_gate(agent, "GATE two", turn_id="b1") == ""
        assert not response_gate_active(agent)
    finally:
        reset_hermes_home_override(token)
    agent._current_turn_id = "a2"
    token = set_hermes_home_override(home_a)
    try:
        instruction_a2 = prepare_response_gate(agent, "GATE three", turn_id="a2")
        path_a2 = agent._response_gate_turn["manifest_path"]
    finally:
        reset_hermes_home_override(token)

    assert instruction_a1 and instruction_a2
    assert str(home_a) in path_a1 and str(home_a) in path_a2
    assert path_a1 != path_a2
