"""Tests for Claude config helpers and CLI doctor checks."""

import json
import sys
from pathlib import Path

import pytest

from utils.claude_config import (
    MCP_SERVER_KEY,
    build_mcp_entry,
    merge_mcp_entry,
    remove_mcp_entry,
    is_mcp_configured,
)


def test_build_mcp_entry_uvx():
    entry = build_mcp_entry(use_uvx=True, data_dir="/tmp/data")
    assert entry["command"] == "uvx"
    assert entry["args"] == ["dataset-analysis-mcp"]
    assert entry["env"]["MCP_DATA_DIR"] == "/tmp/data"


def test_build_mcp_entry_installed_cli():
    entry = build_mcp_entry(
        use_installed_cli=True,
        package_command="/Users/me/.local/bin/dataset-analysis-mcp",
        data_dir="/tmp/data",
    )
    assert entry["command"] == "/Users/me/.local/bin/dataset-analysis-mcp"
    assert entry["args"] == []
    assert entry["env"]["MCP_DATA_DIR"] == "/tmp/data"


def test_merge_and_remove_mcp_entry():
    config = {"mcpServers": {"other": {"command": "echo"}}}
    entry = build_mcp_entry(use_uvx=True)
    merged = merge_mcp_entry(config, entry)
    assert MCP_SERVER_KEY in merged["mcpServers"]

    removed = remove_mcp_entry(merged)
    assert MCP_SERVER_KEY not in removed.get("mcpServers", {})


def test_is_mcp_configured_respects_disabled_flag(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.claude_config.disabled_flag_path", lambda: tmp_path / "disabled")
    monkeypatch.setattr(
        "utils.claude_config.load_config",
        lambda path=None: {"mcpServers": {MCP_SERVER_KEY: build_mcp_entry(use_uvx=True)}},
    )

    assert is_mcp_configured() is True
    (tmp_path / "disabled").write_text("1", encoding="utf-8")
    assert is_mcp_configured() is False


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    config_path = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr("utils.claude_config.claude_config_path", lambda: config_path)
    monkeypatch.setattr("utils.claude_config.state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("utils.claude_config.install_log_path", lambda: tmp_path / "state" / "install.log")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    from utils.claude_config import backup_config, load_config, save_config

    payload = {"mcpServers": {MCP_SERVER_KEY: build_mcp_entry(use_uvx=True)}}
    save_config(payload, config_path)
    loaded = load_config(config_path)
    assert loaded == payload

    backup = backup_config(config_path)
    assert backup is not None
    assert backup.exists()


def test_run_setup_noninteractive_writes_config(tmp_path, monkeypatch):
    config_path = tmp_path / "claude_desktop_config.json"
    data_dir = tmp_path / "datasets"
    monkeypatch.setattr("cli.platform.system", lambda: "Darwin")
    monkeypatch.setattr("cli.resolve_python_command", lambda: sys.executable)
    monkeypatch.setattr("cli.resolve_package_command", lambda: None)
    monkeypatch.setattr("cli.resolve_uvx_command", lambda: None)
    monkeypatch.setattr("cli.claude_config_path", lambda: config_path)
    monkeypatch.setattr("cli.state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("cli.install_log_path", lambda: tmp_path / "state" / "install.log")
    monkeypatch.setattr("cli.ensure_data_dir", lambda path=None: data_dir)
    monkeypatch.setattr("cli.SAMPLE_SOURCE", tmp_path / "missing.csv")
    monkeypatch.setattr("cli.append_install_log", lambda msg: None)

    from cli import run_setup_noninteractive

    assert run_setup_noninteractive(data_dir=str(data_dir)) == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert MCP_SERVER_KEY in saved["mcpServers"]


def test_install_package_success(monkeypatch):
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr("cli.resolve_python_command", lambda: sys.executable)
    monkeypatch.setattr("cli.subprocess.check_output", lambda *a, **k: "3.12.0")
    monkeypatch.setattr("cli.subprocess.run", fake_run)
    monkeypatch.setattr("cli.append_install_log", lambda msg: None)

    from cli import install_package

    assert install_package() == 0
    assert calls[0][0] == sys.executable


def test_doctor_returns_nonzero_when_python_missing(monkeypatch):
    monkeypatch.setattr("cli.resolve_python_command", lambda: None)
    monkeypatch.setattr("cli.append_install_log", lambda msg: None)
    from cli import run_doctor

    assert run_doctor() == 1
