"""Tests for Claude config helpers and CLI doctor checks."""

import json
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

    from utils.claude_config import backup_config, load_config, save_config

    payload = {"mcpServers": {MCP_SERVER_KEY: build_mcp_entry(use_uvx=True)}}
    save_config(payload, config_path)
    loaded = load_config(config_path)
    assert loaded == payload

    backup = backup_config(config_path)
    assert backup is not None
    assert backup.exists()


def test_doctor_returns_nonzero_when_python_missing(monkeypatch):
    monkeypatch.setattr("cli.resolve_python_command", lambda: None)
    from cli import run_doctor

    assert run_doctor() == 1
