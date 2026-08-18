"""
Claude Desktop MCP configuration helpers for macOS setup.
"""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MCP_SERVER_KEY = "dataset-analysis"
DEFAULT_DATA_DIR = os.path.expanduser("~/datasets")


def claude_config_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"


def state_dir() -> Path:
    path = Path.home() / ".local" / "share" / "dataset-analysis-mcp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_log_path() -> Path:
    return state_dir() / "install.log"


def disabled_flag_path() -> Path:
    return state_dir() / "disabled"


def append_install_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with install_log_path().open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = path or claude_config_path()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def backup_config(path: Optional[Path] = None) -> Optional[Path]:
    config_path = path or claude_config_path()
    if not config_path.exists():
        return None
    backup_path = config_path.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(config_path, backup_path)
    append_install_log(f"Backed up Claude config to {backup_path}")
    return backup_path


def save_config(config: Dict[str, Any], path: Optional[Path] = None) -> Path:
    config_path = path or claude_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    append_install_log(f"Wrote Claude config to {config_path}")
    return config_path


def build_mcp_entry(
    *,
    use_uvx: bool = True,
    python_path: Optional[str] = None,
    main_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any]
    if use_uvx:
        entry = {
            "command": "uvx",
            "args": ["dataset-analysis-mcp"],
        }
    else:
        if not python_path or not main_path:
            raise ValueError("python_path and main_path are required when use_uvx=False")
        entry = {
            "command": python_path,
            "args": [main_path],
        }

    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    entry["env"] = {"MCP_DATA_DIR": resolved_data_dir}
    return entry


def merge_mcp_entry(
    config: Dict[str, Any],
    entry: Dict[str, Any],
    *,
    server_key: str = MCP_SERVER_KEY,
) -> Dict[str, Any]:
    merged = deepcopy(config)
    servers = merged.setdefault("mcpServers", {})
    servers[server_key] = entry
    return merged


def remove_mcp_entry(
    config: Dict[str, Any],
    *,
    server_key: str = MCP_SERVER_KEY,
) -> Dict[str, Any]:
    merged = deepcopy(config)
    servers = merged.get("mcpServers", {})
    servers.pop(server_key, None)
    if servers:
        merged["mcpServers"] = servers
    else:
        merged.pop("mcpServers", None)
    return merged


def is_mcp_configured(config: Optional[Dict[str, Any]] = None) -> bool:
    if disabled_flag_path().exists():
        return False
    data = config if config is not None else load_config()
    servers = data.get("mcpServers", {})
    return MCP_SERVER_KEY in servers


def get_mcp_entry(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    data = config if config is not None else load_config()
    return data.get("mcpServers", {}).get(MCP_SERVER_KEY)


def set_disabled(disabled: bool) -> None:
    flag = disabled_flag_path()
    if disabled:
        flag.write_text("1", encoding="utf-8")
        append_install_log("Server marked disabled via companion flag")
    elif flag.exists():
        flag.unlink()
        append_install_log("Server enabled via companion flag")


def is_disabled() -> bool:
    return disabled_flag_path().exists()


def ensure_data_dir(path: Optional[str] = None) -> Path:
    resolved = Path(path or DEFAULT_DATA_DIR).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_uvx_command() -> Optional[str]:
    for candidate in ("uvx", os.path.expanduser("~/.local/bin/uvx")):
        if shutil.which(candidate):
            return candidate
    return None


def resolve_python_command() -> Optional[str]:
    for candidate in ("python3", "python"):
        if shutil.which(candidate):
            return candidate
    return None
