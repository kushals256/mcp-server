"""
CLI entry points for the Dataset Analysis MCP server.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from utils.claude_config import (
    MCP_SERVER_KEY,
    append_install_log,
    backup_config,
    build_mcp_entry,
    claude_config_path,
    ensure_data_dir,
    get_mcp_entry,
    install_log_path,
    is_disabled,
    is_mcp_configured,
    load_config,
    merge_mcp_entry,
    remove_mcp_entry,
    resolve_python_command,
    resolve_uvx_command,
    save_config,
    set_disabled,
    state_dir,
)

PACKAGE_NAME = "dataset-analysis-mcp"
GIT_INSTALL_URL = "git+https://github.com/kushals256/mcp-server.git"
SAMPLE_SOURCE = Path(__file__).resolve().parent / "examples" / "sample_sales.csv"


def run_server() -> None:
    from main import mcp

    mcp.run()


def _prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{message} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print()


def _copy_sample_dataset(data_dir: Path) -> Path:
    destination = data_dir / "sample_sales.csv"
    if SAMPLE_SOURCE.exists():
        shutil.copy2(SAMPLE_SOURCE, destination)
        append_install_log(f"Copied sample dataset to {destination}")
    return destination


def _install_menubar_app() -> bool:
    repo_app = Path(__file__).resolve().parent / "macos" / "build" / "Prism.app"
    target_app = Path("/Applications") / "Prism.app"

    if not repo_app.exists():
        print("Menu bar app bundle not found yet. Build it with: macos/scripts/build-app.sh")
        append_install_log("Menu bar app bundle missing during setup")
        return False

    if target_app.exists():
        shutil.rmtree(target_app)
    shutil.copytree(repo_app, target_app)
    append_install_log(f"Installed menu bar app to {target_app}")
    print(f"Installed menu bar companion to {target_app}")
    return True


def install_package() -> int:
    """Install the MCP package via pip (PyPI first, then git fallback)."""
    python_cmd = resolve_python_command()
    if not python_cmd:
        print("Python 3.10+ is required. Install from https://www.python.org/downloads/")
        return 1

    version = subprocess.check_output(
        [python_cmd, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        text=True,
    ).strip()
    major, minor, *_ = (int(part) for part in version.split("."))
    if (major, minor) < (3, 10):
        print(f"Python {version} found, but 3.10+ is required.")
        return 1

    append_install_log("Package install started")
    for target in (PACKAGE_NAME, GIT_INSTALL_URL):
        print(f"Installing {target}...")
        result = subprocess.run(
            [python_cmd, "-m", "pip", "install", "--user", "--upgrade", target],
            check=False,
        )
        if result.returncode == 0:
            append_install_log(f"Installed package from {target}")
            print("Package installed successfully.")
            return 0
        print(f"Install failed for {target}.")

    print("Could not install Prism. Check your internet connection and try again.")
    append_install_log("Package install failed")
    return 1


def run_setup_cli() -> None:
    """Entry point for dataset-analysis-mcp-setup with optional flags."""
    parser = argparse.ArgumentParser(prog="dataset-analysis-mcp-setup")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive setup with defaults (for Prism.app one-click install)",
    )
    parser.add_argument(
        "--install-package",
        action="store_true",
        help="Run pip install before configuring Claude",
    )
    args = parser.parse_args()

    if args.install_package:
        code = install_package()
        if code != 0:
            raise SystemExit(code)

    if args.yes:
        raise SystemExit(run_setup_noninteractive())

    raise SystemExit(run_setup())


def run_setup() -> int:
    """Interactive macOS setup wizard."""
    _print_header("Prism — Mac Setup")

    if platform.system() != "Darwin":
        print("This setup wizard is intended for macOS.")
        return 1

    state_dir()
    append_install_log("Setup wizard started")

    python_cmd = resolve_python_command()
    if not python_cmd:
        print("Python 3.10+ is required. Install with: brew install python@3.12")
        return 1

    version = subprocess.check_output([python_cmd, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"], text=True).strip()
    major, minor, *_ = (int(part) for part in version.split("."))
    if (major, minor) < (3, 10):
        print(f"Python {version} found, but 3.10+ is required.")
        return 1
    print(f"Python {version} found at {python_cmd}")

    use_uvx = resolve_uvx_command() is not None
    if use_uvx:
        print("uvx found — will configure Claude to launch the server with uvx.")
    else:
        print("uvx not found — will configure Claude with the current Python interpreter.")

    data_dir = ensure_data_dir()
    if _prompt_yes_no(f"Use data folder {data_dir}?", default=True):
        pass
    else:
        custom = input("Enter a custom data folder path: ").strip()
        data_dir = ensure_data_dir(custom)

    config_path = claude_config_path()
    config = load_config(config_path)
    backup_config(config_path)

    if use_uvx:
        entry = build_mcp_entry(use_uvx=True, data_dir=str(data_dir))
    else:
        main_path = str(Path(__file__).resolve().parent / "main.py")
        entry = build_mcp_entry(
            use_uvx=False,
            python_path=python_cmd,
            main_path=main_path,
            data_dir=str(data_dir),
        )

    updated = merge_mcp_entry(config, entry)
    save_config(updated, config_path)
    set_disabled(False)

    print()
    print("Claude Desktop config updated:")
    print(f"  {config_path}")
    print(f"  MCP server key: {MCP_SERVER_KEY}")

    if _prompt_yes_no("Copy sample dataset for your first run?", default=True):
        sample_path = _copy_sample_dataset(data_dir)
        print(f"Sample dataset ready at {sample_path}")

    if _prompt_yes_no("Install menu bar companion app to /Applications?", default=True):
        _install_menubar_app()

    print()
    print("Next steps:")
    print("  1. Run: dataset-analysis-mcp-doctor")
    print("  2. Quit Claude Desktop completely (Cmd+Q)")
    print("  3. Reopen Claude Desktop")
    print("  4. Try: Load ~/datasets/sample_sales.csv and run a data quality check")
    print()
    append_install_log("Setup wizard completed")
    return 0


def _check(label: str, ok: bool, detail: str) -> Tuple[bool, str]:
    status = "OK" if ok else "FAIL"
    line = f"[{status}] {label} — {detail}"
    print(line)
    return ok, line


def run_doctor() -> int:
    _print_header("Prism — Health Check")

    checks: List[bool] = []

    ok, _ = _check(
        "macOS",
        platform.system() == "Darwin",
        platform.platform() if platform.system() == "Darwin" else "setup wizard targets macOS",
    )
    checks.append(ok)

    python_cmd = resolve_python_command()
    if python_cmd:
        version = subprocess.check_output(
            [python_cmd, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            text=True,
        ).strip()
        major, minor, *_ = (int(part) for part in version.split("."))
        ok, _ = _check("Python", (major, minor) >= (3, 10), f"{version} at {python_cmd}")
    else:
        ok, _ = _check("Python", False, "python3 not found on PATH")
    checks.append(ok)

    try:
        import mcp  # noqa: F401
        import pandas  # noqa: F401

        ok, _ = _check("Dependencies", True, "mcp and pandas import successfully")
    except Exception as exc:  # pragma: no cover - defensive
        ok, _ = _check("Dependencies", False, str(exc))
    checks.append(ok)

    config_path = claude_config_path()
    config_exists = config_path.exists()
    ok, _ = _check("Claude config", config_exists, str(config_path))
    checks.append(ok)

    configured = is_mcp_configured() if config_exists else False
    ok, _ = _check(
        "MCP entry",
        configured,
        f"'{MCP_SERVER_KEY}' present in mcpServers" if configured else "missing or disabled",
    )
    checks.append(ok)

    entry = get_mcp_entry() if config_exists else None
    command = entry.get("command") if entry else None
    command_ok = bool(command and (shutil.which(command) or Path(command).exists()))
    ok, _ = _check("Launch command", command_ok, command or "not configured")
    checks.append(ok)

    data_dir = None
    if entry and entry.get("env"):
        data_dir = entry["env"].get("MCP_DATA_DIR")
    data_path = ensure_data_dir(data_dir) if data_dir or configured else Path.home() / "datasets"
    writable = os.access(data_path, os.W_OK)
    ok, _ = _check("Data folder", writable, str(data_path))
    checks.append(ok)

    disabled = is_disabled()
    ok, _ = _check("Server enabled", not disabled, "disabled flag set" if disabled else "ready")
    checks.append(ok)

    print()
    if all(checks):
        print("All checks passed. Restart Claude Desktop if you just ran setup.")
        append_install_log("Doctor passed all checks")
        return 0

    print("Some checks failed. Review the messages above or re-run setup.")
    print(f"Install log: {install_log_path()}")
    append_install_log("Doctor found failing checks")
    return 1


def run_disable() -> int:
    config_path = claude_config_path()
    config = load_config(config_path)
    backup_config(config_path)
    updated = remove_mcp_entry(config)
    save_config(updated, config_path)
    set_disabled(True)
    print("Prism disabled. Quit and reopen Claude Desktop to apply.")
    append_install_log("Server disabled via CLI")
    return 0


def run_enable() -> int:
    return run_setup_noninteractive()


def run_setup_noninteractive(
    *,
    data_dir: str | None = None,
    use_uvx: bool | None = None,
    copy_sample: bool = True,
) -> int:
    if platform.system() != "Darwin":
        print("This setup flow is intended for macOS.")
        return 1

    state_dir()
    append_install_log("Non-interactive setup started")

    python_cmd = resolve_python_command()
    if not python_cmd:
        print("Python 3.10+ is required.")
        return 1

    resolved_uvx = use_uvx if use_uvx is not None else resolve_uvx_command() is not None
    folder = ensure_data_dir(data_dir)

    config_path = claude_config_path()
    config = load_config(config_path)
    backup_config(config_path)

    if resolved_uvx:
        entry = build_mcp_entry(use_uvx=True, data_dir=str(folder))
    else:
        main_path = str(Path(__file__).resolve().parent / "main.py")
        entry = build_mcp_entry(
            use_uvx=False,
            python_path=python_cmd,
            main_path=main_path,
            data_dir=str(folder),
        )

    save_config(merge_mcp_entry(config, entry), config_path)
    set_disabled(False)

    if copy_sample:
        _copy_sample_dataset(folder)

    append_install_log("Non-interactive setup completed")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Run the MCP server")
    setup_parser = subparsers.add_parser("setup", help="Run the macOS setup wizard")
    setup_parser.add_argument("--yes", action="store_true")
    setup_parser.add_argument("--install-package", action="store_true")
    subparsers.add_parser("doctor", help="Run health checks")
    subparsers.add_parser("disable", help="Disable the MCP server in Claude config")
    subparsers.add_parser("enable", help="Re-enable the MCP server")

    args = parser.parse_args(argv)
    if args.command == "setup":
        if args.install_package:
            code = install_package()
            if code != 0:
                return code
        if args.yes:
            return run_setup_noninteractive()
        return run_setup()
    if args.command == "doctor":
        return run_doctor()
    if args.command == "disable":
        return run_disable()
    if args.command == "enable":
        return run_enable()

    run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
