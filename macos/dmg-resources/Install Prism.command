#!/usr/bin/env bash
# Double-click this file from the Prism DMG to install without Gatekeeper issues.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_SRC="$SCRIPT_DIR/Prism.app"
APP_DEST="/Applications/Prism.app"

if [ ! -d "$APP_SRC" ]; then
  osascript -e 'display alert "Prism installer" message "Could not find Prism.app next to this installer." as critical'
  exit 1
fi

echo "Installing Prism to Applications..."
rm -rf "$APP_DEST"
ditto "$APP_SRC" "$APP_DEST"
xattr -cr "$APP_DEST"

echo "Removing download quarantine flags..."
echo "Done. Opening Prism..."
open "$APP_DEST"
