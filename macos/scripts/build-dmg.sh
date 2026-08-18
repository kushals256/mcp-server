#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT_DIR/build/Prism.app"
DMG_DIR="$ROOT_DIR/build/dmg"
DMG_PATH="$ROOT_DIR/build/DatasetAnalysisMCP.dmg"

"$ROOT_DIR/scripts/build-app.sh"

rm -rf "$DMG_DIR" "$DMG_PATH"
mkdir -p "$DMG_DIR"
cp -R "$APP_DIR" "$DMG_DIR/"
cp "$ROOT_DIR/dmg-resources/Install Prism.command" "$DMG_DIR/"
cp "$ROOT_DIR/dmg-resources/README.txt" "$DMG_DIR/"
chmod +x "$DMG_DIR/Install Prism.command"
ln -s /Applications "$DMG_DIR/Applications"

hdiutil create -volname "Prism" -srcfolder "$DMG_DIR" -ov -format UDZO "$DMG_PATH"
echo "Created $DMG_PATH"
