#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SRC_DIR="$ROOT_DIR/DatasetAnalysisMCP"
BUILD_DIR="$ROOT_DIR/build"
BRAND_EXPORT="$REPO_ROOT/brand/export"
APP_NAME="Prism.app"
APP_DIR="$BUILD_DIR/$APP_NAME"
BINARY_NAME="DatasetAnalysisMCP"

mkdir -p "$BUILD_DIR"

if [ ! -f "$BRAND_EXPORT/menubar-idle.png" ]; then
  echo "Brand assets missing. Run: python3 brand/export_assets.py"
  exit 1
fi

echo "Compiling menu bar app..."
swiftc "$SRC_DIR"/*.swift \
  -o "$BUILD_DIR/$BINARY_NAME" \
  -framework AppKit \
  -framework Foundation \
  -framework UserNotifications \
  -parse-as-library

echo "Creating app bundle..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cp "$BUILD_DIR/$BINARY_NAME" "$APP_DIR/Contents/MacOS/$BINARY_NAME"
cp "$SRC_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"
cp "$BRAND_EXPORT/menubar-idle.png" "$APP_DIR/Contents/Resources/"
cp "$BRAND_EXPORT/menubar-idle@2x.png" "$APP_DIR/Contents/Resources/"
cp "$BRAND_EXPORT/menubar-active.png" "$APP_DIR/Contents/Resources/"
cp "$BRAND_EXPORT/menubar-active@2x.png" "$APP_DIR/Contents/Resources/"

if [ -f "$BRAND_EXPORT/Prism.icns" ]; then
  cp "$BRAND_EXPORT/Prism.icns" "$APP_DIR/Contents/Resources/AppIcon.icns"
  /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$APP_DIR/Contents/Info.plist" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile AppIcon" "$APP_DIR/Contents/Info.plist"
fi

/usr/libexec/PlistBuddy -c "Add :CFBundleExecutable string $BINARY_NAME" "$APP_DIR/Contents/Info.plist" 2>/dev/null || true

echo "Code signing app (ad-hoc)..."
codesign --force --deep --sign - "$APP_DIR"
codesign --verify --verbose "$APP_DIR"

# Compatibility symlink for older setup script path
rm -rf "$BUILD_DIR/Dataset Analysis MCP.app"
ln -s "$APP_NAME" "$BUILD_DIR/Dataset Analysis MCP.app"

echo "Built $APP_DIR"
