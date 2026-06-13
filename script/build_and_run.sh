#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="MLXFinetuner"
BUNDLE_ID="com.nevio.MLXFinetuner"
MIN_SYSTEM_VERSION="14.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_RESOURCES="$APP_CONTENTS/Resources"
APP_BINARY="$APP_MACOS/$APP_NAME"
INFO_PLIST="$APP_CONTENTS/Info.plist"
APP_ICON_SOURCE="$ROOT_DIR/assets/AppIcon.icon/Assets/mlx-finetuner-icon.png"
APP_ICON_PACKAGE="$ROOT_DIR/assets/AppIcon.icon"
APP_ICONSET="$DIST_DIR/AppIcon.iconset"
APP_ICNS="$APP_RESOURCES/AppIcon.icns"

pkill -x "$APP_NAME" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
swift build
BUILD_BINARY="$(swift build --show-bin-path)/$APP_NAME"

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_MACOS" "$APP_RESOURCES"
cp "$BUILD_BINARY" "$APP_BINARY"
chmod +x "$APP_BINARY"
cp -R "$ROOT_DIR/Backend" "$APP_RESOURCES/Backend"
cp "$ROOT_DIR/requirements.txt" "$APP_RESOURCES/requirements.txt"

if [[ -f "$APP_ICON_SOURCE" ]]; then
  rm -rf "$APP_ICONSET"
  mkdir -p "$APP_ICONSET"
  /usr/bin/sips -z 16 16 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_16x16.png" >/dev/null
  /usr/bin/sips -z 32 32 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_16x16@2x.png" >/dev/null
  /usr/bin/sips -z 32 32 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_32x32.png" >/dev/null
  /usr/bin/sips -z 64 64 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_32x32@2x.png" >/dev/null
  /usr/bin/sips -z 128 128 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_128x128.png" >/dev/null
  /usr/bin/sips -z 256 256 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_128x128@2x.png" >/dev/null
  /usr/bin/sips -z 256 256 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_256x256.png" >/dev/null
  /usr/bin/sips -z 512 512 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_256x256@2x.png" >/dev/null
  /usr/bin/sips -z 512 512 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_512x512.png" >/dev/null
  /usr/bin/sips -z 1024 1024 "$APP_ICON_SOURCE" --out "$APP_ICONSET/icon_512x512@2x.png" >/dev/null
  /usr/bin/iconutil -c icns "$APP_ICONSET" -o "$APP_ICNS"
  cp -R "$APP_ICON_PACKAGE" "$APP_RESOURCES/AppIcon.icon"
fi

if [[ -n "${MLX_FINETUNER_PYTHON:-}" ]]; then
  PYTHON_PATH="$MLX_FINETUNER_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_PATH="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_PATH="$(command -v python3)"
fi
cat >"$APP_RESOURCES/runtime.json" <<JSON
{"python":"$PYTHON_PATH"}
JSON

cat >"$INFO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIconName</key>
  <string>AppIcon</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MIN_SYSTEM_VERSION</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
PLIST

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

case "$MODE" in
  --build-only|build)
    ;;
  run)
    open_app
    ;;
  --debug|debug)
    lldb -- "$APP_BINARY"
    ;;
  --logs|logs)
    open_app
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    open_app
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"$BUNDLE_ID\""
    ;;
  --verify|verify)
    open_app
    sleep 1
    pgrep -x "$APP_NAME" >/dev/null
    ;;
  *)
    echo "usage: $0 [run|--build-only|--debug|--logs|--telemetry|--verify]" >&2
    exit 2
    ;;
esac
