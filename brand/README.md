# Prism Brand Assets

Production logo: **`source/prism-logo.png`** (user-provided pyramid mark)

## Source

The official Prism logo is the navy pyramid with coral accent block and star, on a cream background.

## Export

```bash
python3 -m venv .venv-brand
.venv-brand/bin/pip install pillow
.venv-brand/bin/python brand/export_assets.py
```

This generates:

- `export/prism-logo.png` — full source export
- `export/prism-logo-transparent.png` — transparent background for menu bar
- `export/menubar-*.png` — menu bar idle/active icons
- `export/prism-app-1024.png` + `Prism.icns` — macOS app icon
- `website/public/prism-logo.png` — website + README
- `website/public/favicon.png`, `apple-touch-icon.png`, `og-image.png`

Legacy SVG explorations remain in `svg/` for reference only.
