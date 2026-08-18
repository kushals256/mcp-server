#!/usr/bin/env python3
"""Export Prism brand assets from the provided source logo PNG."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "prism-logo.png"
EXPORT_DIR = ROOT / "export"
WEBSITE_PUBLIC = ROOT.parent / "website" / "public"
MACOS_ASSETS = ROOT.parent / "macos" / "DatasetAnalysisMCP" / "Assets.xcassets"


def load_image():
    from PIL import Image

    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source logo: {SOURCE}")
    return Image.open(SOURCE).convert("RGBA")


def remove_cream_background(image):
    from PIL import Image

    pixels = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r > 220 and g > 210 and b > 180 and abs(int(r) - int(g)) < 25:
                pixels[x, y] = (r, g, b, 0)
    return image


def resize(image, size: int):
    from PIL import Image

    return image.resize((size, size), Image.Resampling.LANCZOS)


def save_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def create_og_image(logo):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGBA", (1200, 630), (17, 17, 20, 255))
    draw = ImageDraw.Draw(canvas)

    for center, color in [((300, 180), (94, 43, 151, 80)), ((900, 120), (184, 228, 71, 45))]:
        for radius in range(280, 0, -4):
            alpha = max(0, int(color[3] * (radius / 280)))
            draw.ellipse(
                (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                fill=(color[0], color[1], color[2], alpha),
            )

    mark = resize(logo, 300)
    canvas.alpha_composite(mark, (110, 165))

    draw.text((520, 220), "Prism", fill=(255, 255, 255, 245))
    draw.text((520, 290), "Dataset Analysis MCP for Mac", fill=(255, 255, 255, 170))
    draw.text((520, 350), "Turn Claude into your data scientist", fill=(255, 255, 255, 115))

    save_png(canvas, WEBSITE_PUBLIC / "og-image.png")


def create_app_icon(logo):
    from PIL import Image, ImageDraw

    size = 1024
    mark = resize(logo, size)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size, size), radius=224, fill=255)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(mark, (0, 0), mask)
    return canvas


def write_imageset(name: str, files: dict[str, str]) -> None:
    imageset = MACOS_ASSETS / f"{name}.imageset"
    imageset.mkdir(parents=True, exist_ok=True)
    images = [{"filename": filename, "idiom": "mac", "scale": scale} for filename, scale in files.items()]
    (imageset / "Contents.json").write_text(
        json.dumps({"images": images, "info": {"author": "xcode", "version": 1}}, indent=2),
        encoding="utf-8",
    )


def write_appiconset(icon_1024: Path) -> None:
    from PIL import Image

    appiconset = MACOS_ASSETS / "AppIcon.appiconset"
    appiconset.mkdir(parents=True, exist_ok=True)
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for size in sizes:
        img = Image.open(icon_1024).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img.save(appiconset / f"icon_{size}.png")

    images = [
        {"size": "16x16", "idiom": "mac", "filename": "icon_16.png", "scale": "1x"},
        {"size": "16x16", "idiom": "mac", "filename": "icon_32.png", "scale": "2x"},
        {"size": "32x32", "idiom": "mac", "filename": "icon_32.png", "scale": "1x"},
        {"size": "32x32", "idiom": "mac", "filename": "icon_64.png", "scale": "2x"},
        {"size": "128x128", "idiom": "mac", "filename": "icon_128.png", "scale": "1x"},
        {"size": "128x128", "idiom": "mac", "filename": "icon_256.png", "scale": "2x"},
        {"size": "256x256", "idiom": "mac", "filename": "icon_256.png", "scale": "1x"},
        {"size": "256x256", "idiom": "mac", "filename": "icon_512.png", "scale": "2x"},
        {"size": "512x512", "idiom": "mac", "filename": "icon_512.png", "scale": "1x"},
        {"size": "512x512", "idiom": "mac", "filename": "icon_1024.png", "scale": "2x"},
    ]
    (appiconset / "Contents.json").write_text(
        json.dumps({"images": images, "info": {"author": "xcode", "version": 1}}, indent=2),
        encoding="utf-8",
    )


def build_icns(png_1024: Path, icns_path: Path) -> None:
    iconset = EXPORT_DIR / "AppIcon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)

    from PIL import Image

    mapping = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    base = Image.open(png_1024).convert("RGBA")
    for name, size in mapping.items():
        base.resize((size, size), Image.Resampling.LANCZOS).save(iconset / name)

    if shutil.which("iconutil"):
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)], check=True)
    else:
        shutil.copy2(png_1024, icns_path.with_suffix(".png"))


def main() -> int:
    logo = load_image()
    logo_opaque = logo.copy()
    logo_transparent = remove_cream_background(logo.copy())

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    WEBSITE_PUBLIC.mkdir(parents=True, exist_ok=True)

    save_png(logo_opaque, EXPORT_DIR / "prism-logo.png")
    save_png(logo_transparent, EXPORT_DIR / "prism-logo-transparent.png")
    save_png(resize(logo_opaque, 512), EXPORT_DIR / "prism-logo-512.png")
    save_png(resize(logo_opaque, 512), WEBSITE_PUBLIC / "prism-logo.png")
    save_png(resize(logo_opaque, 1024), WEBSITE_PUBLIC / "prism-logo-hero.png")
    save_png(resize(logo_opaque, 180), WEBSITE_PUBLIC / "apple-touch-icon.png")
    save_png(resize(logo_opaque, 32), WEBSITE_PUBLIC / "favicon.png")

    shutil.copy2(SOURCE, WEBSITE_PUBLIC / "prism-logo-source.png")

    save_png(resize(logo_opaque, 22), EXPORT_DIR / "menubar-idle.png")
    save_png(resize(logo_opaque, 44), EXPORT_DIR / "menubar-idle@2x.png")
    save_png(resize(logo_opaque, 22), EXPORT_DIR / "menubar-active.png")
    save_png(resize(logo_opaque, 44), EXPORT_DIR / "menubar-active@2x.png")

    app_icon = create_app_icon(logo_opaque)
    save_png(app_icon, EXPORT_DIR / "prism-app-1024.png")
    create_og_image(logo_opaque)

    MACOS_ASSETS.mkdir(parents=True, exist_ok=True)
    menubar_dir = MACOS_ASSETS / "MenuBarIcon.imageset"
    menubar_dir.mkdir(parents=True, exist_ok=True)
    for name in ["menubar-idle.png", "menubar-idle@2x.png", "menubar-active.png", "menubar-active@2x.png"]:
        shutil.copy2(EXPORT_DIR / name, menubar_dir / name)

    write_imageset("MenuBarIcon", {"menubar-idle.png": "1x", "menubar-idle@2x.png": "2x"})
    write_imageset("MenuBarIconActive", {"menubar-active.png": "1x", "menubar-active@2x.png": "2x"})
    write_appiconset(EXPORT_DIR / "prism-app-1024.png")
    build_icns(EXPORT_DIR / "prism-app-1024.png", EXPORT_DIR / "Prism.icns")

    print("Exported Prism logo assets from source image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
