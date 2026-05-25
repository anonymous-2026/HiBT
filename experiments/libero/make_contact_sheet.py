#!/usr/bin/env python3
"""Build a simple contact sheet from rollout keyframes."""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def collect_frames(patterns: list[str]) -> list[Path]:
    frames: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        frames.extend(Path(item) for item in matches)
    return [path for path in frames if path.is_file()]


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--label-height", type=int, default=28)
    args = parser.parse_args()

    frames = collect_frames(args.frames)
    if not frames:
        raise SystemExit("No input frames found.")

    cols = max(1, args.cols)
    rows = math.ceil(len(frames) / cols)
    width = cols * args.tile_size + (cols - 1) * args.gap
    cell_h = args.tile_size + args.label_height
    height = rows * cell_h + (rows - 1) * args.gap
    sheet = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(sheet)
    label_font = font(max(12, args.label_height - 10))

    for index, frame_path in enumerate(frames):
        row = index // cols
        col = index % cols
        x = col * (args.tile_size + args.gap)
        y = row * (cell_h + args.gap)
        image = Image.open(frame_path).convert("RGB")
        image = ImageOps.fit(image, (args.tile_size, args.tile_size), method=Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        label = frame_path.stem
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + max(0, (args.tile_size - (bbox[2] - bbox[0])) // 2), y + args.tile_size + 5), label, fill=(30, 30, 30), font=label_font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(str(output))


if __name__ == "__main__":
    main()
