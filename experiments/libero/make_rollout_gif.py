#!/usr/bin/env python3
"""Build a compact GIF preview from rollout keyframes."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def collect_frames(patterns: list[str], max_frames: int) -> list[Path]:
    frames: list[Path] = []
    for pattern in patterns:
        frames.extend(Path(item) for item in sorted(glob.glob(pattern)))
    frames = [path for path in frames if path.is_file()]
    if max_frames > 0 and len(frames) > max_frames:
        step = (len(frames) - 1) / (max_frames - 1)
        frames = [frames[round(index * step)] for index in range(max_frames)]
    return frames


def draw_label(image: Image.Image, label: str, label_height: int) -> Image.Image:
    if not label:
        return image
    canvas = Image.new("RGB", (image.width, image.height + label_height), (250, 250, 248))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = font(max(12, label_height - 12))
    bbox = draw.textbbox((0, 0), label, font=label_font)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) // 2, image.height + 6), label, fill=(25, 25, 25), font=label_font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--duration-ms", type=int, default=130)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--label", default="")
    parser.add_argument("--label-height", type=int, default=34)
    args = parser.parse_args()

    frame_paths = collect_frames(args.frames, args.max_frames)
    if not frame_paths:
        raise SystemExit("No input frames found.")

    rendered = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        height = max(1, round(image.height * args.width / image.width))
        image = ImageOps.fit(image, (args.width, height), method=Image.Resampling.LANCZOS)
        image = draw_label(image, args.label, args.label_height)
        rendered.append(image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(
        output,
        save_all=True,
        append_images=rendered[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(str(output))


if __name__ == "__main__":
    main()
