import os

import numpy as np
from PIL import Image


INPUT_DIR = r"C:\Users\angel\Downloads\kan_ocr_v2\malayalam_letters_1_6"
OUTPUT_DIR = r"C:\Users\angel\Downloads\kan_ocr_v2\malayalam_characters_1_6_preprocessed_2"
IMG_SIZE = 64


def preprocess_external_image(image, img_size=IMG_SIZE, red_threshold=40, padding=0.12):
    """Extract colored handwriting and normalize it to a white-on-black glyph."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb.transpose(2, 0, 1)
    red_dominance = red - np.maximum(green, blue)
    foreground = (red >= 80) & (red_dominance >= red_threshold)

    ys, xs = np.where(foreground)
    if len(xs) == 0:
        # Support blue and other dark colored pens on light paper. Ignore the
        # very top strip, where notebook covers or background objects often appear.
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        foreground = (chroma >= 20) & (rgb.min(axis=2) < 190)
        foreground[: max(1, round(foreground.shape[0] * 0.08)), :] = False
        ys, xs = np.where(foreground)
        if len(xs) == 0:
            raise ValueError(
                "No colored handwriting found. Use a clear image with colored ink "
                "on a light background."
            )

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    width, height = x_max - x_min, y_max - y_min
    pad = max(2, round(max(width, height) * padding))
    x_min, x_max = max(0, x_min - pad), min(foreground.shape[1], x_max + pad)
    y_min, y_max = max(0, y_min - pad), min(foreground.shape[0], y_max + pad)

    glyph = Image.fromarray(foreground[y_min:y_max, x_min:x_max].astype(np.uint8) * 255)

    # Preserve the handwriting's proportions and center it like the square training samples.
    scale = (img_size - 8) / max(glyph.size)
    resized_size = (
        max(1, round(glyph.width * scale)),
        max(1, round(glyph.height * scale)),
    )
    glyph = glyph.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (img_size, img_size), 0)
    canvas.paste(glyph, ((img_size - glyph.width) // 2, (img_size - glyph.height) // 2))
    return canvas


def preprocess_image(input_path, output_path):
    with Image.open(input_path) as image:
        processed = preprocess_external_image(image)
    processed.save(output_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted(
        name for name in os.listdir(INPUT_DIR)
        if name.lower().endswith(valid_extensions)
    )
    if not files:
        raise RuntimeError(f"No image files found in: {INPUT_DIR}")

    print(f"Input folder : {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Images found : {len(files)}")

    for index, filename in enumerate(files, start=1):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        preprocess_image(input_path, output_path)
        print(f"[{index}/{len(files)}] Processed: {filename}")


if __name__ == "__main__":
    main()
