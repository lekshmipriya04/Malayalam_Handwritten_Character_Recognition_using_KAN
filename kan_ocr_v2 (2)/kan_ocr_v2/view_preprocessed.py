"""
View preprocessed Malayalam character images correctly.

Your preprocessing saved images with pixel values {0,1}. Normal image
viewers treat both values as almost black, so the character looks blank.

This script converts:
    0 -> 0   (black)
    1 -> 255 (white)

Usage:
    python view_preprocessed.py --image "output\test\26\7.png"

You can also save a normal PNG that can be opened in Windows:
    python view_preprocessed.py --image "output\test\26\7.png" --save viewed.png
"""

import argparse
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}


def load_binary_image(image_path):
    """Load a {0,1} preprocessed image and convert it to visible 0/255."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    print("Original image information:")
    print("  Size:", img.size)
    print("  Minimum pixel value:", arr.min())
    print("  Maximum pixel value:", arr.max())
    print("  Unique pixel values:", np.unique(arr)[:20])

    # Your preprocessing stores the foreground/background as 0 and 1.
    # Scale 0/1 to 0/255 so that ordinary viewers can display it.
    if arr.max() <= 1:
        visible = (arr * 255).astype(np.uint8)
    else:
        # Fallback if the image is already a normal 0-255 image.
        visible = arr.astype(np.uint8)

    return visible


def main():
    parser = argparse.ArgumentParser(
        description="View a preprocessed {0,1} Malayalam character image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the preprocessed PNG image."
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output PNG path, e.g. viewed.png"
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert black/white if the displayed character is reversed."
    )
    parser.add_argument(
        "--foreground-color",
        choices=COLORS,
        default="white",
        help="Display color for the character strokes. Visualization only."
    )
    parser.add_argument(
        "--background-color",
        choices=COLORS,
        default="black",
        help="Display color for the background. Visualization only."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    visible = load_binary_image(args.image)

    if args.invert:
        visible = 255 - visible

    foreground_mask = visible > 0
    displayed = np.empty((*visible.shape, 3), dtype=np.uint8)
    displayed[:] = COLORS[args.background_color]
    displayed[foreground_mask] = COLORS[args.foreground_color]

    # Save a normal PNG if requested.
    if args.save:
        Image.fromarray(displayed, mode="RGB").save(args.save)
        print(f"\nVisible image saved to: {args.save}")

    # Display with nearest-neighbour interpolation so the 64x64 pixels
    # remain clear rather than being blurred.
    plt.figure(figsize=(6, 6))
    plt.imshow(displayed, interpolation="nearest")
    plt.title(
        f"Preprocessed image: {os.path.basename(args.image)}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
