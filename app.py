import io
from functools import lru_cache
from pathlib import Path
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from PIL import Image, ImageOps

from model import MalayalamOCRNet


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "final_best_model.pth"
CLASS_DIR = APP_DIR / "dataset" / "train"
IMAGE_SIZE = 64
TOP_K = 5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CLASS_CHARACTERS = (
    "ആ",  # 0
    "ഇ",  # 1
    "ഉ",  # 2
    "ഋ",  # 3
    "എ",  # 4
    "ഒ",  # 5
    "ക",  # 6
    "ക്ക",  # 7
    "ക്ത",  # 8
    "ക്ല",  # 9
    "ക്ഷ",  # 10
    "ഖ",  # 11
    "ഗ",  # 12
    "ഗ്ഗ",  # 13
    "ഗ്ന",  # 14
    "ഗ്മ",  # 15
    "ഗ്ല",  # 16
    "ഘ",  # 17
    "ങ",  # 18
    "ങ്ക",  # 19
    "ങ്ങ",  # 20
    "ച",  # 21
    "ച്ച",  # 22
    "ച്ഛ",  # 23
    "ഛ",  # 24
    "ജ",  # 25
    "ജ്ജ",  # 26
    "ജ്ഞ",  # 27
    "ഝ",  # 28
    "ഞ",  # 29
    "ഞ്ച",  # 30
    "ഞ്ഞ",  # 31
    "ട",  # 32
    "ട്ട",  # 33
    "ഠ",  # 34
    "ഡ",  # 35
    "ഡ്ഡ",  # 36
    "ഢ",  # 37
    "ണ",  # 38
    "ണ്ട",  # 39
    "ണ്ഡ",  # 40
    "ണ്ണ",  # 41
    "ണ്മ",  # 42
    "ണ്‍",  # 43
    "ത",  # 44
    "ത്ത",  # 45
    "ത്ഭ",  # 46
    "ത്മ",  # 47
    "ത്സ",  # 48
    "ദ",  # 49
    "ദ്ദ",  # 50
    "ദ്ധ",  # 51
    "ധ",  # 52
    "ന",  # 53
    "ന്ത",  # 54
    "ന്ദ",  # 55
    "ന്ധ",  # 56
    "ന്ന",  # 57
    "ന്മ",  # 58
    "ന്‍",  # 59
    "ന്‍റ",  # 60
    "പ",  # 61
    "പ്പ",  # 62
    "പ്ല",  # 63
    "ഫ",  # 64
    "ഫ്ല",  # 65
    "ബ",  # 66
    "ബ്ദ",  # 67
    "ബ്ധ",  # 68
    "ബ്ബ",  # 69
    "ബ്ല",  # 70
    "ഭ",  # 71
    "മ",  # 72
    "മ്പ",  # 73
    "മ്മ",  # 74
    "മ്ല",  # 75
    "യ",  # 76
    "യ്യ",  # 77
    "ര",  # 78
    "ര്‍",  # 79
    "റ",  # 80
    "റ്റ",  # 81
    "ല",  # 82
    "ല്ല",  # 83
    "ല്‍",  # 84
    "ള",  # 85
    "ള്ള",  # 86
    "ള്‍",  # 87
    "ഴ",  # 88
    "വ",  # 89
    "വ്വ",  # 90
    "ശ",  # 91
    "ശ്ച",  # 92
    "ശ്ല",  # 93
    "ശ്ശ",  # 94
    "ഷ",  # 95
    "ഷ്ട",  # 96
    "സ",  # 97
    "സ്ഥ",  # 98
    "സ്ധ",  # 99
    "സ്റ്റ",  # 100
    "സ്ല",  # 101
    "സ്സ",  # 102
    "ഹ",  # 103
    "ഹ്ന",  # 104
    "ഹ്മ",  # 105
    "ഹ്ല",  # 106
    "ാ",  # 107
    "ി",  # 108
    "ീ",  # 109
    "ു",  # 110
    "ൂ",  # 111
    "ൃ",  # 112
    "െ",  # 113
    "്",  # 114
    "്യ",  # 115
    "്ര",  # 116
    "്വ",  # 117
    "ൗ",  # 118
)


def _otsu_threshold(image: np.ndarray) -> int:
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = image.size
    intensity_sum = np.dot(np.arange(256), histogram)
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    threshold = 0

    for value in range(256):
        background_weight += histogram[value]
        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_sum += value * histogram[value]
        background_mean = background_sum / background_weight
        foreground_mean = (intensity_sum - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (
            background_mean - foreground_mean
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = value

    return threshold


def preprocess_image(
    image: Image.Image, already_preprocessed: bool = False
) -> tuple[torch.Tensor, Image.Image, str]:
    """Convert dataset images or ordinary uploads to the model's 1x64x64 input."""
    grayscale = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(grayscale)

    if already_preprocessed:
        resized = grayscale.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        mode = "Training-style image: grayscale and resized to 64×64"
    else:
        border_size = max(1, min(grayscale.size) // 20)
        border = np.concatenate(
            (
                pixels[:border_size, :].ravel(),
                pixels[-border_size:, :].ravel(),
                pixels[:, :border_size].ravel(),
                pixels[:, -border_size:].ravel(),
            )
        )
        if np.median(border) > 127:
            pixels = 255 - pixels

        threshold = _otsu_threshold(pixels)
        foreground = pixels > threshold
        rows, columns = np.where(foreground)
        if rows.size == 0:
            raise ValueError(
                "No character was detected. Upload an image with a clearly "
                "visible character."
            )

        left, right = columns.min(), columns.max() + 1
        top, bottom = rows.min(), rows.max() + 1
        character = Image.fromarray(foreground.astype(np.uint8) * 255).crop(
            (left, top, right, bottom)
        )

        width, height = character.size
        padding = max(2, round(max(width, height) * 0.15))
        side = max(width, height) + 2 * padding
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(character, ((side - width) // 2, (side - height) // 2))
        resized = canvas.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        mode = "External image: grayscale, polarity-corrected, binarized, cropped and padded"

    resized_pixels = np.asarray(resized, dtype=np.float32)
    # Match RescaleBinaryPNG used during training: every nonzero resized pixel
    # is clamped to 1, rather than retaining antialiased gray values.
    resized_pixels = (resized_pixels > 0).astype(np.float32)
    resized = Image.fromarray((resized_pixels * 255).astype(np.uint8))
    tensor = torch.from_numpy(resized_pixels.copy()).unsqueeze(0)
    return tensor, resized, mode


@lru_cache(maxsize=1)
def load_model(
    _checkpoint_mtime_ns: int, _checkpoint_size: int
) -> tuple[MalayalamOCRNet, list[str], torch.device]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not CLASS_DIR.is_dir():
        raise FileNotFoundError(f"Training class folders not found: {CLASS_DIR}")

    classes = sorted(path.name for path in CLASS_DIR.iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found in: {CLASS_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MalayalamOCRNet(num_classes=len(classes))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, classes, device


def predict(
    model: MalayalamOCRNet,
    classes: list[str],
    device: torch.device,
    image_tensor: torch.Tensor,
) -> list[tuple[str, float]]:
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        count = min(TOP_K, len(classes))
        scores, indices = probabilities.topk(count)
    predictions = []
    for index, score in zip(indices.cpu().tolist(), scores.cpu().tolist()):
        try:
            class_id = int(classes[index])
            if class_id < 0:
                raise IndexError
            character = CLASS_CHARACTERS[class_id]
        except (ValueError, IndexError) as error:
            raise ValueError(
                f"No character mapping exists for model class {classes[index]!r}."
            ) from error
        predictions.append((character, float(score)))
    return predictions


def predict_zip(
    archive_bytes: bytes,
    model: MalayalamOCRNet,
    classes: list[str],
    device: torch.device,
    already_preprocessed: bool = False,
) -> list[dict[str, str | float | None]]:
    """Predict the leading character for every image in a ZIP archive."""
    results: list[dict[str, str | float | None]] = []
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        image_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and PurePosixPath(member.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_members:
            raise ValueError("The ZIP file does not contain any supported images.")

        for member in image_members:
            try:
                with archive.open(member) as image_file:
                    with Image.open(image_file) as image:
                        image.load()
                        input_tensor, _, _ = preprocess_image(
                            image, already_preprocessed=already_preprocessed
                        )
                prediction = predict(model, classes, device, input_tensor)[0]
                results.append(
                    {
                        "Image": member.filename,
                        "Predicted character": prediction[0],
                        "Confidence": prediction[1],
                        "Status": "OK",
                    }
                )
            except (BadZipFile, OSError, ValueError, RuntimeError) as error:
                results.append(
                    {
                        "Image": member.filename,
                        "Predicted character": "",
                        "Confidence": None,
                        "Status": f"Error: {error}",
                    }
                )
    return results
