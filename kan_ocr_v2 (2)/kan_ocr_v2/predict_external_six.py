import argparse
import csv
import os

import torch
from PIL import Image
from torchvision import transforms

from external_preprocess import preprocess_external_image
from model import MalayalamOCRNet


DEFAULT_INPUT_DIR = r"C:\Users\angel\Downloads\kan_ocr_v2\malayalam_letters_1_6"
DEFAULT_CHECKPOINT = r"C:\Users\angel\Downloads\kan_ocr_v2\external_six_vowels_model.pth"
DEFAULT_OUTPUT = r"C:\Users\angel\Downloads\kan_ocr_v2\external_six_predictions.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Predict among the six externally labeled Malayalam vowels."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(
        os.path.join(args.input_dir, filename)
        for filename in os.listdir(args.input_dir)
        if os.path.splitext(filename)[1].lower() in extensions
    )
    if not image_paths:
        raise RuntimeError(f"No supported image files found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    characters = checkpoint["characters"]
    img_size = checkpoint["image_size"]
    model = MalayalamOCRNet(num_classes=len(characters), head_type="linear").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    tensor_transform = transforms.ToTensor()

    results = []
    with torch.no_grad():
        for path in image_paths:
            with Image.open(path) as image:
                normalized = preprocess_external_image(image, img_size=img_size)
            tensor = tensor_transform(normalized).unsqueeze(0).to(device)
            probabilities = model(tensor).softmax(dim=1)[0]
            top_indices = probabilities.topk(min(3, len(characters))).indices.tolist()
            row = {
                "image": path,
                "predicted_character": characters[top_indices[0]],
                "confidence": f"{float(probabilities[top_indices[0]]):.6f}",
            }
            for rank, index in enumerate(top_indices, start=1):
                row[f"alternative_{rank}"] = characters[index]
                row[f"alternative_{rank}_confidence"] = f"{float(probabilities[index]):.6f}"
            results.append(row)

    fieldnames = [
        "image",
        "predicted_character",
        "confidence",
        "alternative_1",
        "alternative_1_confidence",
        "alternative_2",
        "alternative_2_confidence",
        "alternative_3",
        "alternative_3_confidence",
    ]
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to: {args.output}")
    print("This model recognizes only the six classes used for fine-tuning.")


if __name__ == "__main__":
    main()
