import argparse
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from external_preprocess import preprocess_external_image
from model import MalayalamOCRNet


DEFAULT_INPUT_DIR = r"C:\Users\angel\Downloads\kan_ocr_v2\malayalam_letters_1_6"
DEFAULT_CHECKPOINT = r"C:\Users\angel\Downloads\kan_ocr_v2\final_best_model.pth"
DEFAULT_OUTPUT = r"C:\Users\angel\Downloads\kan_ocr_v2\external_six_vowels_model.pth"
CHARACTERS = ["അ", "ആ", "ഇ", "ഈ", "ഉ", "ഊ"]
IMAGE_SIZE = 64


class ExternalCharacterDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.images = []

        for path in image_paths:
            with Image.open(path) as image:
                self.images.append(preprocess_external_image(image, img_size=IMAGE_SIZE))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), index


def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train a six-character model for the provided, ordered vowel photos. "
            "This task-specific checkpoint is not a replacement for the 119-class model."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = get_arguments()
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < len(CHARACTERS):
        raise ValueError(
            "epochs and steps-per-epoch must be positive and batch-size must be at least 6."
        )

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(
        os.path.join(args.input_dir, name)
        for name in os.listdir(args.input_dir)
        if os.path.splitext(name)[1].lower() in image_extensions
    )
    if len(image_paths) != len(CHARACTERS):
        raise RuntimeError(
            f"Expected six images ordered as {', '.join(CHARACTERS)}, found "
            f"{len(image_paths)} in {args.input_dir}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MalayalamOCRNet(num_classes=len(CHARACTERS), head_type="linear").to(device)
    source_state = torch.load(args.checkpoint, map_location=device)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in source_state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError(f"No backbone weights found in checkpoint: {args.checkpoint}")
    model.backbone.load_state_dict(backbone_state, strict=True)

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    augmentation = transforms.Compose([
        transforms.RandomAffine(
            degrees=7,
            translate=(0.06, 0.06),
            scale=(0.9, 1.1),
            shear=4,
            fill=0,
        ),
        transforms.ToTensor(),
    ])
    training_data = ExternalCharacterDataset(image_paths, augmentation)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-3,
    )
    rng = random.Random(42)

    for epoch in range(args.epochs):
        model.train()
        model.backbone.eval()
        total_loss = 0.0

        for _ in range(args.steps_per_epoch):
            batch = [
                training_data[rng.randrange(len(training_data))]
                for _ in range(args.batch_size)
            ]
            images = torch.stack([item[0] for item in batch]).to(device)
            labels = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"loss={total_loss / args.steps_per_epoch:.4f}"
            )

    model.eval()
    predictions = []
    with torch.no_grad():
        for path in image_paths:
            with Image.open(path) as image:
                sample = preprocess_external_image(image, img_size=IMAGE_SIZE)
            tensor = transforms.ToTensor()(sample).unsqueeze(0).to(device)
            probabilities = model(tensor).softmax(dim=1)[0]
            predictions.append(int(probabilities.argmax().item()))

    matches = sum(prediction == expected for expected, prediction in enumerate(predictions))
    for expected_index, (path, predicted_index) in enumerate(zip(image_paths, predictions)):
        print(
            f"{os.path.basename(path)}: expected class {expected_index}, "
            f"predicted class {predicted_index}"
        )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "characters": CHARACTERS,
            "image_size": IMAGE_SIZE,
        },
        args.output,
    )
    print(
        f"Matched {matches}/{len(CHARACTERS)} supplied photos "
        "(these are fine-tuning examples, not held-out accuracy)."
    )
    print(f"Six-character checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
