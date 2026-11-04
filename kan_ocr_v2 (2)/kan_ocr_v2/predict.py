import argparse
import os
import csv
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import datasets, transforms

from dataset import RescaleBinaryPNG, DATA_ROOT
from external_preprocess import preprocess_external_image
from model import MalayalamOCRNet


# ============================================================
# CLASS INDEX -> MALAYALAM CHARACTER
# ============================================================

CLASS_INDEX_TO_MALAYALAM = [
    "ആ",       # 0
    "ഇ",       # 1
    "ഉ",       # 2
    "ഋ",       # 3
    "എ",       # 4
    "ഒ",       # 5
    "ക",       # 6
    "ക്ക",      # 7
    "ക്ത",      # 8
    "ക്ല",      # 9
    "ക്ഷ",      # 10
    "ഖ",       # 11
    "ഗ",       # 12
    "ഗ്ഗ",      # 13
    "ഗ്ന",      # 14
    "ഗ്മ",      # 15
    "ഗ്ല",      # 16
    "ഘ",       # 17
    "ങ",       # 18
    "ങ്ക",      # 19
    "ങ്ങ",      # 20
    "ച",       # 21
    "ച്ച",      # 22
    "ച്ഛ",      # 23
    "ഛ",       # 24
    "ജ",       # 25
    "ജ്ജ",      # 26
    "ജ്ഞ",      # 27
    "ഝ",       # 28
    "ഞ",       # 29
    "ഞ്ച",      # 30
    "ഞ്ഞ",      # 31
    "ട",       # 32
    "ട്ട",      # 33
    "ഠ",       # 34
    "ഡ",       # 35
    "ഡ്ഡ",      # 36
    "ഢ",       # 37
    "ണ",       # 38
    "ണ്ട",      # 39
    "ണ്ഡ",      # 40
    "ണ്ണ",      # 41
    "ണ്മ",      # 42
    "ണ്‍",      # 43
    "ത",       # 44
    "ത്ത",      # 45
    "ത്ഭ",      # 46
    "ത്മ",      # 47
    "ത്സ",      # 48
    "ദ",       # 49
    "ദ്ദ",      # 50
    "ദ്ധ",      # 51
    "ധ",       # 52
    "ന",       # 53
    "ന്ത",      # 54
    "ന്ദ",      # 55
    "ന്ധ",      # 56
    "ന്ന",      # 57
    "ന്മ",      # 58
    "ന്‍",      # 59
    "ന്‍റ",     # 60
    "പ",       # 61
    "പ്പ",      # 62
    "പ്ല",      # 63
    "ഫ",       # 64
    "ഫ്ല",      # 65
    "ബ",       # 66
    "ബ്ദ",      # 67
    "ബ്ധ",      # 68
    "ബ്ബ",      # 69
    "ബ്ല",      # 70
    "ഭ",       # 71
    "മ",       # 72
    "മ്പ",      # 73
    "മ്മ",      # 74
    "മ്ല",      # 75
    "യ",       # 76
    "യ്യ",      # 77
    "ര",       # 78
    "ര്‍",      # 79
    "റ",       # 80
    "റ്റ",      # 81
    "ല",       # 82
    "ല്ല",      # 83
    "ല്‍",      # 84
    "ള",       # 85
    "ള്ള",      # 86
    "ള്‍",      # 87
    "ഴ",       # 88
    "വ",       # 89
    "വ്വ",      # 90
    "ശ",       # 91
    "ശ്ച",      # 92
    "ശ്ല",      # 93
    "ശ്ശ",      # 94
    "ഷ",       # 95
    "ഷ്ട",      # 96
    "സ",       # 97
    "സ്ഥ",      # 98
    "സ്ധ",      # 99
    "സ്റ്റ",      # 100
    "സ്ല",      # 101
    "സ്സ",      # 102
    "ഹ",       # 103
    "ഹ്ന",      # 104
    "ഹ്മ",      # 105
    "ഹ്ല",      # 106
    "ാ",       # 107
    "ി",       # 108
    "ീ",       # 109
    "ു",       # 110
    "ൂ",       # 111
    "ൃ",       # 112
    "െ",       # 113
    "്",       # 114
    "്യ",       # 115
    "്ര",       # 116
    "്വ",       # 117
    "ൗ",       # 118
]


NUM_CLASSES = 119


if len(CLASS_INDEX_TO_MALAYALAM) != NUM_CLASSES:
    raise RuntimeError(
        "CLASS_INDEX_TO_MALAYALAM must contain exactly "
        f"{NUM_CLASSES} entries."
    )

def character_for_folder(folder_name):
    try:
        character_index = int(folder_name)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric class folder, got {folder_name!r}.") from exc

    if not 0 <= character_index < NUM_CLASSES:
        raise ValueError(f"Class folder index is out of range: {folder_name!r}.")

    return CLASS_INDEX_TO_MALAYALAM[character_index]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================


def build_inference_transform(img_size=64, raw_binary_png=False):
    """
    Build transform for inference.

    For external images:
        - grayscale
        - resize to 64x64
        - convert to tensor

    Raw external photographs are converted to the training polarity and cropped
    by preprocess_external_image() before this transform is applied.
    """

    transform_list = [
        transforms.Grayscale(num_output_channels=1),
    ]

    transform_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    if raw_binary_png:
        transform_list.append(RescaleBinaryPNG())

    return transforms.Compose(transform_list)

# ============================================================
# UNLABELLED EXTERNAL IMAGE DATASET
# ============================================================

class UnlabelledImageDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_dir,
        transform=None,
        preprocess_raw=True,
        img_size=64,
    ):

        self.image_dir = image_dir
        self.transform = transform
        self.preprocess_raw = preprocess_raw
        self.img_size = img_size

        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        if os.path.isfile(image_dir):
            if os.path.splitext(image_dir)[1].lower() not in image_extensions:
                raise ValueError(
                    f"Unsupported image format: {image_dir}\n"
                    "Supported formats: .png, .jpg, .jpeg, .bmp, .tif, .tiff"
                )
            self.samples = [image_dir]
        elif os.path.isdir(image_dir):
            self.samples = sorted([
                os.path.join(image_dir, filename)
                for filename in os.listdir(image_dir)
                if (
                    os.path.isfile(os.path.join(image_dir, filename))
                    and os.path.splitext(filename)[1].lower() in image_extensions
                )
            ])
        else:
            raise FileNotFoundError(
                f"Input image or directory not found: {image_dir}"
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image files found in:\n{image_dir}\n\n"
                "Supported formats:\n"
                ".png, .jpg, .jpeg, .bmp, .tif, .tiff"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path = self.samples[index]

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not open image:\n{image_path}\n"
                f"Error: {e}"
            )

        if not self.preprocess_raw and image.convert("L").getbbox() is None:
            raise ValueError(
                f"Preprocessed image is completely black:\n{image_path}\n"
                "Regenerate it from the original photograph with "
                "external_preprocess.py."
            )

        # ----------------------------------------------------
        # RAW EXTERNAL IMAGE
        # ----------------------------------------------------

        if self.preprocess_raw:
            image = preprocess_external_image(image, img_size=self.img_size)

        # ----------------------------------------------------
        # FINAL TRANSFORMATION
        # ----------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    num_classes,
    device,
    head_type="kan2",
    kan_hidden=128,
    kan_grid=8,
    kan_order=3,
    dropout=0.3,
):

    model = MalayalamOCRNet(
        num_classes=num_classes,
        head_type=head_type,
        kan_hidden=kan_hidden,
        kan_grid=kan_grid,
        kan_order=kan_order,
        dropout=dropout,
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# PREDICT DATASET
#
# labelled=True:
#     Original dataset/test directory
#
# labelled=False:
#     New external unlabelled images
# ============================================================

@torch.no_grad()
def predict_images(
    model,
    image_dataset,
    device,
    labelled,
    class_folders,
    batch_size=64,
    num_workers=4,
):

    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    results = []

    total = len(image_dataset)
    processed = 0

    print(f"\nPredicting {total} images...")
    print("-" * 70)

    for batch in loader:

        # ----------------------------------------------------
        # LABELLED DATASET
        # ----------------------------------------------------

        if labelled:

            images, labels = batch

        # ----------------------------------------------------
        # UNLABELLED EXTERNAL IMAGES
        # ----------------------------------------------------

        else:

            images, image_paths = batch

        # ----------------------------------------------------
        # Move images to GPU
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(images)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_probabilities, predicted_indices = (
            probabilities.max(dim=1)
        )

        batch_size_actual = images.size(0)

        # ----------------------------------------------------
        # Process predictions
        # ----------------------------------------------------

        for i in range(batch_size_actual):

            predicted_class_index = (
                predicted_indices[i].item()
            )

            confidence = (
                predicted_probabilities[i].item()
            )

            predicted_folder = class_folders[predicted_class_index]
            predicted_character = character_for_folder(predicted_folder)

            # =================================================
            # EXTERNAL UNLABELLED IMAGE
            # =================================================

            if not labelled:

                image_path = image_paths[i]

                results.append({
                    "image": image_path,
                    "predicted_class_index":
                        predicted_class_index,
                    "predicted_folder":
                        predicted_folder,
                    "predicted_character":
                        predicted_character,
                    "confidence":
                        confidence,
                })

            # =================================================
            # ORIGINAL LABELLED TEST DATASET
            # =================================================

            else:

                dataset_index = processed + i

                image_path, _ = image_dataset.samples[dataset_index]
                true_class_index = labels[i].item()

                true_folder = (
                    image_dataset.classes[
                        true_class_index
                    ]
                )

                true_character = character_for_folder(true_folder)

                results.append({
                    "image": image_path,
                    "true_folder": true_folder,
                    "true_character": true_character,
                    "predicted_class_index":
                        predicted_class_index,
                    "predicted_folder":
                        predicted_folder,
                    "predicted_character":
                        predicted_character,
                    "confidence":
                        confidence,
                })

        processed += batch_size_actual

        print(
            f"\rProcessed {processed}/{total} images",
            end=""
        )

    print("\n")

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_predictions_csv(
    results,
    output_path,
    labelled,
):

    if labelled:

        fieldnames = [
            "image",
            "true_folder",
            "true_character",
            "predicted_class_index",
            "predicted_folder",
            "predicted_character",
            "confidence",
        ]

    else:

        fieldnames = [
            "image",
            "predicted_class_index",
            "predicted_folder",
            "predicted_character",
            "confidence",
        ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:

            row = row.copy()

            row["confidence"] = (
                f"{row['confidence']:.6f}"
            )

            writer.writerow(row)

    print(
        f"\nPredictions saved to:\n{output_path}"
    )


# ============================================================
# CALCULATE ACCURACY
#
# Only used for labelled dataset/test images.
# ============================================================

def calculate_accuracy(results):

    if len(results) == 0:
        return 0.0

    correct = 0

    for row in results:

        if (
            row["true_folder"]
            == row["predicted_folder"]
        ):
            correct += 1

    return correct / len(results)


    # ============================================================
    # MAIN
    # ============================================================
def main():
    
    parser = argparse.ArgumentParser(
        description=(
            "Malayalam OCR prediction script supporting "
            "both labelled dataset test images and "
            "unlabelled external images."
        )
    )
    
    parser.add_argument(
        "--raw-binary-png",
        action="store_true",
        help=(
            "Use this if images are already in the "
            "original {0,1} PNG format."
        )
    )

    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help=(
            "Use this when external images have already been "
            "cropped, resized, normalized and prepared for the model."
        )
    )
    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained .pth checkpoint"
    )

    # --------------------------------------------------------
    # NORMAL DATASET ROOT
    # --------------------------------------------------------

    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help=(
            "Dataset root containing train/val/test. "
            "Used when --input-dir is not provided."
        )
    )

    # --------------------------------------------------------
    # EXTERNAL IMAGE DIRECTORY
    # --------------------------------------------------------

    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Path to one new unlabelled image or a directory of images. "
            "When supplied, this takes priority over "
            "--data-root/test."
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        default="test_predictions.csv",
        help="Output CSV file"
    )

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        help="Image size used by the model"
    )

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images processed at once"
    )

    # --------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )


    # --------------------------------------------------------
    # MODEL HEAD
    # --------------------------------------------------------

    parser.add_argument(
        "--head-type",
        default="kan2",
        choices=["linear", "kan1", "kan2"]
    )

    parser.add_argument(
        "--kan-hidden",
        type=int,
        default=128
    )

    parser.add_argument(
        "--kan-grid",
        type=int,
        default=8
    )

    parser.add_argument(
        "--kan-order",
        type=int,
        default=3
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MALAYALAM OCR PREDICTION")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Number of classes:",
        NUM_CLASSES
    )

    train_dir = os.path.join(args.data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training class directory not found: {train_dir}")

    class_folders = sorted(
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    )
    expected_folders = {str(index) for index in range(NUM_CLASSES)}
    if len(class_folders) != NUM_CLASSES or set(class_folders) != expected_folders:
        raise RuntimeError(
            f"Expected training folders named 0 through {NUM_CLASSES - 1} "
            f"under {train_dir}, found {len(class_folders)} folders."
        )

    # ========================================================
    # SELECT INPUT MODE
    # ========================================================

    if args.input_dir is not None:

        # ====================================================
        # MODE 1: EXTERNAL UNLABELLED IMAGES
        # ====================================================

        labelled = False

        input_dir = args.input_dir
        if not os.path.exists(input_dir):
            raise FileNotFoundError(
                f"\nInput image or directory not found:\n{input_dir}"
            )
        if not (os.path.isfile(input_dir) or os.path.isdir(input_dir)):
            raise ValueError(
                f"Input path must be a regular image file or directory: {input_dir}"
            )

        print("\nMode:")
        print("EXTERNAL / UNLABELLED IMAGES")

        print("\nInput image or directory:")
        print(input_dir)

    else:

        # ====================================================
        # MODE 2: ORIGINAL DATASET TEST SET
        # ====================================================

        labelled = True

        input_dir = os.path.join(
            args.data_root,
            "test"
        )

        if not os.path.isdir(input_dir):

            raise FileNotFoundError(
                f"\nTest directory not found:\n"
                f"{input_dir}\n\n"
                "Expected dataset structure:\n"
                "data_root/\n"
                "    train/\n"
                "    val/\n"
                "    test/\n\n"
                "For external unlabelled images, "
                "use --input-dir."
            )

        print("\nMode:")
        print("DATASET TEST SET")

        print("\nTest directory:")
        print(input_dir)

    # ========================================================
    # BUILD TRANSFORM
    # ========================================================

    if labelled:

        transform = build_inference_transform(
            args.img_size,
            raw_binary_png=args.raw_binary_png,
        )

    else:

        transform = build_inference_transform(
            args.img_size,
            raw_binary_png=False,
        )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if labelled:

        image_dataset = datasets.ImageFolder(input_dir, transform=transform)
        print(
            f"\nFound {len(image_dataset)} test images"
        )

        print(
            f"Found {len(image_dataset.classes)} classes"
        )

        # ----------------------------------------------------
        # Make sure test dataset uses 119 classes
        # ----------------------------------------------------

        if image_dataset.classes != class_folders:

            raise RuntimeError(
                "Test folder classes are not in the same order as the "
                "training folders. Refusing to report misleading labels."
            )

    else:

        image_dataset = UnlabelledImageDataset(
            input_dir,
            transform=transform,
            preprocess_raw=not args.already_preprocessed,
            img_size=args.img_size,
        )

        print(
            f"\nFound {len(image_dataset)} external images"
        )

        print(
            "Ground-truth labels: Not available"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=NUM_CLASSES,
        device=device,
        head_type=args.head_type,
        kan_hidden=args.kan_hidden,
        kan_grid=args.kan_grid,
        kan_order=args.kan_order,
    )

    print("Model loaded successfully.")

    # ========================================================
    # PREDICT
    # ========================================================

    results = predict_images(
        model=model,
        image_dataset=image_dataset,
        device=device,
        labelled=labelled,
        class_folders=class_folders,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_predictions_csv(
        results=results,
        output_path=args.output,
        labelled=labelled,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    if labelled:

        # ----------------------------------------------------
        # DATASET TEST RESULTS
        # ----------------------------------------------------

        accuracy = calculate_accuracy(results)

        correct = sum(
            row["true_folder"]
            == row["predicted_folder"]
            for row in results
        )

        print("DATASET TEST RESULTS")
        print("=" * 70)

        print(
            f"Total images : {len(results)}"
        )

        print(
            f"Correct      : {correct}"
        )

        print(
            f"Incorrect    : "
            f"{len(results) - correct}"
        )

        print(
            f"Accuracy     : "
            f"{accuracy * 100:.2f}%"
        )

    else:

        # ----------------------------------------------------
        # EXTERNAL IMAGE RESULTS
        # ----------------------------------------------------

        print("EXTERNAL IMAGE PREDICTIONS")
        print("=" * 70)

        print(
            f"Total images : {len(results)}"
        )

        print(
            "Accuracy     : "
            "Not available because images are unlabelled"
        )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()