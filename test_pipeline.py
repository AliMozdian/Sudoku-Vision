import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import SudokuDigitCNN
from src.board_extractor import (
    preprocess_image,
    find_board_corners,
    warp_perspective,
    extract_cells,
)

TEST_DIR = "data/raw/v2_test"
MODEL_PATH = "models/best_sudoku_cnn.pth"


def parse_dat_file(dat_path: str) -> np.ndarray:
    """Parses a ground truth .dat file into a 9x9 integer array."""
    with open(dat_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    grid_lines = lines[-9:]
    matrix = [[int(v) for v in row.split()] for row in grid_lines]
    return np.array(matrix, dtype=int)


def prepare_cells_tensor(cells: list, device: torch.device) -> torch.Tensor:
    """
    Transforms a list of 81 numpy cell images into a single normalized
    PyTorch tensor of shape (81, 1, 28, 28) ready for GPU inference.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor_list = []
    for cell in cells:
        # Convert OpenCV BGR cell to PIL grayscale
        pil_img = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY))
        tensor_list.append(transform(pil_img))

    # Stack 81 individual (1, 28, 28) tensors into (81, 1, 28, 28)
    batch_tensor = torch.stack(tensor_list).to(device)
    return batch_tensor


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Load the trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    model = SudokuDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. Gather test images
    img_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
    if not img_files:
        print(f"No .jpg files found in {TEST_DIR}!")
        return

    total_images = len(img_files)
    extracted_boards = 0
    perfect_boards = 0
    total_correct_cells = 0
    total_evaluated_cells = 0

    print(f"\nStarting End-to-End Evaluation on {total_images} Test Boards...")
    print("=" * 65)

    for img_path in img_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        dat_path = os.path.join(TEST_DIR, f"{base_name}.dat")

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Phase 1: Board Extraction
        thresh = preprocess_image(image)
        corners = find_board_corners(thresh)

        if corners is None:
            print(f"[{filename}] [FAIL - Extraction] Could not detect 4 corners.")
            continue

        extracted_boards += 1
        warped = warp_perspective(image, corners, output_size=450)
        # Margin ratio 0.14 helps eliminate outer line intrusion
        cells = extract_cells(warped, margin_ratio=0.12)

        # Phase 2: CNN Inference on 81 cells
        batch_tensor = prepare_cells_tensor(cells, device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_grid = preds.cpu().numpy().reshape(9, 9)

        # Ground Truth Comparison
        if not os.path.exists(dat_path):
            print(f"[{filename}] [Extracted] Ground truth .dat missing, skipped comparison.")
            continue

        ground_truth = parse_dat_file(dat_path)
        correct_mask = (predicted_grid == ground_truth)
        num_correct = np.sum(correct_mask)
        accuracy = (num_correct / 81.0) * 100.0

        total_correct_cells += num_correct
        total_evaluated_cells += 81

        if num_correct == 81:
            perfect_boards += 1
            status = "PERFECT (81/81)"
        else:
            status = f"Errors: {81 - num_correct} | Acc: {accuracy:.1f}%"

        print(f"[{filename}] -> {status}")

    # Summary Report
    print("=" * 65)
    print("SUMMARY RESULTS ON TEST SET:")
    print(f"Total Test Images:               {total_images}")
    print(f"Phase 1 Successful Extractions:  {extracted_boards} / {total_images} ({(extracted_boards / total_images) * 100:.1f}%)")
    if total_evaluated_cells > 0:
        overall_cell_acc = (total_correct_cells / total_evaluated_cells) * 100.0
        print(f"Overall Cell Digit Accuracy:     {total_correct_cells} / {total_evaluated_cells} ({overall_cell_acc:.2f}%)")
        print(f"Perfect Boards (Zero Errors):    {perfect_boards} / {extracted_boards} ({(perfect_boards / extracted_boards) * 100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
