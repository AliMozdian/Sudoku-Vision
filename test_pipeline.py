import os
import glob
import argparse
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
    Transforms 81 numpy cell images into a single normalized
    PyTorch tensor of shape (81, 1, 28, 28) for GPU inference.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor_list = []
    for cell in cells:
        pil_img = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY))
        tensor_list.append(transform(pil_img))

    return torch.stack(tensor_list).to(device)


def build_visualization_panel(
    original_img: np.ndarray,
    preproc_img: np.ndarray,
    corners: np.ndarray,
    warped_board: np.ndarray,
    predicted_grid: np.ndarray,
    ground_truth: np.ndarray = None,
) -> np.ndarray:
    """
    Creates a combined visual dashboard showing:
    1. Preprocessed edges with detected corner points.
    2. Warped board with cell grid overlay.
    3. Prediction matrix rendered as an annotated synthetic board.
    """
    panel_h, panel_w = 450, 450

    # 1. Preprocessed view with corner overlays
    preproc_bgr = cv2.cvtColor(preproc_img, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for pt in corners:
            cv2.circle(preproc_bgr, tuple(pt.astype(int)), 12, (0, 0, 255), -1)
    view1 = cv2.resize(preproc_bgr, (panel_w, panel_h))

    # 2. Warped board with grid lines
    view2 = warped_board.copy()
    step = panel_h // 9
    for i in range(1, 9):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(view2, (0, i * step), (panel_w, i * step), (0, 0, 255), thickness)
        cv2.line(view2, (i * step, 0), (i * step, panel_h), (0, 0, 255), thickness)

    # 3. Rendered Prediction Board
    view3 = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    for i in range(1, 9):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(view3, (0, i * step), (panel_w, i * step), (0, 0, 0), thickness)
        cv2.line(view3, (i * step, 0), (i * step, panel_h), (0, 0, 0), thickness)

    # Populate prediction digits
    for r in range(9):
        for c in range(9):
            pred_val = predicted_grid[r, c]
            text = str(pred_val) if pred_val != 0 else "."
            
            # Determine color: green if correct, red if incorrect, gray if empty
            if ground_truth is not None:
                if pred_val == ground_truth[r, c]:
                    color = (120, 120, 120) if pred_val == 0 else (0, 160, 0)  # Gray or Green
                else:
                    color = (0, 0, 220)  # Red (Mismatch)
            else:
                color = (0, 0, 0)

            # Center text in the cell
            pos_x = c * step + int(step * 0.32)
            pos_y = r * step + int(step * 0.70)
            cv2.putText(view3, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Stack the 3 stages horizontally
    combined = np.hstack([view1, view2, view3])

    # Header labels
    header = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "1. Preprocessing & Corners", (70, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(header, "2. Warped Board & Grid", (panel_w + 90, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(header, "3. CNN Predictions (Green: Correct, Red: Error)", (2 * panel_w + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return np.vstack([header, combined])


def run_pipeline(visualize: bool = False, target_image: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    model = SudokuDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    if target_image:
        img_files = [os.path.join(TEST_DIR, target_image)]
    else:
        img_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))

    if not img_files:
        print(f"No valid images found for testing.")
        return

    total_images = len(img_files)
    extracted_boards = 0
    perfect_boards = 0
    total_correct_cells = 0
    total_evaluated_cells = 0

    print(f"\nStarting End-to-End Evaluation on {total_images} Test Sample(s)...")
    print("=" * 65)

    for img_path in img_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        dat_path = os.path.join(TEST_DIR, f"{base_name}.dat")

        image = cv2.imread(img_path)
        if image is None:
            continue

        thresh = preprocess_image(image)
        corners = find_board_corners(thresh)

        if corners is None:
            print(f"[{filename}] [FAIL - Extraction] Could not detect 4 corners.")
            continue

        extracted_boards += 1
        warped = warp_perspective(image, corners, output_size=450)
        cells = extract_cells(warped, margin_ratio=0.12)

        batch_tensor = prepare_cells_tensor(cells, device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_grid = preds.cpu().numpy().reshape(9, 9)

        ground_truth = None
        if os.path.exists(dat_path):
            ground_truth = parse_dat_file(dat_path)
            num_correct = np.sum(predicted_grid == ground_truth)
            accuracy = (num_correct / 81.0) * 100.0
            total_correct_cells += num_correct
            total_evaluated_cells += 81

            if num_correct == 81:
                perfect_boards += 1
                status = "PERFECT (81/81)"
            else:
                status = f"Errors: {81 - num_correct} | Acc: {accuracy:.1f}%"
        else:
            status = "Predictions complete (No .dat ground truth)"

        print(f"[{filename}] -> {status}")

        if visualize:
            dashboard = build_visualization_panel(
                image, thresh, corners, warped, predicted_grid, ground_truth
            )
            cv2.imshow(f"Pipeline Result - {filename}", dashboard)
            print("  --> Displaying visual panel. Press any key for next image (or 'q' to quit visualizer)...")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord('q'):
                print("Visualization terminated by user.")
                break

    print("=" * 65)
    print("SUMMARY RESULTS:")
    print(f"Total Evaluated Images:          {total_images}")
    print(f"Phase 1 Successful Extractions:  {extracted_boards} / {total_images} ({(extracted_boards / max(1, total_images)) * 100:.1f}%)")
    if total_evaluated_cells > 0:
        overall_cell_acc = (total_correct_cells / total_evaluated_cells) * 100.0
        print(f"Overall Cell Digit Accuracy:     {total_correct_cells} / {total_evaluated_cells} ({overall_cell_acc:.2f}%)")
        print(f"Perfect Boards (Zero Errors):    {perfect_boards} / {extracted_boards} ({(perfect_boards / max(1, extracted_boards)) * 100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudoku Vision End-to-End Test Pipeline")
    parser.add_argument("--visualize", action="store_true", help="Display visual dashboard for each image")
    parser.add_argument("--image", type=str, default=None, help="Target specific image file (e.g. image1005.jpg)")
    args = parser.parse_args()

    run_pipeline(visualize=args.visualize, target_image=args.image)
