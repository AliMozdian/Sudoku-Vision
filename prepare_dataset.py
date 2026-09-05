import os
import cv2
import numpy as np
from src.board_extractor import preprocess_image, find_board_corners, warp_perspective, extract_cells

RAW_IMG_DIR = "data/raw/v2_train/"
DESC_FILE = "data/raw/v2_train.desc"
VERIFIED_LIST_FILE = "data/processed/verified_train_list.txt"
CELLS_DIR = "data/processed/cells"


def get_verified_files() -> set:
    """Reads the set of already verified image filenames."""
    if not os.path.exists(VERIFIED_LIST_FILE):
        return set()
    with open(VERIFIED_LIST_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def draw_grid(img: np.ndarray, num_divisions: int = 9) -> np.ndarray:
    """Draws a 9x9 grid on top of an image to check alignment."""
    vis = img.copy()
    step = vis.shape[0] // num_divisions
    for i in range(1, num_divisions):
        # Draw red grid lines
        cv2.line(vis, (0, i * step), (vis.shape[1], i * step), (0, 0, 255), 1)
        cv2.line(vis, (i * step, 0), (i * step, vis.shape[0]), (0, 0, 255), 1)
    return vis


def interactive_verify():
    """
    Shows warped boards with 9x9 grid overlay to the user.
    Controls:
      [Y] or [Space] : Accept and append to verified list
      [N]            : Reject (skip)
      [Q]            : Save & Quit
    """
    os.makedirs("data/processed", exist_ok=True)
    verified = get_verified_files()

    with open(DESC_FILE, "r") as f:
        all_fnames = [line.strip().split("/")[-1] for line in f if line.strip()]

    to_review = [fn for fn in all_fnames if fn not in verified]
    print(f"Total images: {len(all_fnames)} | Already verified: {len(verified)} | Remaining: {len(to_review)}")
    print("Controls: [Y] / [Space] = Accept | [N] = Reject | [Q] = Quit")

    for fn in to_review:
        img_path = os.path.join(RAW_IMG_DIR, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue

        thresh = preprocess_image(img)
        corners = find_board_corners(thresh)

        if corners is None:
            print(f"[{fn}] Failed to detect corners. Auto-skipped.")
            continue

        warped = warp_perspective(img, corners, output_size=450)
        preview = draw_grid(warped)

        # Put instructions on the preview window
        cv2.putText(preview, f"{fn} - [Y]=Keep [N]=Skip [Q]=Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Board Verification", preview)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("Session ended by user.")
            break
        elif key in [ord('y'), 32]:  # 'y' or spacebar
            with open(VERIFIED_LIST_FILE, "a") as f_out:
                f_out.write(fn + "\n")
            verified.add(fn)
            print(f"Accepted: {fn}")
        else:
            print(f"Rejected: {fn}")

    cv2.destroyAllWindows()
    print(f"Verification complete. Total verified images: {len(verified)}")


def parse_dat_file(dat_path: str) -> np.ndarray:
    """
    Parses a .dat file containing the 9x9 Sudoku values into a NumPy array.
    """
    with open(dat_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # In wichtounet dataset, the 9x9 matrix is usually in the last 9 lines
    grid_lines = lines[-9:]
    matrix = []
    for row in grid_lines:
        values = [int(v) for v in row.split()]
        matrix.append(values)
    return np.array(matrix, dtype=int)


def extract_and_save_cells():
    """
    Takes all verified boards, crops their 81 cells, and sorts them
    into subfolders: data/processed/cells/0, /1, ..., /9.
    """
    verified = get_verified_files()
    if not verified:
        print("No verified files found! Run interactive_verify() first.")
        return

    # Create subdirectories 0 to 9
    for i in range(10):
        os.makedirs(os.path.join(CELLS_DIR, str(i)), exist_ok=True)

    cell_counts = {i: 0 for i in range(10)}

    for fn in verified:
        base_name = os.path.splitext(fn)[0]
        img_path = os.path.join(RAW_IMG_DIR, fn)
        dat_path = os.path.join(RAW_IMG_DIR, base_name + ".dat")

        if not os.path.exists(dat_path):
            print(f"Warning: Missing .dat ground truth for {fn}. Skipping.")
            continue

        img = cv2.imread(img_path)
        thresh = preprocess_image(img)
        corners = find_board_corners(thresh)
        if corners is None:
            continue

        warped = warp_perspective(img, corners, output_size=450)
        # Margin ratio 0.12 cleanly trims inner grid borders
        cells = extract_cells(warped, margin_ratio=0.12)
        grid_labels = parse_dat_file(dat_path)  # shape (9, 9)

        # Save each of the 81 cells into its respective class directory
        for idx, cell in enumerate(cells):
            r = idx // 9
            c = idx % 9
            label = grid_labels[r, c]

            cell_fname = f"{base_name}_r{r}_c{c}.png"
            save_path = os.path.join(CELLS_DIR, str(label), cell_fname)
            cv2.imwrite(save_path, cell)
            cell_counts[label] += 1

    print("\nDataset extraction complete!")
    print("Class distribution (number of cell images per class):")
    for digit, count in cell_counts.items():
        name = "Empty (0)" if digit == 0 else str(digit)
        print(f"  Class {name}: {count} images")


if __name__ == "__main__":
    # Step 1: Run the interactive verifier (Accept with Y/Space, Reject with N)
    interactive_verify()

    # Step 2: Once satisfied with verified list, extract & save the dataset
    extract_and_save_cells()