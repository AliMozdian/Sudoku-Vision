import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale, blurs, and applies adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce high-frequency noise while preserving boundaries
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Adaptive threshold creates a binary image robust to varying illumination
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological Closing
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)
    closed_thresh = cv2.morphologyEx(closed_thresh, cv2.MORPH_CLOSE, kernel_v)

    return closed_thresh


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 coordinates: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference (y - x): top-right has smallest diff, bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def find_board_corners(thresh_img: np.ndarray, min_area_ratio=0.15, min_aspect_ratio=0.5) -> np.ndarray:
    """Finds the 4 corners of the Sudoku grid with area and aspect ratio checks."""
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    total_area = thresh_img.shape[0] * thresh_img.shape[1]

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * total_area:
            continue
        hull = cv2.convexHull(c)
        perimeter = cv2.arcLength(hull, True)

        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
            if len(approx) == 4:
                # Calculate bounding box aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = min(w, h) / max(w, h)
                # Sudoku grids are square: aspect ratio should be reasonably close to 1.0
                if aspect_ratio > min_aspect_ratio:
                    candidates.append((area, approx.reshape(4, 2), aspect_ratio))
                    break # Stop checking epsilon for this contour

    if not candidates:
        return None

    # my choice of selecting the best: largest
    best_candidate = max(candidates, key=lambda item: (round(item[2], 2), item[0]))
    return best_candidate[1]


def warp_perspective(image: np.ndarray, corners: np.ndarray, output_size: int = 450) -> np.ndarray:
    """Warps the quadrilateral region into a flat, top-down square."""
    ordered_corners = order_points(corners)

    # Define destination coordinates for a top-down square of size (output_size x output_size)
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))
    return warped
