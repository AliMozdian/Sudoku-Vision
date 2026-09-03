import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale, blurs, and applies adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to reduce high-frequency noise while preserving boundaries
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # Adaptive threshold creates a binary image robust to varying illumination
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


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


def find_board_corners(thresh_img: np.ndarray) -> np.ndarray:
    """Finds the 4 corners of the largest quadrilateral contour."""
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        # Approximate contour with a simpler polygon (epsilon = 2% of perimeter)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        # The Sudoku board boundary is a 4-point polygon
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


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
