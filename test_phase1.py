import cv2
import matplotlib.pyplot as plt
from src.board_extractor import preprocess_image, find_board_corners, warp_perspective

def cut_and_save(filename: str, src_dir: str, dest_dir: str):
    img_path = src_dir + filename
    image = cv2.imread(img_path)
    cv2.imshow('original-'+filename, image)
    cv2.waitKey()
    
    if image is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    thresh = preprocess_image(image)
    cv2.imshow('thresh-'+filename, thresh)
    cv2.waitKey()

    corners = find_board_corners(thresh)

    if corners is None:
        print("Failed to find 4 corners of the board!")
    else:
        print("Found corners successfully:\n", corners)

        warped = warp_perspective(image, corners, output_size=450)

        cv2.imwrite(f"{dest_dir}warped_board-{filename}", warped)
        print(f"Saved 'warped_board-{filename}' to {dest_dir} directory.")
        cv2.imshow('warped-'+filename, warped)
        cv2.waitKey()


def main():
    """cut all and save them"""
    src_dir = "data/raw/v2_train/"
    dst_dir = "data/processed/tests/"

    with open('./data/raw/v2_train.desc', 'r') as f:
        fnames = list(map(lambda name: name.split('/')[1][:-1], f.readlines()))

    for fn in fnames:
       cut_and_save(fn, src_dir, dst_dir)
       cv2.destroyAllWindows()
    # cut_and_save('image29.jpg', src_dir, dst_dir)

if __name__ == '__main__':
    main()
