import cv2
import matplotlib.pyplot as plt
from src.board_extractor import preprocess_image, find_board_corners, warp_perspective

def cut_and_save(filename: str, src_dir: str, dest_dir: str, display=False):
    img_path = src_dir + filename
    image = cv2.imread(img_path)
    w, h, clr= image.shape
    image = cv2.resize(image, (h//2, w//2))
    if display:
        cv2.imshow('original-'+filename, image)
        cv2.waitKey()
    
    if image is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    preproc = preprocess_image(image)
    if display:
        cv2.imshow('preproc-'+filename, preproc)
        cv2.waitKey()

    corners = find_board_corners(preproc)
    
    if corners is None:
        print(f"Failed to find 4 corners of the board! filename={filename}")
    else:
        # print("Found corners successfully:\n", corners)

        if display:
            for c in corners:
                cv2.circle(preproc, center=c, radius=50, color=(255, 255, 255), thickness=2)
            cv2.imshow('preproc-with-corners-'+filename, preproc)
            cv2.waitKey()
        

        warped = warp_perspective(image, corners, output_size=450)

        cv2.imwrite(f"{dest_dir}warped_board-{filename}", warped)
        print(f"Saved 'warped_board-{filename}' to {dest_dir} directory.")

        if display:
            cv2.imshow('warped-'+filename, warped)
            cv2.waitKey()


def main():
    """cut all and save them"""
    src_dir = "data/raw/v2_train/"
    dst_dir = "data/processed/tests/"

    with open('./data/raw/v2_train.desc', 'r') as f:
        fnames = list(map(lambda name: name.split('/')[1][:-1], f.readlines()))

    for fn in fnames:
       cut_and_save(fn, src_dir, dst_dir, display=True)
       cv2.destroyAllWindows()
    # cut_and_save('image1066.jpg', src_dir, dst_dir, display=True)

if __name__ == '__main__':
    main()
