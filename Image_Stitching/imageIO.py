import numpy as np
import cv2
import os


def read_images(image_dir):
    # get the file list in directory
    files = os.listdir(image_dir)
    files.sort()
    extensions = {".jpg", ".JPG", ".png", ".PNG"}
    paths = [
        os.path.join(image_dir, file)
        for file in files
        if os.path.isfile(os.path.join(image_dir, file))
        and any(file.endswith(extension) for extension in extensions)
    ]
    images = []

    for path in paths:
        # read image and append images into LDR list
        img = cv2.imread(path)

        images.append(img)

    # transform images into np array
    images = np.array(images)

    return images


def save_panorama_images(image, output):
    cv2.imwrite(output, image.astype(np.uint8))
