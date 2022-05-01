import numpy as np
import cv2, exifread
import os
import math


def read_images(image_dir, focal_file):
    # get the file list in directory
    files = os.listdir(image_dir)
    files.sort()

    paths = [
        os.path.join(image_dir, file)
        for file in files
        if os.path.isfile(os.path.join(image_dir, file))
    ]
    images = []
    focal_len = None

    for path in paths:
        # read image and append images into LDR list
        img = cv2.imread(path)

        images.append(img)

    # get focal length
    focal_len = get_focal_len(focal_file)
    print(f"{path} -> focal length: {focal_len}")

    # transform images into np array
    images = np.array(images)

    return images, focal_len


def get_focal_len(path):
    with open(path, "r") as f:
        focals = np.array(
            [
                float(focal.split(",")[1].strip())
                for focal in map(str, f)
                if focal != "\n"
            ]
        )
        return np.average(focals)

    # get file's exif tags
    # exif_tags = exifread.process_file(open(path, "rb"))

    # return focal len
    # return transform_exif_fraction_to_float(str(exif_tags["EXIF FocalLength"]))


def transform_exif_fraction_to_float(fraction):
    numbers = list(map(float, fraction.split("/")))

    if len(numbers) == 1:
        return numbers[0]
    else:
        numbers[1] = 2 ** (math.ceil(math.log(numbers[1], 2)))

    return numbers[0] / numbers[1]


def save_HDR_images(image, output):
    cv2.imwrite(output, image.astype(np.float32))
