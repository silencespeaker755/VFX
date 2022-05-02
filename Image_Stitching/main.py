from matplotlib import image
import numpy as np
import argparse
import os, sys
import cv2

from imageIO import read_images
from utils import cylindrical_warping
from Feature.harris_detector import detect_feature_point
from Feature.MOPSdescription import get_feature_descriptor
from Feature.SSDfeature_matcher import detect_simple_features_matching

from ransac import find_translation_matrix
from image_blending import aggregate_translation
from image_blending import image_blending
from utils import bundle_adjustment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="Photos")
    parser.add_argument("-f", "--focal_file", default="focal_length.txt")
    parser.add_argument("--reverse", action="store_false")
    parser.add_argument("-o", "--output", default="output.png")
    args = parser.parse_args()

    # get the image data and the average of their focal length
    images, focal_len = read_images(args.input_dir, args.focal_file)
    images = images[:2]
    print(args.reverse)

    print("-----------Cylinder Warping------------")
    # cylinder warping
    images = np.array(
        [cylindrical_warping(image=img, focal_len=focal_len) for img in images]
    )

    cv2.imwrite(f"output1.png", images[0].astype(np.uint8))

    print("-----------Harris Detection------------")
    # Harris corner detection
    features = [detect_feature_point(image) for image in images]

    images_num = len(images)

    print("-----------MOSP Description------------")
    # MOSP descriptor
    descriptors = [
        get_feature_descriptor(images[i], features[i]) for i in range(images_num)
    ]

    print("-----------SSD Feature Matching------------")
    # SSD feature matching
    matches = [
        detect_simple_features_matching(
            descriptor1=descriptors[i],
            descriptor2=descriptors[(i + 1) % images_num],
        )
        for i in range(images_num)
    ]

    print("-----------Image Matching------------")
    # calculate translation between each image
    translations = [
        find_translation_matrix(
            feature1=features[i],
            feature2=features[(i + 1) % images_num],
            matches=matches[i],
            reverse=args.reverse,
        )
        for i in range(images_num - 1)
    ]
    translations = [np.zeros(2, dtype=np.float64)] + translations

    print("-----------Image Blending------------")
    # image blending
    panorama = image_blending(images=images, translations=translations)

    print("-----------Bundle Adjustment------------")
    panorama = bundle_adjustment(panorama=panorama)

    #############
    # unit test #
    #############

    cv2.imwrite(args.output, panorama.astype(np.uint8))
