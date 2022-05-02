import numpy as np
import argparse
import os, sys
import cv2

from imageIO import read_images, save_panorama_images, draw_match_point
from utils import cylindrical_warping
from Feature.harris_detector import detect_feature_point
from Feature.MOPSdescription import get_feature_descriptor
from Feature.SSDfeature_matcher import detect_simple_features_matching

from ransac import find_translation_matrix
from image_blending import image_blending
from utils import bundle_adjustment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="Photos")
    parser.add_argument("-f", "--focal_len", default=704.916)
    parser.add_argument("-m", "--minimum_score_ratio", default=0.005)
    parser.add_argument("-t", "--threshold", default=0.95)
    parser.add_argument("--reverse", action="store_false")
    parser.add_argument("-o", "--output", default="output.png")
    args = parser.parse_args()

    # get the image data
    images = read_images(args.input_dir)
    images = images[::-1] if args.reverse else images
    # get focal length
    focal_len = float(args.focal_len)

    print("-----------Cylinder Warping------------")
    # cylinder warping
    images = np.array(
        [cylindrical_warping(image=img, focal_len=focal_len) for img in images]
    )

    cv2.imwrite(f"output1.png", images[0].astype(np.uint8))

    print("-----------Harris Detection------------")
    # Harris corner detection
    features = [
        detect_feature_point(image, score_ratio=float(args.minimum_score_ratio))
        for image in images
    ]

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
            threshold=float(args.threshold),
        )
        for i in range(images_num)
    ]
    draw_match_point(images, features, matches)
    print("-----------Image Matching------------")
    # calculate translation between each image
    translations = [
        find_translation_matrix(
            feature1=features[i],
            feature2=features[(i + 1) % images_num],
            matches=matches[i],
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

    save_panorama_images(image=panorama, output=args.output)
