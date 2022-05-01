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


def draw_feature_point(image, feature):
    for idx, m in enumerate(feature):
        center1, center2 = feature[idx]["pt"]
        cv2.circle(image, (int(center1), int(center2)), 7, (0, 0, 255), 10)

    return image


def draw_match_point(image, feature, match):
    n = len(image)
    color = [
        (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        for _ in range(500)
    ]
    for idx, m in enumerate(match):
        cnt = 0
        for i in m:
            center1 = feature[idx][i["queryIdx"]]["pt"]
            center2 = feature[(idx + 1) % n][i["targetIdx"]]["pt"]

            cv2.circle(
                image[idx], (int(center1[0]), int(center1[1])), 4, (0, 0, 255), 5
            )
            cv2.circle(
                image[(idx + 1) % n],
                (int(center2[0]), int(center2[1])),
                4,
                (0, 0, 255),
                5,
            )

            cv2.putText(
                image[idx],
                str(cnt),
                (int(center1[0]), int(center1[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color[idx],
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image[(idx + 1) % n],
                str(cnt),
                (int(center2[0]), int(center2[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color[idx],
                1,
                cv2.LINE_AA,
            )

            cnt += 1
        break

    os.mkdir("outputs-test")
    for i in range(n):
        cv2.imwrite(f"outputs-test/output-match-{i}.png", image[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="Photos")
    parser.add_argument("-f", "--focal_file", default="focal_length.txt")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    # get the image data and the average of their focal length
    images, focal_len = read_images(args.input_dir, args.focal_file)

    print("-----------Cylinder Warping------------")
    # cylinder warping
    images = np.array(
        [cylindrical_warping(image=img, focal_len=focal_len) for img in images]
    )

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

    # draw_match_point(images, features, matches)

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

    cv2.imwrite(f"output.png", panorama.astype(np.uint8))
