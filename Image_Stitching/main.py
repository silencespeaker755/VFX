from matplotlib import image
import numpy as np
import argparse
import os, sys
import cv2

from imageIO import read_images
from utils import cylindrical_warping
from Feature.harris_detector import HarrisDetector
from Feature.MOPSdescription import get_feature_descriptor
from Feature.SSDfeature_matcher import detect_simple_features_matching


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
            center1 = feature[idx][i.queryIdx]["pt"]
            center2 = feature[(idx + 1) % n][i.trainIdx]["pt"]

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
    images = images[1:3]

    # cylinder warping
    # images = [cylindrical_warping(image=img, focal_len=focal_len) for img in images]

    print("-----------Harris Detection------------")
    # Harris corner detection
    harris_model = HarrisDetector()
    features = [harris_model.detect_feature_point(image) for image in images]
    # features = harris_model.detect_feature_point(images[0])
    # i = draw_feature_point(images[0], features)

    #############
    # unit test #
    #############

    # cv2.imwrite("output.png", i)

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
            descriptor1=descriptors[i], descriptor2=descriptors[(i + 1) % images_num]
        )
        for i in range(images_num)
    ]

    draw_match_point(images, features, matches)
