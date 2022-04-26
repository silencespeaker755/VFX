import numpy as np
import cv2
from scipy import ndimage
import math


def get_feature_descriptor(image, features):
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    image = ndimage.gaussian_filter(image, sigma=0.4)
    gradient_y, gradient_x = np.gradient(image)
    degrees = np.arctan2(gradient_x, gradient_y)

    nums = len(features)
    window_size = 8
    descriptor = np.zeros((nums, window_size * window_size))

    for index, feature in enumerate(features):
        x, y = feature["pt"]
        angle = -degrees[y, x]

        transformation_m1 = np.eye(3)
        transformation_m1[0, 2] = -x
        transformation_m1[1, 2] = -y

        transformation_m2 = np.eye(3)
        transformation_m2[:2, 2] = 4

        scale_m = np.eye(3)
        scale_m[0, 0] = 0.2
        scale_m[1, 1] = 0.2

        rotation_m = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        transform_M = transformation_m2.dot(
            scale_m.dot(rotation_m.dot(transformation_m1))
        )[:2, :3]

        result = np.array(
            cv2.warpAffine(
                image, transform_M, (window_size, window_size), flags=cv2.INTER_LINEAR
            )
        ).flatten()
        result = result - np.mean(result)
        if np.std(result) > 1e-5:
            result = result / np.std(result)
            descriptor[index] = result

    return descriptor
