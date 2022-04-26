import numpy as np
import cv2


def cylindrical_warping(image, focal_len):
    h, w = image.shape[:2]

    # how could it be
    focal_len = 704.916

    intrinsics = [[focal_len, 0, w / 2], [0, focal_len, h / 2], [0, 0, 1]]
    # get inverse of intrinsic matrix
    intrinsics_inverse = np.linalg.inv(intrinsics)

    # get the coordinates index matrix
    x_coordinates, y_coordinates = np.meshgrid(np.arange(w), np.arange(h))

    # generate homogeneous vector with reshape for dot implementation
    homo_coordinate = np.stack(
        [x_coordinates, y_coordinates, np.ones((h, w))], axis=-1
    ).reshape(h * w, 3)

    # project 2d image coordinate into camera space
    projected_coordinate = np.dot(intrinsics_inverse, homo_coordinate.T).T

    cylindrical_coordinate = np.stack(
        [
            np.sin(projected_coordinate[:, 0]),
            projected_coordinate[:, 1],
            np.cos(projected_coordinate[:, 0]),
        ],
        axis=-1,
    ).reshape(h * w, 3)

    # project camera space coordinate back to homogeneous vector
    homo_coordinate = np.dot(intrinsics, cylindrical_coordinate.T).T

    # transform from homogeneous vector into normal image coordinate
    image_coordinate = (
        (homo_coordinate[:, :-1] / homo_coordinate[:, [-1]])
        .reshape(h, w, 2)
        .astype(np.float32)
    )

    return cv2.remap(
        image,
        image_coordinate[:, :, 0],
        image_coordinate[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
    )
