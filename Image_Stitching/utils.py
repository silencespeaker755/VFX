import numpy as np
import cv2


def cylindrical_warping(image, focal_len):
    h, w = image.shape[:2]

    # how could it be
    # focal_len = 704.916
    # focal_len = 768
    focal_len = 745

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

    return np.array(
        cv2.remap(
            image,
            image_coordinate[:, :, 0],
            image_coordinate[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
        ),
        dtype=np.float32,
    )


def bundle_adjustment(panorama):
    # get shape of panorama image
    h, w = panorama.shape[:2]

    # get non-black area map of panorama image
    panorama_map = np.sign(np.sum(panorama, axis=2))
    range_x = np.where(np.sum(panorama_map, axis=0) > 0)[0]

    # leftmost and rightmost x-axis value which is not 0
    left_x, right_x = range_x[0], range_x[-1]

    # the relavent top and bottom y-axis value on leftmost x value
    left_coordinate_y = np.where(panorama_map[:, left_x] > 0)[0]
    left_bottom_point = [left_x, left_coordinate_y[-1]]
    left_top_point = [left_x, left_coordinate_y[0]]

    # the relavent top and bottom y-axis value on rightmost x value
    right_coordinate_y = np.where(panorama_map[:, right_x] > 0)[0]
    right_bottom_point = [right_x, right_coordinate_y[-1]]
    right_top_point = [right_x, right_coordinate_y[0]]

    # perspective warping
    input_points = np.array(
        [left_top_point, right_top_point, left_bottom_point, right_bottom_point],
        dtype=np.float32,
    )

    output_points = np.array(
        [[0, 0], [w, 0], [0, h], [w, h]],
        dtype=np.float32,
    )

    perspective_transform = cv2.getPerspectiveTransform(input_points, output_points)
    panorama = cv2.warpPerspective(panorama, perspective_transform, (w, h))

    return panorama
