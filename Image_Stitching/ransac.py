from nis import match
from shutil import move
import numpy as np


def find_translation_matrix(
    feature1, feature2, matches, size=1, iterations=10000, tolerence=10, reverse=True
):
    feature1_coordinates = np.array(
        [feature1[match["queryIdx"]]["pt"] for match in matches]
    )
    feature2_coordinates = np.array(
        [feature2[match["targetIdx"]]["pt"] for match in matches]
    )
    distances = feature2_coordinates - feature1_coordinates

    matches_num = len(matches)
    max_inlier_num = -1
    max_inlier_list = None
    unit = np.eye(2)

    for _ in range(iterations):
        samples_index = np.random.randint(0, matches_num, size)

        sample_coordinate1 = feature1_coordinates[samples_index].flatten()
        sample_coordinate2 = feature2_coordinates[samples_index].flatten()

        B = sample_coordinate2 - sample_coordinate1
        A = np.array([unit] * size).reshape(size * 2, 2)

        move = np.linalg.lstsq(A, B, rcond=None)[0]

        lier_list = np.sum(np.abs(distances - move), axis=1) < tolerence
        inlier_num = np.sum(lier_list)

        if inlier_num > max_inlier_num:
            max_inlier_num = inlier_num
            max_inlier_list = lier_list

    inlier_coordinate1 = feature1_coordinates[max_inlier_list].flatten()
    inlier_coordinate2 = feature2_coordinates[max_inlier_list].flatten()

    B = (
        inlier_coordinate1 - inlier_coordinate2
        if reverse
        else inlier_coordinate2 - inlier_coordinate1
    )
    A = np.array([unit] * max_inlier_num).reshape(max_inlier_num * 2, 2)
    translation = np.linalg.lstsq(A, B, rcond=None)[0]
    print(translation)

    return translation
