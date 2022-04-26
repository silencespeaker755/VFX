import numpy as np
import cv2
import scipy


def detect_simple_features_matching(descriptor1, descriptor2):
    n1 = descriptor1.shape[0]
    n2 = descriptor2.shape[0]

    if n1 == 0 or n2 == 0:
        return []

    # calculate distance
    distances = scipy.spatial.distance.cdist(descriptor1, descriptor2, "euclidean")
    descriptor1_avialable = np.ones(n1, dtype=bool)
    descriptor2_avialable = np.ones(n2, dtype=bool)

    matches = []
    threshold = 0.6

    while threshold <= 1.0:
        for index, ssd in enumerate(distances):
            if descriptor1_avialable[index]:
                sorted_indexes = np.argsort(ssd)
                index_len = len(sorted_indexes)
                for i in range(index_len - 1):
                    if (
                        descriptor2_avialable[sorted_indexes[i]]
                        and (
                            (ssd[sorted_indexes[i]] + 1e-8)
                            / (ssd[sorted_indexes[i + 1]] + 1e-8)
                        )
                        < threshold
                    ):
                        match = cv2.DMatch()
                        match.queryIdx = index
                        target_index = int(sorted_indexes[i])
                        match.trainIdx = target_index
                        match.distance = distances[index, target_index]
                        matches.append(match)
                        descriptor1_avialable[index] = False
                        descriptor2_avialable[target_index] = False

                        break
        threshold += 0.1

    return matches


def detect_minimum_features_matching(descriptor1, descriptor2):
    n1 = descriptor1.shape[0]
    n2 = descriptor2.shape[0]

    if n1 == 0 or n2 == 0:
        return []

    # calculate distance
    distances = scipy.spatial.distance.cdist(descriptor1, descriptor2, "euclidean")
    distance_list = np.argmin(distances, axis=1)
    print("distance")
    print(distances)
    print("list")
    print(distance_list)

    matches = []
    for index, distance_index in enumerate(distance_list):
        match = cv2.DMatch()
        match.queryIdx = index
        match.trainIdx = int(distance_index)
        match.distance = distances[index, int(distance_index)]
        matches.append(match)

    return matches
