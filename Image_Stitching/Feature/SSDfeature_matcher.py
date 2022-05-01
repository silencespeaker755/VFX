import numpy as np
import cv2
from scipy.spatial import distance


def detect_simple_features_matching(descriptor1, descriptor2):
    # calculate distance
    distances = distance.cdist(descriptor1, descriptor2, "euclidean")
    matches = []

    threshold = 0.95

    for index, ssd in enumerate(distances):
        sorted_indexes = np.argsort(ssd)
        index_len = len(sorted_indexes)
        threshold_list = np.array(
            [
                (ssd[sorted_indexes[i]] + 1e-8) / (ssd[sorted_indexes[i + 1]] + 1e-8)
                for i in range(index_len - 1)
            ]
        )
        min_threshold_index = np.argmin(threshold_list)
        if threshold_list[min_threshold_index] > threshold:
            continue
        match = {}
        match["queryIdx"] = index
        target_index = int(sorted_indexes[min_threshold_index])
        match["targetIdx"] = target_index
        match["distance"] = distances[index, target_index]
        matches.append(match)

    return np.array(matches)


def detect_minimum_features_matching(descriptor1, descriptor2):
    n1 = descriptor1.shape[0]
    n2 = descriptor2.shape[0]

    if n1 == 0 or n2 == 0:
        return []

    # calculate distance
    distances = distance.cdist(descriptor1, descriptor2, "euclidean")
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
