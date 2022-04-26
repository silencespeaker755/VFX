from operator import index
from pyexpat import features
from threading import local
import numpy as np
import cv2
from scipy import ndimage
from sklearn.linear_model import LassoLarsIC


class HarrisDetector:
    def __init__(self) -> None:
        self.features = []

    def detect_feature_point(self, image):
        # transform to gray color for detection
        # image_gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)

        # gaussian blur for noise
        image = ndimage.gaussian_filter(image, sigma=0.4)

        gradient_y, gradient_x = np.gradient(image)

        Ix_2 = gradient_x**2
        Ix_y = gradient_x * gradient_y
        Iy_2 = gradient_y**2

        Sx_2 = ndimage.gaussian_filter(Ix_2, sigma=0.4)
        Sx_y = ndimage.gaussian_filter(Ix_y, sigma=0.4)
        Sy_2 = ndimage.gaussian_filter(Iy_2, sigma=0.4)
        pixels_matrix = np.stack([Sx_2, Sx_y, Sx_y, Sy_2], axis=-1)
        h, w = pixels_matrix.shape[:2]
        pixels_matrix = pixels_matrix.reshape(h, w, 2, 2)

        det = np.linalg.det(pixels_matrix)
        trace = np.array(Sx_2 + Sy_2)
        score = det - 0.04 * (trace**2)

        local_maximum = self.find_local_maximum(score, 0.01)

        feature_indexlist = np.transpose(np.where(local_maximum))
        feature_element = {"pt": [], "value": 0}
        for index in feature_indexlist:
            point = feature_element.copy()
            # point attribute
            y, x = index
            point["pt"] = [x, y]
            point["value"] = score[y][x]
            self.features.append(point)

        features = self.nonmax_suppression()

        return features

    def find_local_maximum(
        self, image_scores, threshold=0, neighbor_nums=8, bounding=20
    ):
        max_score = np.max(image_scores)
        result = []
        local_maximum = ndimage.maximum_filter(image_scores, neighbor_nums)
        result = image_scores == local_maximum

        # 0.01
        # print(np.sum(result))
        result[local_maximum < max_score * threshold] = False
        result[:20, :] = False
        result[-20:, :] = False
        result[:, :20] = False
        result[:, -20:] = False
        # print(np.sum(result))

        return result

    def nonmax_suppression(self, nums=500):
        coordinate = [np.array(feature["pt"]) for feature in self.features]
        pixel = [feature["value"] for feature in self.features]
        n = len(self.features)
        # index_list = np.argsort(pixel)[::-1]
        # index_list = index_list[: 500 * 30 if n > 500 * 30 else n]

        radius = np.zeros(n)
        radius[0] = np.Inf

        features = []

        for i in range(n - 1):
            candidate_radii = [np.Inf]
            for j in range(0, i):
                if pixel[i] < pixel[j] * 0.9:
                    candidate_radii.append(
                        np.linalg.norm(
                            np.array(coordinate[j]) - np.array(coordinate[i])
                        )
                    )

            radius[i] = np.min(candidate_radii)

        sorted_indexes = np.argsort(radius)
        features = np.array(self.features)
        feature_points = features[sorted_indexes][:nums]

        return feature_points

        # radius = 10000
        # selected = [coordinate[index_list[0]]]
        # current = 1
        # while current < nums:
        #     for index in index_list:
        #         if (
        #             np.min(
        #                 np.linalg.norm(
        #                     np.array(selected) - np.array(coordinate[index]), axis=1
        #                 )
        #             )
        #             > radius
        #         ):
        #             selected.append(coordinate[index])
        #             features.append(self.features[index])
        #             current += 1
        #             radius = 10000
        #     radius /= 2

        # return features
