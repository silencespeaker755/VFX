from operator import index
import numpy as np


def aggregate_translation(translations):
    n = len(translations)
    aggregations = [translations[0]]
    for i in range(1, n):
        aggregations.append(aggregations[i - 1] + translations[i])

    return np.around(np.array(aggregations)).astype(np.int32)


def generate_panorama(aggregations, shape):
    h, w, c = shape
    aggregations_x = aggregations[:, 0]
    aggregations_y = aggregations[:, 1]

    max_x, min_x = np.max(aggregations_x), np.min(aggregations_x)
    max_y, min_y = np.max(aggregations_y), np.min(aggregations_y)

    pano_h = abs(min_y) + h + abs(max_y)
    pano_w = abs(min_x) + w + abs(max_x)

    panorama = np.zeros(shape=(pano_h, pano_w, c), dtype=np.float32)

    return panorama


def image_blending(images, translations):
    aggregations = aggregate_translation(translations)
    panorama = generate_panorama(aggregations=aggregations, shape=images.shape[1:4])
    origin = np.array(
        [abs(np.min(aggregations[:, 0])), abs(np.min(aggregations[:, 1]))]
    )

    for index, (image, aggregation, translation) in enumerate(
        zip(images, aggregations, translations)
    ):
        image_origin = origin + aggregation
        panorama = linear_blending(
            panorama=panorama,
            image=image,
            origin=image_origin,
            translation=translation,
            initial=(index == 0),
        )

    return panorama


def linear_blending(panorama, image, origin, translation, initial):
    # image type transformation
    image = image.astype(np.float32)

    # parameter initialization
    x, y = origin
    h, w = image.shape[:2]

    if initial:
        panorama[y : y + h, x : x + w] = image
    else:
        accumulation = np.zeros(panorama.shape, dtype=np.float32)
        accumulation[y : y + h, x : x + w] = image

        # weight map
        pano_weight, accumulation_weight = generate_image_weight(panorama, accumulation)
        intersection = get_intersection(pano_weight, accumulation_weight)
        pano_weight, accumulation_weight = calculate_wight_map(
            pano_weight, accumulation_weight, intersection, translation
        )

        # blending
        panorama = pano_weight * panorama + accumulation_weight * accumulation

    return panorama


def generate_image_weight(panorama, accumulation):
    return np.sign(panorama), np.sign(accumulation)


def get_intersection(pano_weight, accumulation_weight):
    total = pano_weight + accumulation_weight
    return total - np.sign(total)


def calculate_wight_map(pano_weight, accumulation_weight, intersection, translation):
    range_x = np.where(np.sum(intersection, axis=0) > 0)[0]
    range_y = np.where(np.sum(intersection, axis=1) > 0)[0]

    start_x, end_x = range_x[0], range_x[-1]
    start_y, end_y = range_y[0], range_y[-1]

    cover_length_x = end_x - start_x + 1
    cover_length_y = end_y - start_y + 1

    linear_weight = np.zeros((cover_length_y, cover_length_x), dtype=np.float32)
    linear_weight += (
        np.linspace(0, 1, cover_length_x)
        if translation[0] >= 0
        else np.linspace(1, 0, cover_length_x)
    )
    linear_weight = np.stack([linear_weight, linear_weight, linear_weight], axis=-1)

    pano_weight[start_y : end_y + 1, start_x : end_x + 1] *= 1 - linear_weight
    accumulation_weight[start_y : end_y + 1, start_x : end_x + 1] *= linear_weight

    return pano_weight, accumulation_weight
