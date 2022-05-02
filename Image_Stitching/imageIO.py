import numpy as np
import cv2
import os


def read_images(image_dir):
    # get the file list in directory
    files = os.listdir(image_dir)
    files.sort()
    extensions = {".jpg", ".JPG", ".png", ".PNG"}
    paths = [
        os.path.join(image_dir, file)
        for file in files
        if os.path.isfile(os.path.join(image_dir, file))
        and any(file.endswith(extension) for extension in extensions)
    ]
    images = []

    for path in paths:
        # read image and append images into LDR list
        img = cv2.imread(path)

        images.append(img)

    # transform images into np array
    images = np.array(images)

    return images


def save_panorama_images(image, output):
    cv2.imwrite(output, image.astype(np.uint8))


def draw_match_point(image_src, feature, match):
    image = np.copy(image_src)
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

    os.mkdir("outputs-test")
    for i in range(n):
        cv2.imwrite(f"outputs-test/output-match-{i}.png", image[i])
