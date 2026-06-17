import math

import cv2 as cv
import numpy as np


def imread_mine(
    image_path,
    scale=1.0,
    rotate=0.0,
    crop=False,
    border_value=(0, 0, 0),
):
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]
    if rotate:
        center = (width // 2, height // 2)
        affine = cv.getRotationMatrix2D(center, rotate, 1.0)
        if crop:
            output_size = (width, height)
        else:
            cosine = abs(affine[0, 0])
            sine = abs(affine[0, 1])
            new_width = int(height * sine + width * cosine)
            new_height = int(height * cosine + width * sine)
            affine[0, 2] += new_width / 2 - center[0]
            affine[1, 2] += new_height / 2 - center[1]
            output_size = (new_width, new_height)

        transformed = cv.warpAffine(
            image,
            affine,
            output_size,
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=border_value,
        )
    else:
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        transformed = image

    if scale != 1.0:
        transformed = cv.resize(
            transformed,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_CUBIC,
        )
        affine = affine * scale

    homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    return transformed, homography


def shear_warp(
    image,
    shear_x=0.0,
    shear_y=0.0,
    homography_x=0.0,
    homography_y=0.0,
    border_value=(0, 0, 0),
):
    height, width = image.shape[:2]
    matrix = np.array(
        [
            [1.0, math.tan(math.radians(shear_x)), 0.0],
            [math.tan(math.radians(shear_y)), 1.0, 0.0],
            [homography_x, homography_y, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = cv.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return transformed, matrix


def expand_image_size(image, factor, border_value=(0, 0, 0)):
    height, width = image.shape[:2]
    new_height = math.ceil(height / factor) * factor
    new_width = math.ceil(width / factor) * factor

    mask = np.zeros((new_height, new_width), dtype=bool)
    mask[:height, :width] = True

    if image.ndim == 2:
        fill_value = border_value[0] if isinstance(border_value, tuple) else border_value
        expanded = np.full(
            (new_height, new_width),
            fill_value,
            dtype=image.dtype,
        )
        expanded[:height, :width] = image
    elif image.ndim == 3 and image.shape[2] == 3:
        expanded = np.full(
            (new_height, new_width, 3),
            border_value,
            dtype=image.dtype,
        )
        expanded[:height, :width] = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return expanded, mask
