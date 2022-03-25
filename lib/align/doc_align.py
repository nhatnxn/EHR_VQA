import math
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import collections

from lib.dewarp.doc_dewarp import get_dewarped_quad

import cv2
from lib.utils.common import *

import logging

MIN_LONG_EDGE = 40**2
NUMBER_BOX_FOR_ALIGNMENT = 10
MAX_ANGLE = 175
MIN_ANGLE = 7
MIN_NUM_BOX_TEXT = 3
SIZE_DETECTION = 1280
CROP_SIZE = 2000


LOGGER = logging.getLogger(__name__)


def dewarp(image, corners):
    """Transform perspective view and crop

    Args:
        image ([type]): [description]
        corners (list): [[top-left], [top-right], [bottom-right], [bottom-left]]
        where top-left, top-right, bottom-right, and bottom-left is a point (x, y)
        e.g. [[0, 0], [200, 0], [200, 100], [0, 100]]

    Returns:
        np.array: dewarped image
    """

    assert image is not None
    bboxes = []
    corners_xy = []
    for corner in corners:
        bboxes.append(corner[:4])
        xmin, ymin, xmax, ymax = corner[:4]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        corners_xy.append((x_center, y_center))


    dewarp_image = image
    if corners is not None:
        if len(corners) == 4:
            dewarp_image = get_dewarped_quad(image, corners_xy, need_validate=True)
        elif len(corners) > 4:
            corners_xy = np.array(corners_xy)
            corners_xy = np.int32([corners_xy])

            rect = cv2.boundingRect(corners_xy)
            x, y, w, h = rect
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            dewarp_image = image[ymin:ymax, xmin:xmax]

    return dewarp_image


def align_image(image, corner_detector, text_detector, text_cls):
    """ Align image by using text detection + text classify models

    Args:
        image (np.ndarray): BGR image
        text_detector (deepmodel): Text detection model
        text_cls (deepmodel): Text classify model

    Returns:
        np.ndarray: Aligned image
    """

    is_blank, image = adjust_image(image, text_detector, text_cls)
    # if not is_blank:
    #     image = padding_image(image)
    #     corners = corner_detector(image, img_size=SIZE_DETECTION)[0]
    #     image = dewarp(image, corners)

    # image = resize_image(image)
    return image


def cal_width(poly_box):
    """Calculate width of a polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    tl, tr, br, bl = poly_box
    edge_s, edge_l = distance(tl, tr), distance(tr, br)

    return max(edge_s, edge_l)


def cal_angle(poly_box):
    """Calculate the angle between two point"""
    a = poly_box[0]
    b = poly_box[1]
    c = poly_box[2]

    # Get the longer edge 
    if distance(a, b) >= distance(b, c):
        x, y = a, b
    else:
        x, y = b, c

    angle = math.degrees(math.atan2(-(y[1]-x[1]), y[0]-x[0]))

    if angle < 0:
        angle = 180 - abs(angle)

    return angle
    

def reject_outliers(data, m=5.):
    """Remove noise angle
    """
    list_index = np.arange(len(data))
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return list_index[s < m], data[s < m]


def adjust_image(image, text_detector, text_cls):
    """ 
    Args:
        image (np.ndarray): BGR image
        text_detector (deepmodel): Text detection model
        num_boxes (int, optional): number of boxes to calculate adjust angle. Defaults to 5.

    Returns:
        np.ndarray
    """
    image_resized = crop_image(image, crop_size=CROP_SIZE)
    poly_box_texts = text_detector(image_resized)
    is_blank = False

    if len(poly_box_texts) <= MIN_NUM_BOX_TEXT:
        is_blank = True
        return is_blank, image

    # Filter small poly
    poly_box_areas = [[cal_width(poly_box), id] for id, poly_box in enumerate(poly_box_texts)]
    poly_box_areas = sorted(poly_box_areas)[-NUMBER_BOX_FOR_ALIGNMENT:]
    poly_box_areas = [poly_box_texts[id[1]] for id in poly_box_areas]

    # Calculate angle
    list_angle = [cal_angle(poly_box) for poly_box in poly_box_areas]
    list_angle = [angle if angle >= MIN_ANGLE else 180 for angle in list_angle ]

    LOGGER.info(f"List angle before reject outlier: {list_angle}")
    list_angle = np.array(list_angle)
    list_index, list_angle = reject_outliers(list_angle)
    LOGGER.info(f"List angle after reject outlier: {list_angle}")

    if len(list_angle):
        angle = np.mean(list_angle)
    else:
        angle = 0
    LOGGER.info(f"List angle: {angle}")

    # Reuse poly boxes detected by text detection
    polys_org = PolygonsOnImage([Polygon(poly_box_areas[index]) for index in list_index], shape=image_resized.shape)
    seq_augment = iaa.Sequential([
        iaa.Rotate(angle, fit_output=True, order=3)
    ])

    # Rotate image by degree
    if angle >= MIN_ANGLE and angle <= MAX_ANGLE:
        image_resized, polys_aug = seq_augment(image=image_resized, polygons=polys_org)
    else:
        angle = 0
        image_resized, polys_aug = image_resized, polys_org

    # Classify image 0 or 180 degree
    list_poly = [poly.coords for poly in polys_aug]

    image_crop_list = [get_dewarped_quad(image_resized, poly) for poly in list_poly]

    cls_res = text_cls(image_crop_list)
    cls_labels = [cls_[0] for cls_ in cls_res[1]]
    counter = collections.Counter(cls_labels)

    
    if counter['0'] <= counter['180']:
        aug = iaa.Rotate(angle + 180, fit_output=True, order=3)
    else:
        aug = iaa.Rotate(angle, fit_output=True, order=3)

    # Rotate the image by degree
    image = aug.augment_image(image)

    return is_blank, image
