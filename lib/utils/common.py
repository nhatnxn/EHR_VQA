import cv2
import numpy as np
import random
import pytesseract
import re
import math

from PIL import Image
from difflib import SequenceMatcher
import fitz

MAX_SIZE_PADDLE = 1920


# ============================================== Visualize bounding boxes ===================================================
def hex_to_BGR(x):
    return (int(x[5:7], base=16), int(x[3:5], base=16), int(x[1:3], base=16))


def is_hex(x):
    return x[0] == "#" and len(x) == 7


colors_str_to_hex = {
    'red': '#e74c3c',
    'orange': '#e67e22',
    'yellow': '#f1c40f',
    'green': '#2ecc71',
    'blue-green': '#1abc9c',
    'blue': '#3498db',
    'purple': '#9b59b6',
    'black': '#2c3e50',
    'white': '#ecf0f1'
}

colors_str_to_BGR = {k: hex_to_BGR(v) for k, v in colors_str_to_hex.items()}


def convert_color(x):
    if is_hex(x):
        color = hex_to_BGR(x)
    elif x in colors_str_to_BGR.keys():
        color = colors_str_to_BGR[x]
    else:
        raise ValueError('Invalid color: color string must be a hex code or one of ' +
                         str(list(colors_str_to_BGR.keys())))

    return color


def draw_bounding_box(image,
                      box,
                      labels=[],  # xmin,ymin,xmax,ymax
                      color='red',
                      font_face=cv2.FONT_HERSHEY_DUPLEX,
                      font_size=.5,
                      font_weight=1,
                      font_color='white',
                      text_padding=3,
                      border_thickness=2
                      ):

    color = convert_color(color)
    font_color = convert_color(font_color)

    box = [int(x) for x in box]
    xmin, ymin, xmax, ymax = box

    #draw bounding box
    image = cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), color, border_thickness)

    text_dims = [cv2.getTextSize(
        x, font_face, font_size, font_weight) for x in labels]
    total_label_height = sum([x[0][1] for x in text_dims]) + sum([x[1]
                                                                  for x in text_dims]) + (text_padding * len(labels) * 2)

    border_offset = (border_thickness //
                     2) if border_thickness % 2 == 0 else (border_thickness // 2) + 1

    label_xmin = xmin - border_offset
    label_ymin = ymin - total_label_height - border_offset
    if label_ymin < 0:
        label_ymin = ymax + border_offset

    for i, label in enumerate(labels):

        text_width = text_dims[i][0][0]
        text_height = text_dims[i][0][1]
        text_baseline = text_dims[i][1]
        label_height = text_dims[i][0][1] + text_dims[i][1] + text_padding * 2

        label_xmax = label_xmin + text_width + text_padding * 2
        label_ymax = label_ymin + label_height

        cv2.rectangle(image, (label_xmin, label_ymin),
                      (label_xmax, label_ymax), color, -1)
        cv2.putText(image, label, (label_xmin + text_padding, label_ymin + text_height +
                    text_padding), font_face, font_size, (255, 255, 255), font_weight, cv2.LINE_AA)

        label_ymin = label_ymax


def visualize(image, boxes, name="vz.png"):
    """
    Args:
        image (cv2)
        boxes ([[xmin, ymin, xmax, ymax]]): list of (xmin, ymin, xmax, ymax)
    """
    image_new = image.copy()
    for idx, box in enumerate(boxes):
        box = [int(b) for b in box]
        color_box = random.sample(('red', 'blue'), 1)[0]
        draw_bounding_box(image_new, box, color=color_box,
                          border_thickness=1, labels=[str(idx)])


# ============================================== Visualize bounding boxes ===================================================

def similar(a, b):
    """Get similarity of two string

    Args:
        a (str): key
        b (str): predict

    Returns:
        float
    """
    return SequenceMatcher(None, a, b).ratio()


def resize_image(image, max_size=MAX_SIZE_PADDLE):
    """Resize to max_size for text detection

    Args:
        max_size (int): maximum size (w, h) 
    """
    h, w = image.shape[:2]
    maxhw = max(h, w)
    if maxhw > max_size:
        ratio = max_size / maxhw
        image = cv2.resize(image, None, fx=ratio, fy=ratio)

    return image


def pdf2images(pdf_obj=None):
    """Conver pdf to image BRG"""

    document = pdf_obj.open_with_fitz()
    zoom = 2.2    # zoom factor
    mat = fitz.Matrix(zoom, zoom)

    for page in document:
        # Org image page
        pix = page.getPixmap(matrix=mat)
        image_org_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_org_data = cv2.cvtColor(np.array(image_org_data), cv2.COLOR_RGB2BGR)

        # Remove annotations
        annotations = page.annots()
        for annotation in annotations: 
            page.deleteAnnot(annotation)

        # Remove Signature.
        widgets = page.widgets()
    
        for widget in widgets:
            annot = widget._annot
            page.deleteAnnot(annot)

        pix = page.getPixmap(matrix=mat)
        image_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_data = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

        yield (image_data, image_org_data)


def crop_image(image, crop_size=1280):
    """Crop center image
    """
    h, w = image.shape[:2]
    x_center, y_center = w // 2, h // 2
    half_size = crop_size // 2

    xmin, ymin = x_center - half_size, y_center - half_size
    xmax, ymax = x_center + half_size, y_center + half_size

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, w)
    ymax = min(ymax, h)

    return image[ymin:ymax, xmin:xmax]


def image_rotate(image):
    """Rotate image (90, 180, 270) to 0 degrees"""
    blank_image = False
    try:
        rot_data = pytesseract.image_to_osd(image)
    except:
        blank_image = True

    if blank_image:
        return 0

    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)
    angle = float(rot)

    return angle


def remove_small_box(boxes, threshold_x=15, threshold_y=15):
    """Lọc các box có kích thước quả nhỏ, tránh việc cluster sai"""
    good_boxes = []
    for box in boxes:
        if box[2] - box[0] <= threshold_x or box[3] - box[1] <= threshold_y:
            continue
        else:
            good_boxes.append(box)
    return good_boxes


def padding_box(h, w, box, vertical_padding, horizontal_padding):
    """Padding box text after using paddle text detection algorithm

    Args:
        h (float): height of the original image
        w (float): width of the original image
        box ([xmin, ymin, xmax, ymax]): box text
        vertical_padding (float): vertical padding
        horizontal_padding (float): horizontal padding

    Returns:
        [xmin, ymin, xmax, ymax]: box after padding
    """

    box[0] = max(0, box[0] - horizontal_padding)
    box[1] = max(0, box[1] - vertical_padding)
    box[2] = min(w, box[2] + horizontal_padding)
    box[3] = min(h, box[3] + vertical_padding)
    return box


def padding_image(image, color=[0, 0, 0]):
    """Rescale image to original size

    Args:
        image (cv2): bbox detect by sign model
        h_org (float): height of the original image
        w_org (float): width of the original image 

    Returns:
        [cv2] : image after padding white pixels
    """
    h, w = image.shape[:2]
    h_org = int(h + (h * 0.05))
    w_org = int(w + (w * 0.05))
    
    image = cv2.copyMakeBorder(image, (h_org - h) // 2, (h_org - h) // 2, (w_org - w) // 2, (w_org - w) // 2, cv2.BORDER_CONSTANT, value=color)
    return image


def polylist2bboxlist(poly_list):
    """ Convert polygon list to bbox list

    Args:
        poly_list (np.array): shape N*4*2

    Returns:
        np.array: shape: 
    """
    boxes = np.array(poly_list, dtype = int)
    top_lefts = np.stack([boxes[:, :, 0].min(axis=1), boxes[:,:,1].min(axis=1)], axis=1)
    bot_rights = np.stack([boxes[:, :, 0].max(axis=1), boxes[:,:,1].max(axis=1)], axis=1)
    bbox_list = np.concatenate([top_lefts, bot_rights], axis=1)
    return bbox_list


def distance(p1, p2):
    """Calculate Euclid distance
    """
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return dist