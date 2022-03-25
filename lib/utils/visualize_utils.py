
import os
import glob
import math
import json
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def visualize_ehr_output(ehr_res,
                         vis_dir,
                         image,
                         name_vis_image='img_visualize',
                         font_path='test/fonts/latin.ttf',
                         ):
    table_reconstruct_result = ehr_res['table_reconstruct_result']
    paragraph_resuslt = ehr_res['paragraph_result']
    ocr_result = ehr_res['ocr_result']
    if len(table_reconstruct_result) > 0:
        table_boxes = [item['table_box'] for item in table_reconstruct_result]
        cell_boxes = [cell_box for item in table_reconstruct_result
                    for cell_box in item['cell_boxes']]
    else:
        table_boxes = None 
        cell_boxes = None
    para_boxes = [item[0] for item in paragraph_resuslt]

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    boxes, txts = [], []
    for item in ocr_result:
        box, label, prob = item
        boxes.append(box)
        txts.append(label)
    img_visual = draw_ocr_box_txt(image, boxes, txts, font_path=font_path,
                                  table_boxes=table_boxes,
                                  cell_boxes=cell_boxes,
                                  para_boxes=para_boxes
                                  )

    paths = sorted(glob.glob(vis_dir + "/" + name_vis_image + "*"),
                   key=lambda path: int(path.split(".jpg")[0].split("_")[-1]))
    if len(paths) == 0:
        idx_name = '1'
    else:
        idx_name = str(int(paths[-1].split(".jpg")[0].split("_")[-1]) + 1)
    cv2.imwrite(os.path.join(vis_dir, name_vis_image +
                "_" + idx_name + ".jpg"), img_visual)


def export_to_csv(table_reconstruct_text, vis_dir, csv_name='table_text_reconstruct'):
    paths = sorted(glob.glob(vis_dir + "/" + csv_name + "*"),
                   key=lambda path: int(path.split(".csv")[0].split("_")[-1]))
    if len(paths) == 0:
        idx_name = '1'
    else:
        idx_name = str(int(paths[-1].split(".csv")[0].split("_")[-1]) + 1)
    df = pd.DataFrame(table_reconstruct_text)
    df.to_csv(os.path.join(vis_dir, csv_name +
              "_" + idx_name + ".csv"), index=False)


def save_json(data, vis_dir, json_name='ehr_result'):
    """ save dictionary to json file

    Args:
        data (dict): 
        vis_dir (str):  path to save json 
        json_name (str, optional): json name. Defaults to 'ehr_result'.
    """
    paths = sorted(glob.glob(vis_dir + "/" + json_name + "*"),
                   key=lambda path: int(path.split(".json")[0].split("_")[-1]))
    if len(paths) == 0:
        idx_name = '1'
    else:
        idx_name = str(int(paths[-1].split(".json")[0].split("_")[-1]) + 1)
    outpath = os.path.join(vis_dir, json_name + "_" + idx_name + ".json")
    with open(outpath, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="test/fonts/latin.ttf",
                     table_boxes=None,
                     cell_boxes=None,
                     para_boxes=None):
    """

    Args:
        image (np.ndarray / PIL): BGR image or PIL image 
        boxes (list / np.ndarray): list of polygon boxes
        txts (list): list of text labels
        scores (list, optional): probality. Defaults to None.
        drop_score (float, optional): . Defaults to 0.5.
        font_path (str, optional): Path of font. Defaults to "test/fonts/latin.ttf".

    Returns:
        np.ndarray: BGR image
    """
    color_vis = {
        'table': (255, 192, 70),
        'cell': (218, 66, 15),
        'paragraph': (0, 187, 148)
    }

    random.seed(0)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if isinstance(boxes, list):
        boxes = np.array(boxes)

    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)

    if table_boxes is not None:
        img_left = draw_rectangle_pil(
            img_left,
            table_boxes,
            color=color_vis['table'],
            width=6,
            label='table'
        )
    if cell_boxes is not None:
        img_left = draw_rectangle_pil(
            img_left,
            cell_boxes,
            color=color_vis['cell'],
            width=5,
            label='cell'
        )
    if para_boxes is not None:
        img_left = draw_rectangle_pil(
            img_left,
            para_boxes,
            color=color_vis['paragraph'],
            width=2,
            label='para'
        )

    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    img_show = cv2.cvtColor(np.array(img_show), cv2.COLOR_RGB2BGR)
    return img_show


def draw_rectangle_pil(pil_image,
                       boxes,
                       color,
                       width=1,
                       label=None,
                       font_path="test/fonts/latin.ttf"
                       ):
    """

    Args:
        pil_image ([type]): [description]
        boxes (list): list of [xmin, ymim, xmax, ymax]
        color (list): list of (R, G, B)
    """
    drawer = ImageDraw.Draw(pil_image)
    color = tuple((int(color[0]), int(color[1]), int(color[2])))
    for box in boxes:
        drawer.rectangle([(int(box[0]), int(box[1])), (int(
            box[2]), int(box[3]))], outline=color, width=width)
        
        if label:
            font_size = 35
            font = ImageFont.truetype(font_path, size=32, encoding="utf-8")
            drawer.text([int(box[0]) + 5, int(box[1]) - font_size - 5],
                        label, fill=color, font=font)
    return pil_image
