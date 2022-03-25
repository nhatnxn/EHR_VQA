from .tool.predictor import Predictor
from .tool.utils import four_point_transform, get_paragraph
from lib.utils.config import Cfg

import numpy as np

class VietOCR():
    def __init__(self, config_path, paddle_ocr=None, use_gpu=True):
        # config_path = "config/vgg-seq2seq-dmec.yml"
        self.config = Cfg.load_config_from_file(config_path)
        self.predictor = Predictor(self.config, use_gpu=use_gpu)
        self.paddle_ocr = paddle_ocr

        self.__warmup__()
        print("Loaded VietOCR")


    def __warmup__(self, max_width=2000, max_height=32):
        print("Warming text recognize model...")
        imgs = [np.random.randint(0, 255, (max_height, max_width, 3), dtype=np.uint8)]
        self.predictor.batch_predict(imgs)

    def recognize(self, 
                image, 
                horizontal_list=None, 
                free_list=None, 
                detail=1, 
                paragraph=False, 
                reformat=True,
                y_ths=0.5, 
                x_ths=1.0):
        """
        Args:
            image (np.ndarray): [description]
            horizontal_list (list, optional): [description]. Defaults to None.
            free_list (list, optional): [description]. Defaults to None.
            detail (int, optional): [description]. Defaults to 1.
            paragraph (bool, optional): [description]. Defaults to False.
            reformat (bool, optional): [description]. Defaults to True.
            y_ths (float, optional): [description]. Defaults to 0.5.
            x_ths (float, optional): [description]. Defaults to 1.0.

        Returns:
            list: (poly box, text, conf)
        """
        '''
        :param image: IMPORTANT: cv2 image color (not grayscale)
        :param free_list: [(tl, tr, br, bl), (tl, tr, br, bl), ...] : tl ~  top left, ...
        :return:
        '''

        image_list, max_width = self.get_image_list(horizontal_list=horizontal_list, free_list=free_list, img=image)
        if len(image_list) == 0:
            return []
        result = self.get_text(image_list)

        if paragraph:
            direction_mode = 'ltr'
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode=direction_mode)
        return  result

    def get_text(self, image_list):
        coord = [item[0].tolist() for item in image_list]
        img_list = [item[1] for item in image_list]
        text_preds, probs = self.predictor.batch_predict(img_list)
        result = []
        for i, (box, pred, prob) in enumerate(zip(coord, text_preds, probs)):
            result.append((box, pred, prob))
        return result


    def get_image_list(self, horizontal_list=None, free_list=None, img=None):
        '''

        :param horizontal_list:
        :param free_list:
        :param img:  cv2 color image
        :param model_height:
        :param sort_output:
        :return:
        '''
        image_list = []
        if free_list is not None:
            for box in free_list:
                rect = np.array(box, dtype="float32")
                transformed_img = four_point_transform(img, rect)
                image_list.append((box, transformed_img))
        else:
            for box in horizontal_list:
                xmin, ymin, xmax, ymax = box
                cropped_img = img[ymin:ymax, xmin:xmax]
                image_list.append((box, cropped_img))

        max_width = None
        return image_list, max_width

