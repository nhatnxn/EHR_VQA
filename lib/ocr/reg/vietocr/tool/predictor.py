import os
import math
from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
import torch


from .translate import translate, process_input, build_model
from ..model.vocab import Vocab
from ..model.transformerocr_onnx import OnnxVietOCR

class Predictor():
    def __init__(self, config, use_gpu=False):

        self.config = config
        self.batch_size_infer = config['batch_size_infer']            
        if use_gpu:
            self.config['device'] = 'cuda'
        else:
            self.config['device'] = 'cpu'

        if config['weights']['use_onnx']:
            self.model = OnnxVietOCR(**config['weights']['onnx_weight_path'], use_gpu=use_gpu)
            self.vocab = Vocab(config['vocab'])
        else:

            model, vocab = build_model(config)
            torch_weight_path = config['weights']['torch_weight_path']
            if os.path.exists(torch_weight_path):
                model.load_state_dict(torch.load(torch_weight_path, map_location=torch.device(config['device'])))
            else:
                raise Exception("Cannot find weight: ", torch_weight_path)
            
            self.model = model
            self.vocab = vocab
        
    def predict(self, img):
        '''
        :param img:
        :return:
        '''
        img = process_input(img, self.config['dataset']['image_height'],
                            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'],
                            is_padding=self.config['dataset']['is_padding'], round_to=self.config['dataset']['round_to'])
        img = img.to(self.config['device'])

        s, prob = translate(img, self.model, get_prob=False)
        s = s[0].tolist()
        s = self.vocab.decode(s)
        return s

    def batch_predict(self, images):
        """
        Recognize images on batch

        Parameters:
        images(list): list of cropped images
        set_buck_thresh(int): threshold to merge bucket in images

        Return:
        result(list string): ocr results
        """
        if len(images) == 0:
            return ['']
        batch_dict, indices = self.batch_process(images)

        list_keys = [i for i in batch_dict if batch_dict[i]
                     != batch_dict.default_factory()]
        text_preds = list([])
        probs = []

        for width in list_keys:
            batch = batch_dict[width]
            if len(batch) > self.batch_size_infer:
                batches = self.split_to_batch(batch, max_batch_size=self.batch_size_infer)
            else:
                batches = [batch]
            for batch in batches:
                batch = np.asarray(batch)
                batch = torch.FloatTensor(batch)
                batch = batch.to(self.config['device'])
                sent, prob = translate(batch, self.model, get_prob=True)
                sent = sent.tolist()
                batch_text = self.vocab.batch_decode(sent)
                text_preds.extend(batch_text)
                probs.extend(prob.tolist())

        # sort text result corresponding to original coordinate
        z = zip(text_preds, probs, indices)
        sorted_result = sorted(z, key=lambda x: x[2])
        text_preds, probs, indices = zip(*sorted_result)

        return text_preds, probs

    def split_to_batch(self, imgs, max_batch_size):
        if len(imgs) <= max_batch_size:
            return [imgs]
        img_batches = []
        num_batch = math.ceil(len(imgs) / max_batch_size)
        for idx_batch in range(num_batch):
            start_idx_batch = idx_batch * max_batch_size
            end_idx_batch = start_idx_batch + max_batch_size
            img_batches.append(imgs[start_idx_batch:end_idx_batch])

        return img_batches

    def preprocess_input(self, img, image_height, image_min_width, image_max_width, round_to=True,
                  padding_type='right'):
        """
        Preprocess input image (resize, normalize)

        Parameters:
        image: has shape of (H, W, C)   :cv2 Image

        Return:
        img: has shape (H, W, C)
        """
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w, h = img.size
        new_w = int(image_height * float(w) / float(h))
        if new_w < image_max_width and new_w != 0:
            if new_w != w:
                img = img.resize((new_w, image_height), Image.ANTIALIAS)
            if padding_type == 'center':
                box_paste = ((image_max_width - new_w) // 2, 0)
            elif padding_type == 'right':
                box_paste = (0, 0)
            else:
                raise Exception("Not implement padding_type")

            new_w, image_height = self.get_width_for_cluster(w, h, image_height, image_min_width, image_max_width, round_to=round_to)
            new_img = Image.new('RGB', (new_w, image_height), 'white')  # padding white
            new_img.paste(img, box=box_paste)
            img = new_img
        else:
            img = img.resize((image_max_width, image_height), Image.ANTIALIAS)

        img = np.asarray(img).transpose(2, 0, 1)
        img = img / 255

        return img

    def batch_process(self, images):
        """
        Preprocess list input images and divide list input images to sub bucket which has same length

        Parameters:
        image: has shape of (B, H, W, C)
            set_buck_thresh(int): threshold to merge bucket in images

        Return:
        batch_img_dict: list
            list of batch imgs
        indices:
            position of each img in "images" argument
        """

        batch_img_dict = defaultdict(list)
        image_height = self.config['dataset']['image_height']
        image_min_width = self.config['dataset']['image_min_width']
        image_max_width = self.config['dataset']['image_max_width']
        padding_type = self.config['dataset']['padding_type']
        round_to = self.config['dataset']['round_to']
        img_li = [self.preprocess_input(img,  image_height, image_min_width, image_max_width, round_to=round_to,
                  padding_type=padding_type) for img in images]
        img_li, width_list, indices = self.sort_width(img_li, reverse=False)

        for i, image in enumerate(img_li):
            c, h, w = image.shape
            batch_img_dict[w].append(image)

        return batch_img_dict, indices

    @staticmethod
    def sort_width(batch_img: list, reverse: bool = False):
        """
        Sort list image correspondint to width of each image

        Parameters
        ----------
        batch_img: list
            list input image

        Return
        ------
        sorted_batch_img: list
            sorted input images
        width_img_list: list
            list of width images
        indices: list
            sorted position of each image in original batch images
        """

        def get_img_width(element):
            img = element[0]
            c, h, w = img.shape
            return w
        batch = list(zip(batch_img, range(len(batch_img))))
        sorted_batch = sorted(batch, key=get_img_width, reverse=reverse)
        sorted_batch_img, indices = list(zip(*sorted_batch))
        width_img_list = list(map(get_img_width, batch))

        return sorted_batch_img, width_img_list, indices

    @staticmethod
    def get_width_for_cluster(w: int, h: int, expected_height: int, image_min_width: int, image_max_width: int, round_to=50):
        """
        Get expected height and width of image

        Parameters
        ----------
        w: int
            width of image
        h: int
            height
        expected_height: int
        image_min_width: int
        image_max_width: int
            max_width of

        Return
        ------

        """
        new_w = int(expected_height * float(w) / float(h))
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height
