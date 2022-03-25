
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.tools.infer.predict_cls import TextClassifier
from paddleocr.paddleocr import parse_args
from lib.utils.config import Cfg

class PaddleTextDetector(object):
    def __init__(
        self,
        config_path: str,
        use_gpu=False
    ):
        config = Cfg.load_config_from_file(config_path)

        self.args = parse_args(mMain=False)
        self.args.__dict__.update(
            det_model_dir=config['model_dir'],
            gpu_mem=config['gpu_mem'],
            use_gpu=use_gpu,
            use_zero_copy_run=True,
            max_batch_size=1,
            det_limit_side_len=config['det_limit_side_len'],         #960
            det_limit_type=config['det_limit_type'],                 #'max'
            det_db_unclip_ratio=config['det_db_unclip_ratio'], 
            det_db_thresh=config['det_db_thresh'], 
            det_db_box_thresh=config['det_db_box_thresh'], 
            det_db_score_mode=config['det_db_score_mode'], 
        )
        self.text_detector = TextDetector(self.args)

    def __call__(self, image):
        """

        Args:
            image (np.ndarray): BGR images

        Returns:
            np.ndarray: numpy array of poly boxes - shape 4x2
        """
        dt_boxes, time_infer = self.text_detector(image)
        return dt_boxes


class PaddleTextClassifier(object):
    def __init__(
        self,
        config_path: str,
        use_gpu=False
    ):
        config = Cfg.load_config_from_file(config_path)

        self.args = parse_args(mMain=False)
        self.args.__dict__.update(
            cls_model_dir=config['model_dir'],
            gpu_mem=config['gpu_mem'],
            use_gpu=use_gpu,
            use_zero_copy_run=True,
            cls_batch_num=config['max_batch_size'], 
        )
        self.text_classifier = TextClassifier(self.args)

    def __call__(self, images):
        """
        Args:
            images (np.ndarray): list of BGR images

        Returns:
            img_list, cls_res, elapse : cls_res format = (label, conf)
        """
        out= self.text_classifier(images)
        return out