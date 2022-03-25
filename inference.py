import os
import cv2
import numpy as np

from lib.align.doc_align import align_image
from lib.ocr.reader import Reader
from lib.ocr.det.db_paddle.infer import PaddleTextClassifier


from copy import deepcopy
from PIL import Image

import paddle
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForTokenClassification
from paddlenlp.transformers import LayoutLMModel, LayoutLMTokenizer, LayoutLMForTokenClassification

# relative reference
from vqa_utils import parse_args, get_image_file_list, draw_ser_results, get_bio_label_maps

from vqa_utils import pad_sentences, split_page, preprocess, postprocess, merge_preds_list_with_ocr_info

MODELS = {
    'LayoutXLM':
    (LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForTokenClassification),
    'LayoutLM':
    (LayoutLMTokenizer, LayoutLMModel, LayoutLMForTokenClassification)
}


def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


def parse_ocr_info_for_ser(ocr_result):
    ocr_info = []
    for res in ocr_result:
        ocr_info.append({
            "text": res[1][0],
            "bbox": trans_poly_to_bbox(res[0]),
            "poly": res[0],
        })
    return ocr_info


class SerPredictor(object):
    def __init__(self, args, det_name='DB',
                 det_config_path='config/ocr/ocr_det_db.yml',
                 det_use_gpu=False,
                 reg_name='seq2seq',
                 reg_config_path='config/ocr/ocr_reg_seq2seq.yml',
                 reg_use_gpu=True,
                 cls_config_path='config/ocr/ocr_cls.yml',
                 cls_use_gpu=False
                 ):
        self.args = args
        self.max_seq_length = args.max_seq_length

        # init ser token and model
        tokenizer_class, base_model_class, model_class = MODELS[
            args.ser_model_type]
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path)
        self.model = model_class.from_pretrained(args.model_name_or_path)
        self.model.eval()
        
        self.reader = Reader(
            det_name=det_name,
            det_config_path=det_config_path,
            det_use_gpu=det_use_gpu,
            reg_name=reg_name,
            reg_config_path=reg_config_path,
            reg_use_gpu=reg_use_gpu,
        )
        self.text_cls = PaddleTextClassifier(
            config_path=cls_config_path,
            use_gpu=cls_use_gpu)

        # # init ocr_engine
        # from paddleocr import PaddleOCR

        # self.ocr_engine = PaddleOCR(
        #     rec_model_dir=args.rec_model_dir,
        #     det_model_dir=args.det_model_dir,
        #     use_angle_cls=False,
        #     show_log=False)

        # init dict
        label2id_map, self.id2label_map = get_bio_label_maps(
            args.label_map_path)
        self.label2id_map_for_draw = dict()
        for key in label2id_map:
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]

    def __call__(self, img):

        image_align = align_image(image=img,
                                  corner_detector=self.corner_detector,
                                  text_detector=self.reader.detect_model,
                                  text_cls=self.text_cls
                                  )

        ocr_result = self.reader(image_align)

        ocr_info = parse_ocr_info_for_ser(ocr_result)

        inputs = preprocess(
            tokenizer=self.tokenizer,
            ori_img=img,
            ocr_info=ocr_info,
            max_seq_len=self.max_seq_length)

        if self.args.ser_model_type == 'LayoutLM':
            preds = self.model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"])
        elif self.args.ser_model_type == 'LayoutXLM':
            preds = self.model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                image=inputs["image"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"])
            preds = preds[0]

        preds = postprocess(inputs["attention_mask"], preds, self.id2label_map)
        ocr_info = merge_preds_list_with_ocr_info(
            ocr_info, inputs["segment_offset_id"], preds,
            self.label2id_map_for_draw)
        return ocr_info, inputs


def make_input(ser_input, ser_result, max_seq_len=512):
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}

    entities = ser_input['entities'][0]
    assert len(entities) == len(ser_result)

    # entities
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_result, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])
    entities = dict(start=start, end=end, label=label)

    # relations
    head = []
    tail = []
    for i in range(len(entities["label"])):
        for j in range(len(entities["label"])):
            if entities["label"][i] == 1 and entities["label"][j] == 2:
                head.append(i)
                tail.append(j)

    relations = dict(head=head, tail=tail)

    batch_size = ser_input["input_ids"].shape[0]
    entities_batch = []
    relations_batch = []
    for b in range(batch_size):
        entities_batch.append(entities)
        relations_batch.append(relations)

    ser_input['entities'] = entities_batch
    ser_input['relations'] = relations_batch

    ser_input.pop('segment_offset_id')
    return ser_input, entity_idx_dict

class SerReSystem(object):
    def __init__(self, args):
        self.ser_engine = SerPredictor(args)
        self.tokenizer = LayoutXLMTokenizer.from_pretrained(
            args.re_model_name_or_path)
        self.model = LayoutXLMForRelationExtraction.from_pretrained(
            args.re_model_name_or_path)
        self.model.eval()

    def __call__(self, img):
        ser_result, ser_inputs = self.ser_engine(img)
        re_input, entity_idx_dict = make_input(ser_inputs, ser_result)

        re_result = self.model(**re_input)

        pred_relations = re_result['pred_relations'][0]
        # 进行 relations 到 ocr信息的转换
        result = []
        used_tail_id = []
        for relation in pred_relations:
            if relation['tail_id'] in used_tail_id:
                continue
            used_tail_id.append(relation['tail_id'])
            ocr_info_head = ser_result[entity_idx_dict[relation['head_id']]]
            ocr_info_tail = ser_result[entity_idx_dict[relation['tail_id']]]
            result.append((ocr_info_head, ocr_info_tail))

        return result


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # get infer img list
    infer_imgs = get_image_file_list(args.infer_imgs)

    # loop for infer
    ser_re_engine = SerReSystem(args)
    with open(
            os.path.join(args.output_dir, "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, img_path in enumerate(infer_imgs):
            save_img_path = os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(img_path))[0] + "_re.jpg")
            print("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))

            img = cv2.imread(img_path)

            result = ser_re_engine(img)
            fout.write(img_path + "\t" + json.dumps(
                {
                    "result": result,
                }, ensure_ascii=False) + "\n")

            img_res = draw_re_results(img, result)
            cv2.imwrite(save_img_path, img_res)
