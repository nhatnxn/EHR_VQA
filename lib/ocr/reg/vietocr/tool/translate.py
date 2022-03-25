import torch
import numpy as np
import math
from PIL import Image
from torch.nn.functional import softmax
from ..model.vocab import Vocab
from ..model.transformerocr import VietOCR


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2, get_prob=False):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)
        translated_sentence = torch.empty((1, len(img)), dtype=torch.int64, device=device).fill_(sos_token)
        char_probs = torch.ones((1, len(img)), dtype=torch.int64, device=device)

        max_length = 0

        while max_length <= max_seq_length and not all(torch.any(translated_sentence.T == eos_token, axis=1)):

            tgt_inp = translated_sentence

            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = torch.unsqueeze(indices, 0)

            translated_sentence = torch.cat((translated_sentence, indices), dim=0)
            if get_prob:
                values = values[:, -1, 0]
                values = torch.unsqueeze(values, 0)
                char_probs = torch.cat((char_probs, values), dim=0)

            max_length += 1

            del output

        if get_prob:
            char_probs = char_probs.T
            char_probs = torch.mul(char_probs, translated_sentence.T > 3)
            char_probs = torch.sum(char_probs, dim=-1) / (char_probs > 0).sum(-1)
        else:
            char_probs = None

        translated_sentence = translated_sentence.T

    return translated_sentence, char_probs.cpu().numpy()



def build_model(config):

    vocab = Vocab(config['vocab'])
    device = config['device']

    model = VietOCR(len(vocab),
                    config['backbone'],
                    config['cnn'],
                    config['transformer'],
                    config['seq_modeling'])

    model = model.to(device)

    return model, vocab


def resize(w, h, expected_height, image_min_width, image_max_width, round_to=50):
    new_w = int(expected_height * float(w) / float(h))

    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def resize_for_cluster(w, h, expected_height, image_min_width, image_max_width, round_to):
    new_w = int(expected_height * float(w) / float(h))
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width, is_padding=True, img_mode='RGB',
                  padding_type='right', round_to=100):
    img = image.convert(img_mode)

    w, h = img.size
    if not is_padding:
        new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
        img = img.resize((new_w, image_height), Image.ANTIALIAS)
    else:
        new_w = int(image_height * float(w) / float(h))
        if new_w != w:
            if new_w < image_max_width :
                img = img.resize((new_w, image_height), Image.ANTIALIAS)
                if padding_type == 'center':
                    box_paste = ((image_max_width - new_w) // 2, 0)
                elif padding_type == 'right':
                    box_paste = (0, 0)
                else:
                    raise Exception("Not implement padding_type")

                new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width, round_to)
                new_img = Image.new(img_mode, (new_w, image_height), 'white')  # padding white
                new_img.paste(img, box=box_paste)

                img = new_img

            else:
                # resize
                img = img.resize((image_max_width, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width, is_padding=True, round_to=50):
    img = process_image(image, image_height, image_min_width, image_max_width, is_padding=is_padding, round_to=round_to)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)

    return s

