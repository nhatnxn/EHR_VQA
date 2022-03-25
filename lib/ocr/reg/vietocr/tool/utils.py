import gdown
import numpy as np
import cv2
from PIL import Image, JpegImagePlugin


def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def calculate_ratio(width,height):
    '''
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    '''
    ratio = width/height
    if ratio < 1.0:
        ratio = 1./ratio
    return ratio

def compute_ratio_and_resize(img,width,height,model_height):
    '''
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = calculate_ratio(width,height)
        img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.ANTIALIAS)
    else:
        img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
    return img,ratio




def get_paragraph(raw_result, x_ths=1, y_ths=0.5, mode = 'ltr'):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append([box[1], min_x, max_x, min_y, max_y, height, 0.5*(min_y+max_y), 0]) # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7]==0]) > 0:
        box_group0 = [box for box in box_group if box[7]==0] # group0 = non-group
        # new group
        if len([box for box in box_group if box[7]==current_group]) == 0:
            box_group0[0][7] = current_group # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7]==current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths*mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths*mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths*mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths*mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx<=box[1]<=max_gx) or (min_gx<=box[2]<=max_gx)
                same_vertical_level = (min_gy<=box[3]<=max_gy) or (min_gy<=box[4]<=max_gy)
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box==False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7]==i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ''
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [box for box in current_box_group if box[6]<highest+0.4*mean_height]
            # get the far left
            if mode == 'ltr':
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left: best_box = box
            elif mode == 'rtl':
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right: best_box = box
            text += ' '+best_box[0]
            current_box_group.remove(best_box)

        result.append([ [[min_gx,min_gy],[max_gx,min_gy],[max_gx,max_gy],[min_gx,max_gy]], text[1:]])

    return result


def reformat_input(image):
    # if type(image) == str:
    #     if image.startswith('http://') or image.startswith('https://'):
    #         tmp, _ = urlretrieve(image , reporthook=printProgressBar(prefix = 'Progress:', suffix = 'Complete', length = 50))
    #         img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
    #         os.remove(tmp)
    #     else:
    #         img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #         image = os.path.expanduser(image)
    #     img = loadImage(image)  # can accept URL
    if type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2: # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3: # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4: # RGBAscale
            img = image[:,:,:3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(image) == JpegImagePlugin.JpegImageFile:
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Invalid input type. Suppoting format = string(file path or url), bytes, numpy array')

    return img, img_cv_grey