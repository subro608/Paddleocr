# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess
import json
import base64
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import json
import cv2
import csv
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls

import os
import sys
import logging
import functools
import paddle.distributed as dist
import logging

import imghdr
import cv2
import random
import numpy as np
import openai
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from transformers import GPT2Tokenizer
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
import paddle
logger_initialized = {}


@functools.lru_cache()
def get_logger(name='ppocr', log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if dist.get_rank() == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    logger.propagate = False
    return logger
logger = get_logger()

def check_and_read(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logger = logging.getLogger('ppocr')
            logger.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:] in ['pdf']:
        import fitz
        from PIL import Image
        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.pageCount):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.getPixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False

def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

def extract_data(input_string):
    # Split the input_string into filename and json_string
    _, json_string = input_string.split("\t", 1)
    # Parse json_string into a Python data structure
    data = json.loads(json_string)
    return data

import math

def convert_data(data):
    # print(data, "dataaaaa")
    points = data['points']
    transcription_data = data['transcription']

    # Sort points by their x-coordinate in ascending order
    points_sorted = sorted(points, key=lambda p: p[0])

    # Reverse the order of points
    points_reversed = points_sorted[::-1]

    return {
        "transcription": transcription_data,
        "points": points_sorted
    }



import numpy as np

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def count_closest(dictionary):
    count = 0
    if 'closest' in dictionary:
        count += 1
        count += count_closest(dictionary['closest'])
    return count

def flatten_dictionary(dictionary, result=None):
    if result is None:
        result = {}
    if 'transcription' in dictionary and 'points' in dictionary:
        result[dictionary['transcription']] = dictionary['points']
    if 'closest' in dictionary:
        flatten_dictionary(dictionary['closest'], result)
    return result

def rename_keys(dictionary):
    return {'Column' + str(i + 1): v for i, (k, v) in enumerate(dictionary.items())}

def sort_by_y(processed_results):
    # Custom sorting function to sort by the first y-coordinate of the first dictionary in each list
    def custom_sort(item):
        return item[0]['points'][0][1]  # Returns the y-coordinate of the first point in the first dictionary
    sorted_results = sorted(processed_results, key=custom_sort)
    
    return sorted_results
def remove_points(processed_results): #Function for removal of the values "points" from the processed_results. 
    new_results = [[{key: value for key, value in item.items() if key != 'points'} for item in sublist] for sublist in processed_results]
    return new_results

args = utility.parse_args()
text_sys = TextSystem(args)


def get_extreme_points(data):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for item in data:
        for point in item['points']:
            x, y = point
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

    return {'top_left': [min_x, min_y], 'top_right': [max_x, min_y],
            'bottom_left': [min_x, max_y], 'bottom_right': [max_x, max_y]}
def crop_image_with_cv2(img, extreme_points, padding_ratio=0.1):
    # Define corners with the existing points
    top_left = (extreme_points['top_left'][1], extreme_points['top_left'][0])
    bottom_right = (extreme_points['bottom_right'][1], extreme_points['bottom_right'][0])
    
    # Calculate the width and height of the bounding box
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    
    # Define padding for width and height
    padding_width = int(width * padding_ratio)
    padding_height = int(height * padding_ratio)

    # Create new points with the padding, ensuring they stay within the image bounds
    padded_top_left = (max(top_left[0] - padding_height, 0), max(top_left[1] - padding_width, 0))
    padded_bottom_right = (min(bottom_right[0] + padding_height, img.shape[0]), min(bottom_right[1] + padding_width, img.shape[1]))

    cropped_img = img[padded_top_left[0]:padded_bottom_right[0], padded_top_left[1]:padded_bottom_right[1]]
    
    return cropped_img
import math
is_visualize = True
font_path = args.vis_font_path
drop_score = args.drop_score
draw_img_save_dir = args.draw_img_save_dir
os.makedirs(draw_img_save_dir, exist_ok=True)
def process_json(img, image_file):
    # Get the bounding boxes, recognition results and time dictionary
    dt_boxes, rec_res, time_dict = text_sys(img)

    # Process results into desired format
    res = [{
        "transcription": rec_res[i][0],
        "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
    } for i in range(len(dt_boxes))]
    retval, buffer = cv2.imencode('.jpg', img)
    img_as_text = base64.b64encode(buffer).decode()

    save_pred = img_as_text + "\t" + json.dumps(res, ensure_ascii=False) + "\n"
    save_results = []
    save_results.append(save_pred)
    if is_visualize:
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt(
            image,
            boxes,
            txts,
            scores,
            drop_score=drop_score,
            font_path=font_path)
        
        save_file = image_file
        cv2.imwrite(
            os.path.join(draw_img_save_dir,
                            os.path.basename(save_file)),
            draw_img[:, :, ::-1])
        logger.debug("The visualized image saved in {}".format(
            os.path.join(draw_img_save_dir, os.path.basename(
                save_file))))

    final_json = extract_data(save_results[0])
    return final_json


import math
def calculate_angle_between_transcriptions(transcription1, transcription2):
    points1 = transcription1['points']
    points2 = transcription2['points']

    # Calculate mid-points of the transcriptions
    mid_point1 = ((points1[0][0] + points1[1][0]) / 2, (points1[0][1] + points1[1][1]) / 2)
    mid_point2 = ((points2[0][0] + points2[1][0]) / 2, (points2[0][1] + points2[1][1]) / 2)

    # Calculate the slope of the line between the mid-points
    if mid_point2[0] - mid_point1[0] == 0:
        return 90  # The line is vertical, so the angle is 90 degrees
    else:
        slope = (mid_point2[1] - mid_point1[1]) / (mid_point2[0] - mid_point1[0])
    
    # Calculate the angle of the line (in degrees)
    angle = math.atan(slope) * (180 / math.pi)  # convert from radians to degrees
    
    return angle
def process_pair_slope(result):
    
    result_copy = result.copy()
    angle_list = []
    count_nw = 0
    total_count = []
    while count_nw < len(result_copy):
        nw_result = result.copy()
        current_item = nw_result.pop(count_nw)
        points = current_item.get("points")
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points
        closest_point = None
        min_distance = np.inf
        line_eq = None

        count_left = 0
        count_right = 0
        
        new_row = []
        new_row.append(current_item)
        if current_item in total_count:
            count_nw+=1
            continue
        total_count.append(current_item)
        for other_item in nw_result[:]:
            points_j = other_item.get("points")
            (x11, y11), (x22, y22), (x33, y33), (x44, y44) = points_j
            # Calculate mid points of current and other item
            mid_point_current = ((x3+x4)/2, (y3+y4)/2)
            mid_point_j = ((x11+x22)/2, (y11+y22)/2)
            other_point11 = (x11, y11)
            other_point22 = (x22, y22)

            dist11 = distance(mid_point_current, other_point11)
            dist22 = distance(mid_point_current, other_point22)
            dist = dist11+dist22
            intercept = 0  # Assign a default value
            if (x4-x3) != 0:
                slope = (y4 - y3) / (x4 - x3)
                perpendicular_slope = -1 / slope if slope != 0 else np.inf
                intercept = mid_point_current[1] - perpendicular_slope * mid_point_current[0]
                line_eq = lambda x: perpendicular_slope * x + intercept if perpendicular_slope != np.inf else x3
                # print("firrrrssst intercept", intercept)
            else:
                slope = np.inf
                perpendicular_slope = 0
                intercept = mid_point_current[1]
                line_eq = lambda x: mid_point_current[1]
                # print("second intercept", intercept)

            
            # Calculate intersection point
            x11_sign = y11-(line_eq(x11))
            x22_sign = y22-(line_eq(x22))

            if (x11_sign * x22_sign) < 0 and dist < min_distance:
                # if dist < min_distance:
                closest_point = other_item
                min_distance = dist
            else:
                not_matched = other_item
                current_item = new_row[-1]
                points = current_item.get("points")
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points
                mid_point_current = ((x3+x4)/2, (y3+y4)/2)
                mid_point_j = ((x11+x22)/2, (y11+y22)/2)
                other_point11 = (x11, y11)
                other_point22 = (x22, y22)
                intercept = 0  # Assign a default value
                if (x22-x11) != 0:
                    slope = (y22 - y11) / (x22 - x11)
                    perpendicular_slope = -1 / slope if slope != 0 else np.inf
                    intercept = mid_point_j[1] - perpendicular_slope * mid_point_j[0]
                    line_eq = lambda x: perpendicular_slope * x + intercept if perpendicular_slope != np.inf else x11
                    # print("firrrrssst intercept", intercept)
                else:
                    slope = np.inf
                    perpendicular_slope = 0
                    intercept = mid_point_j[1]
                    line_eq = lambda x: mid_point_j[1]
                x3_sign = y3-(line_eq(x3))
                x4_sign = y4-(line_eq(x4))
                dist11 = distance(mid_point_current, other_point11)
                dist22 = distance(mid_point_current, other_point22)
                dist = dist11+dist22
                if (x3_sign * x4_sign) < 0 and dist < min_distance:
                    # if dist < min_distance:
                    closest_point = other_item
                    min_distance = dist
                        
                # print("not matchedddddd", current_item, other_item)

            if closest_point is not None and closest_point not in total_count:
                # print("agnele calculation", closest_point, current_item)
                total_count.append(closest_point)
                angle_list.append(calculate_angle_between_transcriptions(closest_point, current_item))
            
        count_nw+= 1
    if angle_list:
        mean = sum(angle_list)/len(angle_list)
    else:
        mean = 0
    
    # print(mean)
    # exit(-1)
    return mean


def calculate_average_angle(data):
    angles = []
    for item in data:
        points = item['points']
        x1, y1 = points[0]
        x2, y2 = points[1]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        angle = math.atan(slope)
        angles.append(math.degrees(angle))
    if angles:
        average_angle = sum(angles) / len(angles)
    else:
        average_angle = 0
    # print("average angles", len(angles), "data length", len(data))
    # print("averageeeeeeeeee", average_angle)
    
    return average_angle # negative because rotation direction depends on coordinate system

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return result

    return corrected_data
def extract_and_sort_transcriptions(data):
    transcriptions = []
    for item in data:
        transcriptions.append((item['transcription'], item['points'][0][1]))
    sorted_transcriptions = sorted(transcriptions, key=lambda x: x[1])
    return [transcription[0] for transcription in sorted_transcriptions]

def process_data(final_json):
    full_result = []
    for res in final_json:
        converted_data = convert_data(res)
        full_result.append(converted_data)
    result = sorted(full_result, key=lambda x: x['points'][1])
    return result

def process_image(image_file):
    img = cv2.imread(image_file)
    final_json = process_json(img, image_file)
    nw_processed = extract_and_sort_transcriptions(final_json)
    final_string = ""
    for item in nw_processed:
        s = ""
        s = s +" | "+item+" |"
        s += "\n"
        final_string+= s
    return final_string

import openai
openai.api_key = "" #Put your OPENAI API kEY
model_name = "gpt-4"

def generate_result(user_input):
    sys_content = "As an AI language model your job is to extract information \
    from the given string. The output must be in json format such that the keys are 'Perons's name', 'Contact info', \
    'Business name', 'Business Contact info', 'Location' and any other information as 'miscellaneous'."

    
    mymessages = [{"role": "system", "content": sys_content}]
    mymessages.append({"role": "user", "content": user_input})
    temperature = 1.0
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=mymessages,
        temperature=temperature
    )

    assistant_output = response['choices'][0]['message']['content']
    # print("assistant_output", assistant_output)
    # exit(-1)
    return {"Info": assistant_output}

def read_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

def save_data_as_csv(data, csv_file="data.csv"):
    # Extract the keys and values from the data
    headers = []
    rows = []
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                # print("itemmmm", item)
                if not isinstance(item, str):
                    keys_list = list(item.keys())
                    values_list = list(item.values())
                    new_list = []
                    for i in range(len(values_list)):
                        new_list.append(str(keys_list[i])+" : "+str(values_list[i]))
                    rows.append(new_list)
                else:
                    rows.append([key, value])
        else:
            # headers.append(key)
            rows.append([key, value])

    # Write the data to the CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Data saved to {csv_file} successfully.")

def process_data_and_query(final_string):
    # user_input = final_string
    sys_content = "As an AI assistant your job is to extract information related to visiting cards from the bag of words given to you as input. You will be given queries by the user,\
     based on that you have to extract data. Properly format the data before sending the final output."

    while(True):
        user_query = input('Ask query->')
        mymessages = [{"role": "system", "content": sys_content+final_string}]
        mymessages.append({"role": "user", "content": user_query})
        temperature = 1.0
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=mymessages,
            temperature=temperature
        )

        assistant_output = response['choices'][0]['message']['content']
        print(assistant_output)
    

def main():
    image_file = args.image_dir
    final_string = process_image(image_file)
    print("output data", final_string)
    data = generate_result(final_string)
    print("data", data)
    process_data_and_query(final_string)


if __name__ == "__main__":
    #command: python app.py --image_dir="/media/yobi/hugeDrive/Paddleocr/ocr-api/IMG_3277.jpg" --det_model_dir="./tools/en_PP-OCRv3_det_infer" --rec_model_dir="./tools/en_PP-OCRv3_rec_infer" --use_angle_cls=false
    main()


