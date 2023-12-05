import os
import time

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as t_models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset
from tqdm import tqdm

from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized
from sklearn.cluster import DBSCAN

from config import *

img_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def unusual_address(area_list, detections, rois, loc_list, input_type):
    mean, std = np.mean(area_list, axis=0), np.std(area_list, axis=0)
    remove_list_index = []
    for i, x in enumerate(area_list):
        if not (mean - 3 * std) < x < (mean + 3 * std):
            remove_list_index.append(i)
    print('unusual_address', len(remove_list_index), input_type)
    if len(remove_list_index):
        for counter, index in enumerate(remove_list_index):
            index = index - counter
            if input_type == 0:
                detections[0].pop(index)
                rois.pop(index)
            loc_list.pop(index)
        print(f'{len(remove_list_index)} unusual results have been removed!')
    if input_type == 0:
        return detections, rois, loc_list
    else:
        return loc_list


def plot_boxes(raw_img, loc_list, color):
    if color == 0:
        color_line = (0, 255, 0)
    else:
        color_line = (0, 0, 0)
    for x in loc_list:
        (x1, y1), (x2, y2) = x[0], x[1]
        cv2.rectangle(raw_img, (x1, y1), (x2, y2), color=color_line, thickness=1)


def extra_empty_address(loc_list_ci, loc_list_la, img_input):
    textsize = int(min(img_input.shape[0], img_input.shape[1]) / 100)
    fontStyle = ImageFont.truetype("simsun.ttc", textsize, encoding="utf-8")

    for i in range(len(loc_list_la)):
        flag = 0
        x1, y1, x2, y2 = loc_list_la[i][0][0], loc_list_la[i][0][1], loc_list_la[i][1][0], loc_list_la[i][1][1]
        loc_x, loc_y1, loc_y2, loc_y3 = int((x1 + x2) / 2), y1 + (y1 - y2) / 2, y1 + (y1 - y2), y1 + (y1 - y2) * 1.5
        # print(y1, y2, y1 + (y1-y2)/3, 2*y1-y2)
        for j in range(len(loc_list_ci)):
            x11, y11, x12, y12 = loc_list_ci[j][0][0], loc_list_ci[j][0][1], loc_list_ci[j][1][0], loc_list_ci[j][1][1]
            if x11 < loc_x < x12 and (y11 < loc_y1 < y12 or y11 < loc_y2 < y12 or y11 < loc_y3 < y12):
                flag = 1
                # img_ = img0.copy()
                # cv2.rectangle(img_, (x11, y11), (x12, y12), (255, 255, 255), thickness=-1)
                # cv2.circle(img_, (loc_x, loc_y), 4, (0, 0, 0), thickness=-1)
                # cv2.imshow('ori', img_)
                # cv2.waitKey(0)
                break
        if flag == 0:  # 未识别
            print('*' * 10, flag, x1, x2, y1, y2, loc_y1, loc_y2, loc_y3)
            '''cv2.rectangle(img_input, (x1, y1 + y1 - y2), (x2, y1), color=(255, 255, 255), thickness=-1)
            x, y = int((x1 + x2) / 2), int((3 * y1 - y2) / 2)
            cv2.putText(img_input, 'E', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)'''
            img_input = np.ascontiguousarray(img_input)
            x_center = int((x2 - x1) / 2 + x1)
            y_center = int(y1 + (y2 - y1) / 2) - 60

            cv2.circle(img_input, (x_center, y_center), int((x2 - x1) / 4), color=(0, 255, 0), thickness=-1)
            x, y = int((x1 + x2) / 2), int((3 * y1 - y2) / 2)
            # cv2.putText(img_input, 'empty', (x_center-textsize, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if isinstance(img_input, np.ndarray):  # 判断是否OpenCV图片类型
                img_input = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_input)
            draw.text((x_center - textsize, y_center), '空位', (0, 0, 0), font=fontStyle)
            img_input = cv2.cvtColor(np.asarray(img_input), cv2.COLOR_RGB2BGR)
    return img_input


def detect_cig(raw_img, model, device, img_size=640, conf_thresh=0.3, iou_thresh=0.1, color_line=(18, 157, 218)):
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, stride)
    print('img size', img_size)

    # img process
    img0 = raw_img.copy()

    print('copy img', img0.shape)
    img = letterbox(img0, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    print('transpose img', img.shape)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # Norm
    print('torch img', img.shape)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print('normalize img', img.shape)

    # predict
    t1 = time_synchronized()
    # print(img.shape)
    h, w = raw_img.shape[0], raw_img.shape[1]
    print('img.shape: ', img.shape)
    pred = model(img, augment=False)[0]  # store_true = false
    # print(pred)

    # NMS apply
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=False)
    # print(pred)
    t2 = time_synchronized()
    detections_ci = []
    rois_ci = []
    count = 0
    # process prediction
    det = pred[0]
    current = []
    loc_list_ci, loc_list_la = [], []
    area_list_ci, area_list_la = [], []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_img.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            if cls == 0:  # Write to file
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x_centre = int((x1 + x2) / 2)
                y_centre = int((y1 + y2) / 2)
                roi = raw_img[y1:y2, x1:x2]

                if roi.sum():
                    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    # area calculate
                    h, w = roi.shape[0], roi.shape[1]
                    area = h * w
                    area_list_ci.append(area)
                    loc_list_ci.append([(x1, y1), (x2, y2)])

                    roi = cv2.resize(roi, (224, 224))

                    current.append([(x_centre, y_centre)])
                    current[count].append(0)
                    current[count].append(roi)

                    img = Image.fromarray(np.uint8(roi))
                    roi = img_trans(img)
                    rois_ci.append(roi.numpy())

                    detections_ci.append(current)
                    count += 1
            elif cls == 1:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                roi = raw_img[y1:y2, x1:x2]
                if roi.sum():
                    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    # area calculate
                    h, w = roi.shape[0], roi.shape[1]
                    area = h * w
                    area_list_la.append(area)
                    loc_list_la.append([(x1, y1), (x2, y2)])
    detections_ci, rois_ci, loc_list_ci = unusual_address(area_list_ci, detections_ci, rois_ci, loc_list_ci, 0)
    plot_boxes(raw_img, loc_list_ci, 0)
    loc_list_la = unusual_address(area_list_la, [], [], loc_list_la, 1)
    plot_boxes(raw_img, loc_list_la, 1)

    return detections_ci, rois_ci


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else:
            raise "Finetuning not supported on this architecture yet"

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def classify(test_loader, model, weights=[0.2, 0.8]):
    results = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # compute output
            input = torch.tensor([np.array(item).astype(float) for item in input], dtype=torch.float,
                                 device=torch.device('cpu'))
            input.squeeze_(dim=0)
            print(input.shape)
            if isinstance(model, list):
                output = torch.zeros((input.shape[0], 241))
                for w, m in zip(weights, model):
                    o = torch.softmax(m(input), dim=1)
                    output += o * w
                    print('classify', output.shape, o.shape, output, o)

            else:
                output = model(input)
            results.extend(torch.argmax(output.data, axis=1).detach().cpu().numpy())

            _, indices = torch.max(output, 1)
            percentage = torch.softmax(output, dim=1)[0] * 100
            # if percentage[0] < 0.1:
            #    results[i] = 241

    return results


def add_chinese_text(img, text, position, textColor=(192, 255, 62), textSize=18):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def add_chinese_text_v3(img, text, x1, y1, x2, y2, jishuqi=1, textColor=(0, 0, 0)):
    # textsize = int((x2 - x1) / len(text)) + 1

    textsize = int(min(img.shape[0], img.shape[1]) / 100)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textsize, encoding="utf-8")

    # 创建背景
    colorline = (144, 238, 144)
    # 绘制文本

    x_center = int((x2 - x1) / 2 + x1)
    y_center = int(y1 + (y2 - y1) / 2)
    x_length = len(text) * textsize
    x_start = int(x_center - x_length / 2)
    x_end = int(x_center + x_length / 2)

    # print(x1, y1, x2, y2, x_center, x_start, x_end, colorline)
    if '空位' in text:
        cv2.circle(img, (x_center, y_center), int((x2 - x1) / 4), color=(0, 255, 0), thickness=-1)
        x, y = int((x1 + x2) / 2), int((3 * y1 - y2) / 2)
        # cv2.putText(img, '空位', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, textColor, 2)
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text((x_center - textsize, y_center), '空位', textColor, font=fontStyle)

    else:
        if jishuqi == 1:
            y = int(0.2 * (y2 - y1) + y1)
            x = x1
            cv2.rectangle(img, (x_start, y1), (x_end, int(y1 + textsize * 1.2)), color=colorline, thickness=-1)
            if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            draw.text((x_start, y1), text, textColor, font=fontStyle)

        else:
            y = int(y2 - textsize * 1.2)
            cv2.rectangle(img, (x_start, int(y2 - textsize * 1.2)), (x_end, y2), color=colorline, thickness=-1)
            if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            draw.text((x_start, y), text, textColor, font=fontStyle)

    # 转换回OpenCV格式
    img_return = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # cv2.imshow('ori', img_return)
    # cv2.waitKey(0)

    return img_return, abs(1 - jishuqi)


def DBSCAN_transform(coords, eps):
    results = []
    co_results = []
    x, y = [], []
    coords = np.array(coords)
    y = coords[:, 1].reshape(-1, 1)
    x = coords[:, 0].reshape(-1, 1)
    print('y', coords.shape, coords[0], y.shape, y[0])
    clustering = DBSCAN(eps=eps, min_samples=2).fit(y)
    return clustering.labels_


def img_process_v1(img_url, yolo_model, classify_model, mode=0):
    # load img
    print('get image url', img_url)
    img_bytes = requests.get(img_url).content
    img_np1 = np.frombuffer(img_bytes, dtype=np.uint8)
    img_input = cv2.imdecode(img_np1, cv2.IMREAD_COLOR)

    ori_img = img_input.copy()
    print('input image', img_input.shape)

    with torch.no_grad():
        # detect yolo result
        result, images = detect_cig(img_input, model=yolo_model, device=torch.device('cpu'))
        print(np.array(images).shape)
        images = torch.tensor(np.array(images).astype(float))
        images = TensorDataset(images)
        print(len(result[0]), 'result in total')

        # make classify image dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset=images,
            batch_size=len(result[0]),
            shuffle=False,
            num_workers=0,
        )

        # classify each tiny pic and display on raw image
        classes = classify(test_loader, classify_model)
        # print(classes)

        loc = []
        # store all classify results
        labels_results = []
        for i in range(len(result[0])):
            x_center, y_center = result[0][i][0][0], result[0][i][0][1]
            loc.append([x_center, y_center])

        # DBSCAN sort
        coords = np.array(loc)
        # sort by col
        coords_sort = coords[np.lexsort(coords.T)]
        classes = np.array(classes)[np.lexsort(coords.T)]
        index_list = DBSCAN_transform(coords_sort, eps=30)

        # means split
        # capture index
        max_value = max(index_list) + 1
        num_list = []
        for i in range(max_value):
            list_curr = [x for (x, y) in enumerate(index_list) if y == i]
            num_list.append(len(list_curr))
        # print(num_list)
        curr = 0
        col = []
        for i in range(len(num_list)):
            list_split = coords_sort[curr:curr + num_list[i]]
            list_split_c = classes[curr:curr + num_list[i]]
            list_split_t = np.c_[list_split, list_split_c]
            col.append(list_split_t)
            curr += num_list[i]

        # sort by row

        for i in range(len(col)):
            col[i] = col[i][np.lexsort(col[i][:, ::-1].T)]

        info = []
        for i in range(len(col)):
            col_curr = col[i]
            for j in range(len(col_curr)):
                col_curr_ = col_curr[j]
                # cv2.putText(img_input, str(f'{i}+{j}'), (col_curr[j][0], col_curr[j][1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,(100, 200, 200), 5)
                x_center, y_center, label = col_curr_[0], col_curr_[1], col_curr_[2]
                label_cn = class_cn_v1[label]
                img_input = add_chinese_text(img_input, f'{label_cn}', (x_center - 30, y_center - 50))
                info.append({
                    "rowIndex": f'{i + 1}',
                    "colIndex": f'{j + 1}',
                    # "resultType": 3 if labels_results[i] == '空位' else 1,
                    "name": f'{label_cn}'
                })

        cv2.imwrite(f'static/result/result.jpg', img_input)
        with open('static/result/result.txt', 'w') as f:
            for i in range(len(info)):
                f.write(str(info[i]))
                print(info[i])


def img_process_v2(img_url, image_name, yolo_model, classify_model, mode=1):
    print('image process v2', img_url, image_name, mode)
    if mode == 1:
        print('read url content', img_url)
        img_bytes = requests.get(img_url).content

        img_np1 = np.frombuffer(img_bytes, dtype=np.uint8)
        img_input = cv2.imdecode(img_np1, cv2.IMREAD_COLOR)
        cv2.imwrite(f'static/upload/%s' % (image_name), img_input)
    else:
        img_input = cv2.imread(r'./static/upload/%s' % (image_name))

    ori_img = img_input.copy()
    print('input image', img_input.shape)

    with torch.no_grad():
        # detect yolo result
        result, images = detect_cig(img_input, model=yolo_model, device=torch.device('cpu'))
        print(np.array(images).shape)
        images = torch.tensor(np.array(images).astype(float))
        images = TensorDataset(images)
        print(len(result[0]), 'result in total')

        # make classify image dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset=images,
            batch_size=len(result[0]),
            shuffle=False,
            num_workers=0,
        )

        # classify each tiny pic and display on raw image
        classes = classify(test_loader, classify_model)
        # print(classes)

        loc = []
        # store all classify results
        labels_results = []
        for i in range(len(result[0])):
            x_center, y_center = result[0][i][0][0], result[0][i][0][1]
            loc.append([x_center, y_center])

        # DBSCAN sort
        coords = np.array(loc)
        # sort by col
        coords_sort = coords[np.lexsort(coords.T)]
        classes = np.array(classes)[np.lexsort(coords.T)]
        index_list = DBSCAN_transform(coords_sort, eps=30)

        # means split
        # capture index
        max_value = max(index_list) + 1
        num_list = []
        for i in range(max_value):
            list_curr = [x for (x, y) in enumerate(index_list) if y == i]
            num_list.append(len(list_curr))
        # print(num_list)
        curr = 0
        col = []
        for i in range(len(num_list)):
            list_split = coords_sort[curr:curr + num_list[i]]
            list_split_c = classes[curr:curr + num_list[i]]
            list_split_t = np.c_[list_split, list_split_c]
            col.append(list_split_t)
            curr += num_list[i]

        # sort by row

        for i in range(len(col)):
            col[i] = col[i][np.lexsort(col[i][:, ::-1].T)]

        info = []
        for i in range(len(col)):
            col_curr = col[i]
            for j in range(len(col_curr)):
                col_curr_ = col_curr[j]
                # cv2.putText(img_input, str(f'{i}+{j}'), (col_curr[j][0], col_curr[j][1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,(100, 200, 200), 5)
                x_center, y_center, label = col_curr_[0], col_curr_[1], col_curr_[2]
                label_cn = class_cn_v2[label]
                img_input = add_chinese_text(img_input, f'{label_cn}', (x_center - 30, y_center - 50))
                info.append({
                    "rowIndex": f'{i + 1}',
                    "colIndex": f'{j + 1}',
                    "resultType": 3 if '空位' in label_cn else 1,
                    "code": '未识别' if label_cn not in name_to_number else f'{name_to_number[label_cn]}',
                    "name": label_cn
                })

        cv2.imwrite(f'static/result/%s' % (image_name), img_input)
        with open('static/result/result.txt', 'w') as f:
            for i in range(len(info)):
                f.write(str(info[i]))
                print(info[i])
    return info


def img_process_loop(img_name, yolo_model, classify_model):
    img_input = cv2.imread(f'testv3/{img_name}')
    ori_img = img_input.copy()
    with torch.no_grad():
        # detect yolo result
        result, images = detect_cig(img_input, model=yolo_model, device=torch.device('cpu'))

        print(len(result[0]), 'result in total')
        classes = [0] * len(result[0])

        loc = []
        # store all classify results
        labels_results = []
        for i in range(len(result[0])):
            x_center, y_center = result[0][i][0][0], result[0][i][0][1]
            loc.append([x_center, y_center])

        # DBSCAN sort
        coords = np.array(loc)
        # sort by col
        coords_sort = coords[np.lexsort(coords.T)]
        classes = np.array(classes)[np.lexsort(coords.T)]
        index_list = DBSCAN_transform(coords_sort, eps=30)

        # means split
        # capture index
        max_value = max(index_list) + 1
        num_list = []
        for i in range(max_value):
            list_curr = [x for (x, y) in enumerate(index_list) if y == i]
            num_list.append(len(list_curr))
        # print(num_list)
        curr = 0
        col = []
        for i in range(len(num_list)):
            list_split = coords_sort[curr:curr + num_list[i]]
            list_split_c = classes[curr:curr + num_list[i]]
            list_split_t = np.c_[list_split, list_split_c]
            col.append(list_split_t)
            curr += num_list[i]

        # sort by row

        for i in range(len(col)):
            col[i] = col[i][np.lexsort(col[i][:, ::-1].T)]

        info = []
        for i in range(len(col)):
            col_curr = col[i]
            for j in range(len(col_curr)):
                col_curr_ = col_curr[j]
                cv2.putText(img_input, str(f'{i}+{j}'), (col_curr[j][0], col_curr[j][1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                            (100, 200, 200), 5)

        cv2.imwrite(f'static/result_loop/{img_name}', img_input)


def detect_cig_v3(raw_img, model, device, img_size=640, conf_thresh=0.3, iou_thresh=0.1, color_line=(18, 157, 218)):
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, stride)
    print('img size', img_size)

    # img process
    img0 = raw_img.copy()

    print('copy img', img0.shape)
    img = letterbox(img0, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    print('transpose img', img.shape)
    img = torch.from_numpy(img).to(device)
    print('torch img', img.shape, type(img))
    img = img.float()  # uint8 to fp32
    print('img to float', img.shape)
    img /= 255.0  # Norm

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print('normalize img', img.shape)

    # predict
    # t1 = time_synchronized()
    # print(img.shape)
    h, w = raw_img.shape[0], raw_img.shape[1]
    print('img.shape: ', img.shape)
    pred = model(img, augment=False)[0]  # store_true = false
    # print(pred)

    # NMS apply
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=False)
    # print(pred)
    # t2 = time_synchronized()
    detections_ci = []
    rois_ci = []
    count = 0
    # process prediction
    det = pred[0]
    current = []
    loc_list_ci, loc_list_la = [], []
    area_list_ci, area_list_la = [], []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_img.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            if cls == 0:  # Write to file
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x_centre = int((x1 + x2) / 2)
                y_centre = int((y1 + y2) / 2)
                roi = raw_img[y1:y2, x1:x2]

                if roi.sum():
                    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    # area calculate
                    h, w = roi.shape[0], roi.shape[1]
                    area = h * w
                    area_list_ci.append(area)
                    loc_list_ci.append([(x1, y1), (x2, y2)])

                    roi = cv2.resize(roi, (224, 224))

                    current.append([(x_centre, y_centre)])
                    current[count].append(0)
                    current[count].append(roi)

                    img = Image.fromarray(np.uint8(roi))
                    roi = img_trans(img)
                    rois_ci.append(roi.numpy())

                    detections_ci.append(current)
                    count += 1
            elif cls == 1:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                roi = raw_img[y1:y2, x1:x2]
                if roi.sum():
                    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    # area calculate
                    h, w = roi.shape[0], roi.shape[1]
                    area = h * w
                    area_list_la.append(area)
                    loc_list_la.append([(x1, y1), (x2, y2)])
    detections_ci, rois_ci, loc_list_ci = unusual_address(area_list_ci, detections_ci, rois_ci, loc_list_ci, 0)

    plot_boxes(raw_img, loc_list_ci, 0)
    loc_list_la = unusual_address(area_list_la, [], [], loc_list_la, 1)
    plot_boxes(raw_img, loc_list_la, 1)
    return detections_ci, rois_ci, loc_list_ci, loc_list_la


def DBSCAN_transform_v3(coords, eps):
    results = []
    co_results = []
    x, y = [], []
    coords = np.array(coords)
    y = coords[:, 2].reshape(-1, 1)
    x = coords[:, 0].reshape(-1, 1)
    print('y', coords.shape, coords[0], y.shape, y[0])
    clustering = DBSCAN(eps=eps, min_samples=2).fit(y)
    return clustering.labels_


def plot_boxes_bar(raw_img, loc_list, type):
    if type == 'blue':
        color_line = (10, 215, 255)
        thickness = 8
    elif type == 'red':
        color_line = (0, 0, 255)
        thickness = 12
    x1, y1, x2, y2 = loc_list[0], loc_list[1], loc_list[2], loc_list[3]
    cv2.rectangle(raw_img, (x1, y1), (x2, y2), color=color_line, thickness=thickness)


def img_process_v3(img_url, image_name, nantong_txt, shop_txt, yolo_model, classify_model, mode=0):
    print('read url content', img_url, mode)
    if mode == 1:
        img_bytes = requests.get(img_url).content
        img_np1 = np.frombuffer(img_bytes, dtype=np.uint8)
        img_input = cv2.imdecode(img_np1, cv2.IMREAD_COLOR)
        cv2.imwrite(f'./static/upload/%s' % (image_name), img_input)

    img_input = cv2.imread(r'./static/upload/%s' % (image_name))

    print('input image name ', r'./static/upload/%s' % (image_name))
    ori_img = img_input.copy()
    print('input image', img_input.shape)

    with torch.no_grad():
        # detect yolo result
        result, images, loc_list, loc_list_la = detect_cig_v3(img_input, model=yolo_model, device=torch.device('cpu'))

        print('loc list ci', np.array(loc_list).shape, loc_list_la)
        # 额外的空位识别
        img_input = extra_empty_address(loc_list, loc_list_la, img_input)

        print('after empty', np.array(images).shape)
        images = torch.tensor(np.array(images).astype(float))
        images = TensorDataset(images)
        print(len(result[0]), 'result in total')

        # make classify image dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset=images,
            batch_size=len(result[0]),
            shuffle=False,
            num_workers=0,
        )

        # classify each tiny pic and display on raw image
        classes = classify(test_loader, classify_model)
        # print(classes)

        loc = []
        # store all classify results
        labels_results = []
        for i in range(len(result[0])):
            x_center, y_center = result[0][i][0][0], result[0][i][0][1]
            x1, y1 = loc_list[i][0]
            x2, y2 = loc_list[i][1]
            # loc.append([x_center, y_center])
            loc.append([x_center, (x1, y1, x2, y2), y_center])

        # DBSCAN sort
        coords = np.array(loc)
        # sort by col
        coords_sort = coords[np.lexsort(coords.T)]
        classes = np.array(classes)[np.lexsort(coords.T)]
        index_list = DBSCAN_transform_v3(coords_sort, eps=30)

        # means split
        # capture index
        max_value = max(index_list) + 1
        num_list = []
        for i in range(max_value):
            list_curr = [x for (x, y) in enumerate(index_list) if y == i]
            num_list.append(len(list_curr))
        # print(num_list)
        curr = 0
        col = []
        for i in range(len(num_list)):
            list_split = coords_sort[curr:curr + num_list[i]]
            list_split_c = classes[curr:curr + num_list[i]]
            list_split_t = np.c_[list_split, list_split_c]
            col.append(list_split_t)
            curr += num_list[i]

        # sort by row

        for i in range(len(col)):
            col[i] = col[i][np.lexsort(col[i][:, ::-1].T)]

        info = []

        # read nantong and shop .txt
        if len(nantong_txt) > 0:
            with open('./static/upload/' + nantong_txt, 'r+', encoding='utf-8') as f:
                nantong_list = [i[:-1].split(',') for i in f.readlines()]
        else:
            with open('./static/nantong.txt', 'r+', encoding='utf-8') as f:
                nantong_list = [i[:-1].split(',') for i in f.readlines()]
        if len(shop_txt) > 0:
            with open('./static/upload/' + shop_txt, 'r+', encoding='utf-8') as f:
                shop_list = [i[:-1].split(',') for i in f.readlines()]
        else:
            with open('./static/shop.txt', 'r+', encoding='utf-8') as f:
                shop_list = [i[:-1].split(',') for i in f.readlines()]

        print('list', nantong_list, shop_list)

        jishuqi = 0
        for i in range(len(col)):
            col_curr = col[i]
            for j in range(len(col_curr)):
                col_curr_ = col_curr[j]
                # cv2.putText(img_input, str(f'{i}+{j}'), (col_curr[j][0], col_curr[j][1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,(100, 200, 200), 5)
                x_center, curr_loc, y_center, label = col_curr_[0], col_curr_[1], col_curr_[2], col_curr_[3]
                # x_center, y_center, label = col_curr_[0], col_curr_[1], col_curr_[2]
                label_cn = class_cn_v2[label]
                x1, y1, x2, y2 = curr_loc[0], curr_loc[1], curr_loc[2], curr_loc[3]
                # plot box
                if label_cn in name_to_number:
                    if str(name_to_number[label_cn]) == '000':
                        color_line = (0, 255, 0)
                        cv2.rectangle(img_input, (x1, y1), (x2, y2), color=color_line, thickness=-1)
                    else:
                        pass
                        # if str(name_to_number[label_cn]) not in shop_list[0]:
                        #     plot_boxes_bar(img_input, curr_loc, type='blue')
                        # if str(name_to_number[label_cn]) not in nantong_list[0]:
                        #     plot_boxes_bar(img_input, curr_loc, type='red')

                if label_cn in name_to_number and '空位' not in label_cn and (
                        len(nantong_list) > 0 or len(shop_list) > 0):
                    print(label_cn, label_cn in name_to_number, len(nantong_list), len(shop_list))
                    # plot red and blue box
                    if str(name_to_number[label_cn]) not in shop_list[0]:
                        plot_boxes_bar(img_input, curr_loc, type='blue')
                    if str(name_to_number[label_cn]) not in nantong_list[0]:
                        plot_boxes_bar(img_input, curr_loc, type='red')

                img_input, jishuqi = add_chinese_text_v3(img_input, f'{label_cn}', x1, y1, x2, y2, jishuqi)

                resultType = 1
                if label_cn == '无法识别':
                    resultType = 4
                elif '空位' in label_cn:
                    resultType = 3

                info.append({
                    "rowIndex": f'{i + 1}',
                    "colIndex": f'{j + 1}',
                    "resultType": resultType,
                    "code": '未识别' if label_cn not in name_to_number else f'{name_to_number[label_cn]}',
                    "name": label_cn
                })

        cv2.imwrite(r'static/result/%s' % (image_name), img_input)
        # cv2.imencode('.jpg', img_input)[1].tofile(r'static/result/%s' % (image_name))
        with open('static/result/result.txt', 'w') as f:
            for i in range(len(info)):
                f.write(str(info[i]))
                print(info[i])
    return info


if __name__ == "__main__":
    from models.experimental import attempt_load

    device = torch.device('cuda', index=0)

    with torch.no_grad():
        weights = './train_model/yolo_best_model_v4.pt'
        yolo_model = attempt_load(weights, map_location=device)

        original_model = t_models.__dict__['vgg16'](pretrained=True)
        ft_model = model.FineTuneModel(original_model, 'vgg16', 239)

        checkpoint = torch.load('./train_model/model_best.pth.tar', map_location=device)
        ft_model.load_state_dict(checkpoint['state_dict'], False)
        ft_model.eval()

    if True:
        start = time.perf_counter()
        detect_array = img_process_v3('01a43ce14df711c676b416fac210e1b.jpg', '', '', yolo_model, ft_model, mode=0)

        end = time.perf_counter()
        print(end - start)
    else:
        dir_name = 'testv3/'
        start = time.perf_counter()
        for filename in tqdm(os.listdir(f'{dir_name}')):
            img_name = filename
            img_process_loop(img_name, yolo_model, model)
        end = time.perf_counter()
        print(end - start)
