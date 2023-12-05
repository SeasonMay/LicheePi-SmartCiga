import numpy as np
import cv2
import os
import serial
import time

# 串口配置
port = "/dev/ttyS1"  # 串口号
baudrate = 9600  # 波特率

# 合成命令
SYN_StopCom = [0xFD, 0x00, 0x02, 0x02, 0xFD]  # 停止合成
SYN_SuspendCom = [0xFD, 0x00, 0x02, 0x03, 0xFC]  # 暂停合成
SYN_RecoverCom = [0xFD, 0x00, 0x02, 0x04, 0xFB]  # 恢复合成
SYN_ChackCom = [0xFD, 0x00, 0x02, 0x21, 0xDE]  # 状态查询
SYN_PowerDownCom = [0xFD, 0x00, 0x02, 0x88, 0x77]  # 进入POWER DOWN状态命令


def uart_send(ser, data):
    ser.write(data)
    ser.flush()


def SYN_FrameInfo(ser, music, hz_data):
    # 需要发送的文本
    frame_info = []

    # 帧固定配置信息
    frame_info.extend([0xFD, 0x00, len(hz_data) + 3, 0x01, 0x01 | (music << 4)])

    # 校验码计算
    ecc = 0
    for byte in frame_info:
        ecc ^= byte
    for byte in hz_data:
        ecc ^= byte

    # 发送帧信息
    frame_info.extend(hz_data)
    frame_info.append(ecc)

    uart_send(ser, frame_info)


def image_preprocess(image, target_size):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized

    return image_padded


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes=read_class_names("coco.names"), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = (255, 0, 0)
        bbox_thick = 1
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (255, 255, 255), bbox_thick, lineType=cv2.LINE_AA)

    return image


# input size
input_hight = 384
input_width = 640
# 打开串口
ser = serial.Serial(port, baudrate)

print(" ********** preprocess image **********")
original_image = cv2.imread("360.jpg")
rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

image_preprocessed = image_preprocess(np.copy(rgb_image), [input_hight, input_width])
time.sleep(5)
image_preprocessed = image_preprocessed / 255.0
img_ndarray = np.array(image_preprocessed).astype("float32")
img_ndarray = img_ndarray.transpose(2, 0, 1)
img_ndarray.tofile("image_preprocessed.tensor", "\n")
img_ndarray.tofile("image_preprocessed.bin")

model_inference_command = "./yolov5s_example ./hhb_out/hhb.bm image_preprocessed.bin"

os.system(model_inference_command)

print(" ********** processing **********")
bboxes = []
with open("detect.txt", 'r') as f:
    x_min = f.readline().strip()

    while x_min:
        y_min = f.readline().strip()
        x_max = f.readline().strip()
        y_max = f.readline().strip()
        probability = f.readline().strip()
        cls_id = f.readline().strip()
        bbox = [float(x_min), float(y_min), float(x_max), float(y_max), float(probability), int(cls_id)]
        print(bbox)
        bboxes.append(bbox)
        x_min = f.readline().strip()

image_data = image_preprocess(np.copy(original_image), [input_hight, input_width])
image = draw_bbox(image_data, bboxes)

cv2.imwrite("result.jpg", image)
