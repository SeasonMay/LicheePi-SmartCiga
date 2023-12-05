from flask import Flask, request, jsonify
from flask import current_app
import logging
import json
import urllib
import numpy as np
import cv2
import requests
import torchvision.models as t_models
import torch

app = Flask(__name__)
device = torch.device('cpu')

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

# load 2 models

import model
from models.experimental import attempt_load

with torch.no_grad():
    weights = './train_model/yolo_best_model.pt'
    yolo_model = attempt_load(weights, map_location=device)

    original_model = t_models.__dict__['vgg16'](pretrained=True)
    ft_model = model.FineTuneModel(original_model, 'vgg16', 49)

    checkpoint = torch.load('train_model/model_best.pth.tar', map_location=device)
    ft_model.load_state_dict(checkpoint['state_dict'], False)
    ft_model.eval()
    app.logger.warning('load models')


# detect image
def detect(data, image_url, image_name, yolo_model, ft_model):
    print('begin to detect!')
    current_app.logger.info('detect image')

    detect_array = detect_cig(image_url, image_name, yolo_model, ft_model)
    counters = []
    for i in range(3):
        counter = {'counterIndex': i + 1, 'resultPhotoUrl': 'http://139.196.180.158/static/result/%s' % (image_name),
                   'grids': detect_array[i]}

    result = {"recordId": data['recordId'], 'mctCode': data['mctCode'], 'msCode': data['msCode'], 'counters': counters}

    url = 'http://139.196.180.158/api/counter/photo/result'
    r = requests.post(url, data=result)
    print(r.text)


@app.route("/")
@app.route("/api/counter/photo/upload", methods=['POST'])
def home():
    # get json
    if request.method == 'POST':
        data = request.json
        # data = jsonify(request.json)
        image_url = data['counters'][0]['photoUrl']
        image_name = image_url.split('/')[-1]
        print('data is ', data, 'image_url', image_url, 'image_name', image_name)
        current_app.logger.warning('data=%s, image_url=%s, image_name=%s' % (str(data), image_url, image_name))

        # image_bytes = requests.get(image_url).content
        # image_np1 = np.frombuffer(image_bytes, dtype=np.uint8)
        # image_np2 = cv2.imdecode(image_np1, cv2.IMREAD_COLOR)

        # process image
        executor.submit(detect, data, image_url, image_name, yolo_model, ft_model)

        result = {'code': 0, 'desc': ""}
        return jsonify(result)

    result = {'code': 0, 'desc': ''}
    return jsonify(result)


if __name__ == "__main__":
    app.debug = True
    handler = logging.FileHandler('log/flask.log')
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0', port=5000)
