import io
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
from PIL import Image

from general import check_img_size, non_max_suppression, scale_coords, get_report, plot_one_box
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import xyxy2xywh


class Inference:

    def __init__(self, weights_path):
        self.weights_path = [weights_path]

        print(weights_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(self.weights_path, map_location=self.device)

        self.stride = int(self.model.stride.max())
        self.image_size = 640

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).to(torch.float32))

    def run(self, data: bytes):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        source = deepcopy(img)
        img = letterbox(img, self.image_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).to(torch.float32)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        old_img_w = old_img_h = self.image_size
        old_img_b = 1

        # Warmup
        if self.device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            for i in range(3):
                _ = self.model(img)[0]

        with torch.no_grad():
            prediction = self.model(img)[0]

        prediction = non_max_suppression(prediction, 0.496)[0]

        if not len(prediction):
            result = io.BytesIO()
            Image.fromarray(source).save(result, format='JPEG')
            result = result.getvalue()
            return result, get_report([])

        prediction[:, :4] = scale_coords(img.shape[2:], prediction[:, :4], source.shape).round()

        boxes = []

        for *xyxy, conf, cls in reversed(prediction):
            label = f'{self.names[int(cls)]}'
            plot_one_box(xyxy, source, label=label, color=self.colors[int(cls)], line_thickness=2)

            gn = torch.tensor(source.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            boxes.append([*xywh, cls.item()])

        result = io.BytesIO()
        Image.fromarray(source).save(result, format='JPEG')
        result = result.getvalue()
        return result, get_report([self.names[int(cls)] for cls in prediction[:, -1]])
