import torch
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import  TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords


class YOLOV7ObjectDetector():
    def __init__(
        self,
        img_size: int,
        weights_path: str,
        trace: bool
    ):  
        # initializing variables 
        self.weights_path = weights_path
        self.trace = trace
        self.device = self.__set_device()
        self.half = self.device.type != 'cpu'
        self.model = self.__load_model()
        self.stride = int(self.model.stride.max())
        self.img_size =  check_img_size(img_size, s=self.stride)
        
        # extra model configs 
        self.__config_model(img_size)
        
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, img_size, img_size).to(self.device).type_as(next(self.model.parameters())))
        
    def __set_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __load_model(self):
        return attempt_load(self.weights_path, map_location=self.device)

    def __config_model(self, img_size):
        if self.trace:
            self.model = TracedModel(self.model, self.device, img_size)
        if self.half:
            self.model.half()
    
    def __load_image(self, img0):
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0
    
    def pred(
        self, 
        img,
        augment: bool = False,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45, 
        classes: list = None,
        agnostic_nms: bool = False
    ):
        
        img, im0 = self.__load_image(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad(): 
            pred = self.model(img, augment=augment)[0]
            
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
          
        det = pred[0]  
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
        return det.detach().cpu().numpy()