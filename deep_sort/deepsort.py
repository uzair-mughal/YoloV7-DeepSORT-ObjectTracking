import os
import numpy as np
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

class DeepSort():
    def __init__(
        self, 
        class_names,
        classes_to_use, 
        nms_max_overlap : float = 1.0,
        max_cosine_distance: float = 0.4,
        nn_budget : int = None
    ):
        self.__class_names =  class_names
        self.__nms_max_overlap = nms_max_overlap
        self.__encoder = gdet.create_box_encoder(model_filename=os.getenv('ENCODER_WEIGHTS_PATH'), batch_size=1)
        self.__metric = nn_matching.NearestNeighborDistanceMetric(
            metric="cosine", matching_threshold=max_cosine_distance, budget=nn_budget)
        self.__tracker = Tracker(self.__metric, max_iou_distance=0.7, max_age=30, n_init=3)
        
    def __get_meta_data(self, yolo_output):
        if yolo_output is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
        else:
            bboxes = yolo_output[:,:4]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            scores = yolo_output[:,4]
            classes = yolo_output[:,-1]
            num_objects = bboxes.shape[0]
            
        names = np.array([self.__class_names[int(classes[i])] for i in range(num_objects)])
        return bboxes, scores, names
    
    def get_tracker(self, img, yolo_output):
        bboxes, scores, class_names = self.__get_meta_data(yolo_output)
        features = self.__encoder(img, bboxes)
        detections = [Detection(bbox, score, feature, class_name) for bbox, score, feature, class_name in zip(bboxes, scores, features, class_names)]
        
        boxs = np.array([d.tlwh for d in detections]) 
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxs, self.__nms_max_overlap, scores)
        detections = [detections[i] for i in indices] 
        
        self.__tracker.predict()
        self.__tracker.update(detections)

        return self.__tracker