import sys
sys.path.append('yolov7')

import os
import cv2
import time
import math
import torch
from deep_sort import DeepSort
from yolov7 import YOLOV7ObjectDetector
from yolov7.utils.torch_utils import  time_synchronized


class ObjectTracker():
    def __init__(
        self, 
        img_size: int = 1280,
        yolo_trace: bool = False
    ):
        # loading YoloV7
        with torch.no_grad():
            self.yolo = YOLOV7ObjectDetector(weights_path=os.getenv('YOLO_MODEL_WEIGHTS'), img_size=img_size, trace=yolo_trace)
        
        #classes
        self.classes = self.yolo.model.module.names if hasattr(self.yolo.model, 'module') else self.yolo.model.names
        self.classes_to_use = ['motorcycle', 'bus', 'truck', 'car']
        self.colors = [(180, 30, 10), (41, 63, 171), (27, 194, 77), (171, 16, 227)]
        
        # creating DeepSort object
        self.deep_sort = DeepSort(self.classes, self.classes_to_use)

    def detect(self, video_path):
        
        video_cap = cv2.VideoCapture(video_path)
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        center_of_frame = (frame_width//2, frame_height//2)
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        print('Frames per second of video: %d' % fps)
        
        prev_frame_time = 0
        new_frame_time = 0
        vehicle_count = {'motorcycle': [], 'bus': [], 'truck': [], 'car': []}
        
        while(True):
            ret, frame = video_cap.read()
            if not ret:
                break
            new_frame_time = time.time()
            
            t1 = time_synchronized()
            pred = self.yolo.pred(frame.copy(), classes=[2, 3, 5, 7]) # car, motorcycle, bus, truck
            t2 = time_synchronized()
            print(f'\nYoloV7: ({(1E3 * (t2 - t1)):.1f}ms)')
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if len(pred)>=1:
                t3 = time_synchronized()
                tracker = self.deep_sort.get_tracker(frame, pred)
                t4 = time_synchronized()
                print(f'DeepSORT: ({(1E3 * (t4 - t3)):.1f}ms)')
                print(f'Total: ({(1E3 * (t4 - t1)):.1f}ms)')
                
                for track in  tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    
                    color = self.classes_to_use.index(class_name)
                    center = ((int(bbox[0])+int(bbox[2]))//2, (int(bbox[1])+int(bbox[3]))//2)
                    dist = int(abs(center[1] - center_of_frame[1]))
                    if dist<=5:
                        if track.track_id not in vehicle_count[class_name]:
                            vehicle_count[class_name].append(track.track_id)
                    
                    cv2.circle(frame, center, 6, (255, 255, 255), -1) 
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.colors[color], 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), self.colors[color], -1)
                    
                    cv2.putText(img=frame,
                                text=f"{class_name} : {track.track_id}",
                                org=(int(bbox[0]), int(bbox[1]-11)),
                                fontFace=0, 
                                fontScale=0.6, 
                                color=(255,255,255), 
                                thickness=1, 
                                lineType=cv2.LINE_AA
                            )    
            
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time 
            fps = str(int(fps))
            
            text=""
            for key in vehicle_count.keys():
                text += f"{key}: {len(vehicle_count[key])}\n"
                
            y0, dy = 100, 30
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (33, 49, 194), 2, cv2.LINE_AA)
                
            cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (0, frame_height//2), (frame_width, frame_height//2), (0, 255, 0), 2)
            resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            cv2.imshow('Object Tracking', resized)    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_cap.release()
        cv2.destroyAllWindows()