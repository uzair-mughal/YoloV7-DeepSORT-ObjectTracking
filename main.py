import os
from dotenv import load_dotenv
from distutils.util import strtobool
from object_tracktor import ObjectTracker

load_dotenv('config/.env')

object_tracktor = ObjectTracker(img_size=int(os.getenv('IMAGE_SIZE')), yolo_trace=strtobool(os.getenv('YOLO_TRACE')))
object_tracktor.detect(video_path=os.getenv('VIDEO_PATH'))