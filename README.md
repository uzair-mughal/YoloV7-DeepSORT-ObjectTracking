# YoloV7-DeepSORT-ObjectTracking


## Run Locally

Clone the project

```bash
  git clone https://github.com/uzair-mughal/YoloV7-DeepSORT-ObjectTracking
```

Go to the project directory

```bash
  cd YoloV7-DeepSORT-ObjectTracking
```

Install dependencies

```bash
  conda env create -f config\environment.yml
```

Activate conda environment

```bash
  conda activate object-tracking
```
Install PyTroch

```bash
  pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

Downlaod YOLOV7 Weights and place under "yolov7\weights"

```bash
  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
```

Check config\.env file for basic configs and add video path

```bash
  YOLO_MODEL_WEIGHTS = 'yolov7/weights/yolov7-w6.pt'
  TRACED_MODEL_WEIGHTS = 'yolov7/weights/traced_model.pt'
  ENCODER_WEIGHTS_PATH = 'deep_sort/weights/mars-small128.pb'

  IMAGE_SIZE = 1280
  YOLO_TRACE = 'false'
  VIDEO_PATH = 'inference/video1.mp4'
```
Test

```bash
  python main.py
```

## Demo

![image](https://user-images.githubusercontent.com/49607360/230225071-a7b4c853-34d2-473a-9049-db32a443c848.png)
