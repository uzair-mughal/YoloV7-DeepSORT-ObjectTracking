# YoloV7-DeepSORT-ObjectTracking

1-conda env create -f config\environment.yml
2-conda activate object-tracking
3-pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
4-downlaod yolo weights and place under "yolov7\weights". https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
5-check env for basic configs and add video path
6-python main.py