from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
 
if __name__ == '__main__':
 
 
    # Load Model
    model = YOLO("./ultralytics/cfg/models/11/yolo11_EfficientNet.yaml") # Put the path to the modified yaml file
    # Use the model
    results = model.train(data=r"/home/jye/GitHub/YOLO11_EfficientNet/Dataset/data.yaml", # Put the path of the dataset yaml file
                          epochs=300, batch=16, imgsz=640, workers=4, name=Path(model.cfg).stem) 