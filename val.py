from ultralytics import YOLO
import os

def main():
    # Load the trained model
    model = YOLO("/home/jye/GitHub/YOLO11_EfficientNet/Yolo11mEfficientNet/weights/Yolo11mEfficientNet.pt")
    
    # Validate the model using the test dataset
    results = model.val(
        data="/home/jye/GitHub/YOLO11_EfficientNet/Dataset/data.yaml",  # Path to your dataset YAML
        imgsz=640,
        batch=16,
        workers=4
    )

if __name__ == "__main__":
    main()
