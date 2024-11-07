from ultralytics import YOLO
import cv2
from pathlib import Path

def main():
    # Load the trained YOLO model
    model = YOLO("/home/jye/GitHub/YOLO11_EfficientNet/Yolo11mEfficientNet/weights/Yolo11mEfficientNet.pt")

    # Load the video
    video_path = "/home/jye/GitHub/YOLO11_EfficientNet/city.mp4"  # Video Path
    cap = cv2.VideoCapture(video_path)
    output_path = "output_inference_video.mp4"  # Output Video Path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Draw the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the output video
        out.write(annotated_frame)

        # Optionally, display the frame in real-time
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
