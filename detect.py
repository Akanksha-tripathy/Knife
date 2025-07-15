# detect.py

from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import cv2

# Load the YOLOv8 model (smallest version for speed; change to yolov8s.pt if needed)
model = YOLO("yolov8n.pt")

# List of knife-related labels to look for
knife_keywords = ["knife", "cleaver", "dagger", "sword", "machete"]

# Main detection function
def detect_knives_yolo(video_path, sample_every=5, confidence_threshold=0.25):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * fps)

    detections = []

    for frame_idx in range(0, total_frames, sample_every):
        timestamp = round(frame_idx / fps, 2)
        frame = clip.get_frame(frame_idx / fps)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame_bgr)[0]

        for box in results.boxes:
            label_id = int(box.cls.item())
            label = model.names[label_id]
            conf = box.conf.item()

            # Check if label is knife-related and passes threshold
            if any(kw in label.lower() for kw in knife_keywords) and conf >= confidence_threshold:
                detections.append({
                    "frame": frame_idx,
                    "time": timestamp,
                    "label": label,
                    "confidence": round(conf * 100, 2)
                })

    return detections
