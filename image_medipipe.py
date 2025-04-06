
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLO Model
img_pth = "train/images/V_1_F75_jpg.rf.40391029bc87100b77d413a35a86cfa9.jpg"
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize MediaPipe Pose for key points detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load Image
image = cv2.imread(img_pth)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Run YOLOv8 Prediction
results = model(source=img_pth)
res_plotted = results[0].plot()  # Plotted image from YOLO output

# Run MediaPipe Keypoint Detection
pose_results = pose.process(image_rgb)

# Draw Pose Landmarks if Detected
if pose_results.pose_landmarks:
    mp_drawing.draw_landmarks(
        res_plotted, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

# Display Image with Both YOLO and MediaPipe Results
cv2.imshow("YOLO + MediaPipe Key Points", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()

