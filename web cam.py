import cv2
import math
import torch
import cvzone
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLO Model
model = YOLO("best.pt")

# Initialize MediaPipe Pose for keypoint detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Input Video Path
video_capture = cv2.VideoCapture(0)

# Get Video Frame Dimensions
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Video Writer to Save Output
output_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

# Class Labels
classnames = ['violence', 'non-violence']

with torch.no_grad():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit if no frame is read

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO Detection
        results = model(source=frame)[0]  # Extract only the first result

        # Run MediaPipe Keypoint Detection
        pose_results = pose.process(frame_rgb)

        # Draw Pose Landmarks if Detected
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Extract Bounding Boxes and Class Information
        boxes = results.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100)
            Class = int(box.cls[0])

            if confidence > 2:  # Reduced confidence threshold from 5% to 2%
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw Bounding Box on Original Frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%',
                                   [x1 + 8, y1 + 100], scale=1.5, thickness=2)

        # Show Video Frame with Annotations
        cv2.imshow("YOLOv8 + MediaPipe Pose Detection", frame)

        # Write Frame to Output Video
        out.write(frame)

        # Press 'q' to Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release Resources
video_capture.release()
out.release()
cv2.destroyAllWindows()
