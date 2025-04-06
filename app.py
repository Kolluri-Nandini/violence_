import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO Model
model = YOLO("best.pt")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_image(image_path):
    """Process an image with YOLO and MediaPipe Pose detection."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)[0]
    res_plotted = results.plot()

    pose_results = pose.process(image_rgb)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            res_plotted, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    output_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(output_path, res_plotted)
    return output_path

def generate_video_feed(video_path):
    """Process and stream video frame by frame."""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)[0]
        res_plotted = results.plot()

        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                res_plotted, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        ret, buffer = cv2.imencode('.jpg', res_plotted)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
def generate_real_time_feed():
    """Stream real-time webcam processing with detections above 80% confidence."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)[0]

        # Start with a copy of the original frame for custom drawing
        output_frame = frame.copy()

        # Loop through detected boxes and only draw those with confidence >= 0.8 (80%)
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= 0.8:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, f"{conf*100:.0f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Process pose landmarks and draw them on the output frame
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                output_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Landing Page
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# Image Detection Routes
# -----------------------------
@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded"
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    processed_path = process_image(file_path)
    # Send paths relative to the static folder for display
    return render_template('image.html',
                           original_image=file_path,
                           detected_image=processed_path)

# -----------------------------
# Video Detection Routes
# -----------------------------
@app.route('/video')
def video_page():
    return render_template('video.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file uploaded"
    file = request.files['video']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return render_template('video.html', video_feed=file_path)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', '')
    return Response(generate_video_feed(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# Camera (Real-Time) Detection Routes
# -----------------------------
@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/real_time_feed')
def real_time_feed():
    return Response(generate_real_time_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
