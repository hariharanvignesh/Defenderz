from flask import Flask, jsonify, render_template,redirect, url_for,Response, request
from flask_cors import CORS
import requests
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from threading import Thread
import os
import cvzone
from firebase_admin import db




app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'upload/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# # Load YOLOv5 for object detection
# yolo_model = YOLO('yolov5s.pt')

# # Load MobileNetV2 for defect classification
# classifier_model = MobileNetV2(weights='imagenet')





# Firebase base URL
firebase_base_url = "https://museum-d67ee-default-rtdb.firebaseio.com/"

def fetch_firebase_data():
    exhibit_path = f"{firebase_base_url}/machine_data.json"
    response = requests.get(exhibit_path)
    if response.status_code == 200:
        data = response.json()
        if data:
            latest_dynamic_id = list(data.keys())[-1]
            dynamic_path = f"{firebase_base_url}/machine_data/{latest_dynamic_id}.json"
            response_dynamic = requests.get(dynamic_path)
            if response_dynamic.status_code == 200:
                exhibit_data = response_dynamic.json()
                exhibit_data["Current"] = float(exhibit_data.get("Current", 0.0))
                exhibit_data["Vibration"] = int(exhibit_data.get("Vibration", 0))
                return exhibit_data
            else:
                print(f"Failed to retrieve data from {dynamic_path}")
        else:
            print("No data found in /machine_data")
    else:
        print(f"Failed to retrieve data from {exhibit_path}. Error: {response.status_code}")
    return None



data = fetch_firebase_data()

@app.route('/navigate', methods=['GET'])
def navigate():
    firebase_data = fetch_firebase_data()
    if firebase_data and "Type" in firebase_data:
        type_value = firebase_data["Type"]  # Get the type value
        if type_value == 1:
            return redirect(url_for('type1'))
        elif type_value == 2:
            return redirect(url_for('type2'))
        elif type_value == 3:
            return redirect(url_for('type3'))
        elif type_value == 4:
            return redirect(url_for('type4'))
    return jsonify({"error": "Type not found or invalid"}), 404


# Routes for each type
@app.route('/type1', methods=['GET'])
def type1():
    firebase_data = fetch_firebase_data()
    if firebase_data and firebase_data.get("Type") == 1:
        return render_template('type1.html', data=firebase_data)
    return redirect(url_for('admin_dashboard'))



@app.route('/type2', methods=['GET'])
def type2():
    firebase_data = fetch_firebase_data()
    if firebase_data and firebase_data.get("Type") == 2:
        return render_template('type2.html', data=firebase_data)
    return redirect(url_for('admin_dashboard'))


@app.route('/type3', methods=['GET'])
def type3():
    firebase_data = fetch_firebase_data()
    if firebase_data and firebase_data.get("Type") == 3:
        return render_template('type3.html', data=firebase_data)
    return redirect(url_for('admin_dashboard'))


@app.route('/type4', methods=['GET'])
def type4():
    # Render the type4.html template
    return render_template('type4.html')


camera = cv2.VideoCapture(0)  # Replace with IP camera URL, e.g., "http://192.168.x.x:8080/video"

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use yield to stream the frames
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




yolo_model = YOLO('yolov5s.pt')

# Load MobileNetV2 for defect classification
classifier_model = MobileNetV2(weights='imagenet')

# Preprocess frames for defect classification
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = tf.keras.applications.mobilenet_v2.preprocess_input(frame_array)
    return frame_array

# Classify the object as Defect or Normal
def classify_defect(cropped_frame):
    preprocessed = preprocess_frame(cropped_frame)
    predictions = classifier_model.predict(preprocessed)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    top_prediction = decoded_predictions[0][0][1]
    defect_keywords = ["broken", "damaged", "cracked", "faulty"]
    return "Defect" if any(word in top_prediction.lower() for word in defect_keywords) else "Normal"

# Detect motion by comparing bounding box positions across frames
def detect_motion(previous_boxes, current_boxes):
    motion_status = {}
    for i, box in enumerate(current_boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        center_current = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Compare with previous boxes
        for j, prev_box in enumerate(previous_boxes):
            x1_p, y1_p, x2_p, y2_p = prev_box[:4].astype(int)
            center_previous = ((x1_p + x2_p) // 2, (y1_p + y2_p) // 2)

            # Calculate Euclidean distance
            distance = np.linalg.norm(np.array(center_current) - np.array(center_previous))
            if distance > 10:  # Threshold for significant motion
                motion_status[i] = True
                break
        else:
            motion_status[i] = False  # No motion detected
    return motion_status

# Streaming video feed from the camera
camera = cv2.VideoCapture(0)  # Change to the IP camera URL if needed
def generate_frames():
    previous_boxes = []
    motionless_counter = {}

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Object detection with YOLO
            results = yolo_model(frame)
            current_boxes = results[0].boxes.data.cpu().numpy()

            # Detect motion
            motion_status = detect_motion(previous_boxes, current_boxes)

            for i, box in enumerate(current_boxes):
                x1, y1, x2, y2, _, _ = box.astype(int)
                label = "Normal"

                if motion_status.get(i, False):
                    motionless_counter[i] = 0  # Reset counter if motion detected
                else:
                    motionless_counter[i] = motionless_counter.get(i, 0) + 1

                    # If motionless for too long, classify as defect
                    if motionless_counter[i] >= 3:  # Adjust threshold as needed
                        label = "Defect (No Motion)"
                    else:
                        # Crop object and classify for other defects
                        cropped_object = frame[y1:y2, x1:x2]
                        label = classify_defect(cropped_object)

                color = (0, 0, 255) if "Defect" in label else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            previous_boxes = current_boxes  

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = tf.keras.applications.mobilenet_v2.preprocess_input(frame_array)
    return frame_array

# Classify the object as Defect or Normal
def classify_defect(cropped_frame):
    preprocessed = preprocess_frame(cropped_frame)
    predictions = classifier_model.predict(preprocessed)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    top_prediction = decoded_predictions[0][0][1]
    defect_keywords = ["broken", "damaged", "cracked", "faulty"]
    return "Defect" if any(word in top_prediction.lower() for word in defect_keywords) else "Normal"

# Detect motion by comparing bounding box positions across frames
def detect_motion(previous_boxes, current_boxes):
    motion_status = {}
    for i, box in enumerate(current_boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        center_current = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Compare with previous boxes
        for j, prev_box in enumerate(previous_boxes):
            x1_p, y1_p, x2_p, y2_p = prev_box[:4].astype(int)
            center_previous = ((x1_p + x2_p) // 2, (y1_p + y2_p) // 2)

            # Calculate Euclidean distance
            distance = np.linalg.norm(np.array(center_current) - np.array(center_previous))
            if distance > 10:  # Threshold for significant motion
                motion_status[i] = True
                break
        else:
            motion_status[i] = False  # No motion detected
    return motion_status

# Process the uploaded video
def process_video(video_path, frame_interval=30, motion_threshold=3):
    cap = cv2.VideoCapture(video_path)
    previous_boxes = []
    motionless_counter = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection with YOLO
        results = yolo_model(frame)
        current_boxes = results[0].boxes.data.cpu().numpy()

        # Detect motion
        motion_status = detect_motion(previous_boxes, current_boxes)
        
        for i, box in enumerate(current_boxes):
            x1, y1, x2, y2, _, _ = box.astype(int)
            label = "Normal"

            if motion_status.get(i, False):
                motionless_counter[i] = 0  # Reset counter if motion detected
            else:
                motionless_counter[i] = motionless_counter.get(i, 0) + 1

                # If motionless for too long, classify as defect
                if motionless_counter[i] >= motion_threshold:
                    label = "Defect (No Motion)"
                else:
                    # Crop object and classify for other defects
                    cropped_object = frame[y1:y2, x1:x2]
                    label = classify_defect(cropped_object)

            color = (0, 0, 255) if "Defect" in label else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, ( x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        previous_boxes = current_boxes  
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_webcam_stream():
    cap = cv2.VideoCapture(0)  # Open the webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection with YOLO
        results = yolo_model(frame)
        current_boxes = results[0].boxes.data.cpu().numpy()

        for box in current_boxes:   
            x1, y1, x2, y2, _, _ = box.astype(int)
            cropped_object = frame[y1:y2, x1:x2]
            label = classify_defect(cropped_object)

            # Draw bounding box and label
            color = (0, 0, 255) if "Defect" in label else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with detection
        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to stop the webcam stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)

    video_file = request.files['video']

    if video_file.filename == '':
        return redirect(request.url)

    if video_file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        # Process the uploaded video using the defect detection logic
        process_video(video_path)

        return "Video uploaded and processed successfully."


@app.route('/start-webcam', methods=['POST'])
def start_webcam():
    # This triggers the webcam stream and processing in a new thread to avoid blocking the main Flask app
    
    return Response(process_webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Route to stream the video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





model1 = YOLO("yolo11s.pt")
names = model1.model.names
cap1 = cv2.VideoCapture(0)
person_count = 0  # Global variable to track total persons detected

def generate_frames_person_2():
    global person_count
    count = 0
    cy1 = 261
    cy2 = 286
    offset = 8
    enter = []
    exitp = []

    while True:
        success, frame = cap1.read()
        if not success:
            break
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLO detection on the frame
        results = model1.track(frame, persist=True, classes=0)

        person_count = 0  # Reset person count for each frame

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                if c == "person":  # Assuming 'person' is the label for persons in YOLO
                    person_count += 1

        # Show detection on the webpage
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames_person_2(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/type5')
def index():
    return render_template('index5.html', person_count=person_count)



@app.route('/dashboard', methods=['GET'])
def admin_dashboard():
    # Fetch data to display on the dashboard
    firebase_data = fetch_firebase_data()
    return render_template('admin_dashboard.html', data=firebase_data)

@app.route('/data', methods=['GET'])
def send_json():
    firebase_data = fetch_firebase_data()
    if firebase_data:
        return jsonify(firebase_data)
    else:
        return jsonify({"error": "Failed to retrieve data from Firebase"}), 500
@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True, port=5678)
