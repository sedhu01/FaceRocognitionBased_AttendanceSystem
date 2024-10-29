from flask import Flask, request, render_template, redirect, url_for, flash, Response
import os
import cv2
import numpy as np
import time
import imutils
import pickle
from imutils import paths
import dlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from datetime import datetime
import csv
import mysql.connector

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'attendance_system'
}

# Function to get a database connection
def get_db_connection():
    return mysql.connector.connect(**db_config)


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Directory and file paths
DATASET_DIR = 'dataset'
MODEL_PATH = "model"
OUTPUT_PATH = "output"
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMBEDDING_FILE = os.path.join(OUTPUT_PATH, "embeddings.pickle")
RECOGNIZER_FILE = os.path.join(OUTPUT_PATH, "recognizer.pickle")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_PATH, "le.pickle")
ATTENDANCE_FILE = "attendance.csv"
SHAPE_PREDICTOR = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
FACE_RECOGNITION_MODEL = os.path.join(MODEL_PATH, "dlib_face_recognition_resnet_model_v1.dat")
CONFIDENCE_THRESHOLD = 0.5

# Ensure necessary directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
embedder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)

# Route for dataset creation
@app.route('/create_dataset', methods=['GET', 'POST'])
def create_dataset():
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        
        # Define path to save images for this person
        person_path = os.path.join(DATASET_DIR, name)
        os.makedirs(person_path, exist_ok=True)
        
        # Save student's information in a CSV file
        with open('student.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([name, roll_number])
        
        # Start capturing images
        cam = cv2.VideoCapture(0)
        time.sleep(2.0)  # Warm up the camera
        total = 0
        
        while total < 50:
            ret, frame = cam.read()
            if not ret:
                break

            img = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(img, 1)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img_path = os.path.join(person_path, f"{str(total).zfill(5)}.png")
                cv2.imwrite(img_path, img)
                total += 1

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        flash('Dataset creation successful!', 'success')
        return redirect(url_for('index'))
    
    return render_template('create_dataset.html')

# Route for preprocessing embeddings
@app.route('/preprocess_embeddings', methods=['POST'])
def preprocess_embeddings():
    image_paths = list(paths.list_images(DATASET_DIR))
    
    known_embeddings = []
    known_names = []
    total = 0

    for (i, image_path) in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}")
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = detector(rgb_image, 1)

        for box in boxes:
            shape = predictor(rgb_image, box)
            face_descriptor = embedder.compute_face_descriptor(rgb_image, shape)
            known_names.append(name)
            known_embeddings.append(np.array(face_descriptor))
            total += 1

    data = {"embeddings": known_embeddings, "names": known_names}
    with open(EMBEDDING_FILE, "wb") as f:
        f.write(pickle.dumps(data))

    flash(f'Preprocessed {total} embeddings successfully!', 'success')
    return redirect(url_for('index'))

# Route for training the SVM model
@app.route('/train_model', methods=['POST'])
def train_model():
    if not os.path.exists(EMBEDDING_FILE):
        flash('Embeddings file not found. Please preprocess embeddings first.', 'danger')
        return redirect(url_for('index'))
    
    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    with open(RECOGNIZER_FILE, "wb") as f:
        f.write(pickle.dumps(recognizer))
    with open(LABEL_ENCODER_FILE, "wb") as f:
        f.write(pickle.dumps(label_encoder))

    flash('Model training completed successfully!', 'success')
    return redirect(url_for('index'))

# Store recognized names for attendance
recognized_names = set()

# Function to generate video frames for streaming
def generate_frames():
    # Video capture setup
    cam = cv2.VideoCapture(0)
    # Load the recognizer and label encoder if available
    recognizer = None
    le = None
    if os.path.exists(RECOGNIZER_FILE) and os.path.exists(LABEL_ENCODER_FILE):
        with open(RECOGNIZER_FILE, "rb") as f:
            recognizer = pickle.load(f)
        with open(LABEL_ENCODER_FILE, "rb") as f:
            le = pickle.load(f)

    # Ensure that the recognizer and label encoder are loaded properly
    if recognizer is None or le is None:
        print("Error: Recognizer or label encoder not found. Please train the model first.")
        exit(1)
    
    global recognized_names  # Track recognized names globally
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector(rgb_frame)

            for box in boxes:
                shape = predictor(rgb_frame, box)
                face_embedding = embedder.compute_face_descriptor(rgb_frame, shape)
                preds = recognizer.predict_proba([np.array(face_embedding)])[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j] if proba >= CONFIDENCE_THRESHOLD else "Unknown"

                if name != "Unknown":
                    recognized_names.add(name)  # Add recognized name to the set only if it's not unknown

                # Draw the bounding box and label
                text = f"{name}: {proba * 100:.2f}%" if name != "Unknown" else "Unknown"
                (startX, startY, endX, endY) = (box.left(), box.top(), box.right(), box.bottom())
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route to display webcam stream
@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to log attendance into MySQL database
def log_attendance(name):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert attendance record
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sql = "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)"
    cursor.execute(sql, (name, timestamp))
    conn.commit()

    cursor.close()
    conn.close()


# Route to mark attendance for all recognized faces
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    global recognized_names
    # Log attendance for all recognized faces during the session
    for name in recognized_names:
        log_attendance(name)
        flash(f'Attendance marked for {name}!', 'success')
    
    # Clear the recognized names after attendance is marked
    recognized_names.clear()

    return redirect(url_for('index'))

@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('view_attendance.html', records=records)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
