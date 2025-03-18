from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
MODEL_PATH = "mnt/data/best.pt"
model = YOLO(MODEL_PATH)

# Upload folders
UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = "annotated_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# Severity classification threshold
SEVERITY_THRESHOLD = 0.5

def draw_annotations(image_path, results, output_path):
    """Draws bounding boxes and severity labels on the image."""
    img = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            severity = "Severe" if confidence > SEVERITY_THRESHOLD else "Mild"

            # Draw bounding box
            color = (0, 0, 255) if severity == "Severe" else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add text (severity)
            label = f"{severity} ({confidence:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, img)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load image and run inference
    image = Image.open(filepath)
    results = model(image)

    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            severity = "Severe" if confidence > SEVERITY_THRESHOLD else "Mild"

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class": class_id,
                "severity": severity
            })

    # Annotate and save the image
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
    draw_annotations(filepath, results, annotated_path)

    return jsonify({
        "detections": detections,
        "annotated_image_url": f"http://127.0.0.1:5000/download/{filename}"
    })

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["ANNOTATED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
