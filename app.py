from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
MODEL_PATH = "mnt/data/best.pt"  # Ensure this is the correct model path
model = YOLO(MODEL_PATH)

# Upload folders
UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = "annotated_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# Load Hugging Face LLM (GPT-based for better text generation)
try:
    llm = pipeline("text-generation", model="gpt2")
    print("LLM Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading LLM model: {str(e)}")
    llm = None  # Fallback if LLM fails to load

# Severity classification threshold
SEVERITY_THRESHOLD = 0.5


def draw_annotations(image_path, results, output_path):
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



def generate_suggestion(severity,confidence):
    if severity == "No Fracture":
        return {
            "severity_assessment": "No Fracture Detected",
            "urgency": "No medical intervention required.",
            "recommendations": [
                "Maintain a healthy diet and regular exercise for bone strength.",
                "Ensure proper calcium and vitamin D intake.",
                "Stay hydrated and avoid excessive stress on bones."
            ],
            "complexity": "None"
        }

    if not llm:
        # Fallback if LLM fails to load
        return {
            "severity_assessment": f"{severity} Fracture Detected",
            "urgency": "Consult a doctor immediately." if severity == "Severe" else "Schedule a doctor's appointment.",
            "recommendations": [
                "Avoid putting weight on the affected area.",
                "Use ice packs to reduce swelling.",
                "Take over-the-counter pain relievers if necessary."
            ],
            "complexity": "High" if severity == "Severe" else "Low"
        }

    # Specific prompt for medical recommendations based on severity
    prompt = f"""
    A patient has a bone fracture classified as {severity}.
    Provide a medical recommendation, including:
    1. Severity assessment (e.g., 'Severe Fracture Detected').
    2. Urgency (e.g., 'Consult a doctor immediately' or 'Schedule a doctor's appointment').
    3. Recommended treatments (e.g., 'Use a cast', 'Avoid weight-bearing activities').
    4. Complexity level (e.g., 'High', 'Moderate', 'Low').
    Ensure the response is strictly medical and relevant to bone fractures.
    """

    try:
        response = llm(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"].split(". ")
        return {
            "severity_assessment": response[0] if len(response) > 0 else "No data",
            "urgency": response[1] if len(response) > 1 else "No data",
            "recommendations": response[2:] if len(response) > 2 else ["No data"],
            "complexity": "High" if "surgery" in response[0].lower() else "Low" if "rest" in response[0].lower() else "Moderate"
        }
    except Exception as e:
        return {
            "error": f"Failed to generate LLM response: {str(e)}",
            "severity_assessment": f"{severity} Fracture Detected",
            "urgency": "Consult a doctor immediately." if severity == "Severe" else "Schedule a doctor's appointment.",
            "recommendations": [
                "Avoid putting weight on the affected area.",
                "Use ice packs to reduce swelling.",
                "Take over-the-counter pain relievers if necessary."
            ],
            "complexity": "High" if severity == "Severe" else "Low"
        }

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, runs YOLOv8 detection, and generates AI-based suggestions."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath)
        results = model(image)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    detections, highest_confidence, most_severe = [], 0, "No Fracture"
    try:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                severity = "Severe" if confidence > SEVERITY_THRESHOLD else "Mild"

                if confidence > highest_confidence:
                    highest_confidence = confidence
                if severity == "Severe":
                    most_severe = "Severe"

                detections.append({"bbox": [x1, y1, x2, y2], "confidence": confidence, "severity": severity})
    except Exception as e:
        return jsonify({"error": f"Error during object detection: {str(e)}"}), 500

    patient_suggestion = generate_suggestion(most_severe, highest_confidence)
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
    draw_annotations(filepath, results, annotated_path)

    return jsonify({
        "detections": detections,
        "annotated_image_url": f"/download/{filename}",
        "patient_suggestion": patient_suggestion
    })

# Define the path for annotated images
ANNOTATED_FOLDER = "annotated_images"

@app.route("/get_annotated/<filename>")
def get_annotated_image(filename):
    """Fetch the annotated image from the 'annotated_images' folder."""
    return send_from_directory(ANNOTATED_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)