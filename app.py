from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
from ultralytics import YOLO
import requests
import re


app = Flask(__name__)
CORS(app)

# ==== Config ====
UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = "annotated_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# ==== YOLO Model ====
MODEL_PATH = "mnt/data/best.pt"  # Replace with your actual model path
model = YOLO(MODEL_PATH)

# ==== Groq LLM Setup ====
GROQ_API_KEY = "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"  # Replace with actual key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# ==== Severity Config ====
SEVERITY_THRESHOLD = 0.5


# ==== Annotation Utility ====
def draw_annotations(image_path, results, output_path):
    img = cv2.imread(image_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            severity = "Severe" if confidence > SEVERITY_THRESHOLD else "Mild"
            color = (0, 0, 255) if severity == "Severe" else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{severity} ({confidence:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, img)


def generate_suggestion(severity, confidence):
    try:
        # Create a dynamic medical prompt
        prompt = f"""
        A patient has a bone fracture detected via X-ray.
        The model confidence is {confidence:.2f} and severity is classified as {severity}.

        Please provide a structured orthopedic medical recommendation in JSON format with these fields:
        - severity_assessment
        - urgency
        - recommendations (as a list of 3 specific suggestions)
        - complexity (High, Moderate, Low, None)

        Ensure your response is strictly formatted as a Python dictionary (not plain text).
        """

        payload = {
            "model": "qwen-2.5-32b",
            "messages": [
                {"role": "system", "content": "You are an expert orthopedic AI assistant providing structured medical reports."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 300
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            report_dict = eval(content)  # Parse the returned string as Python dictionary
            return report_dict
        else:
            return {
                "error": f"Groq API Error: {response.status_code}",
                "severity_assessment": f"{severity} Fracture Detected",
                "urgency": "Consult a doctor immediately." if severity == "Severe" else "Schedule a doctor's appointment.",
                "recommendations": [
                    "Avoid putting weight on the affected area.",
                    "Use ice packs to reduce swelling.",
                    "Take over-the-counter pain relievers if necessary."
                ],
                "complexity": "High" if severity == "Severe" else "Moderate"
            }

    except Exception as e:
        return {
            "error": f"Failed to generate report using Groq: {str(e)}",
            "severity_assessment": f"{severity} Fracture Detected",
            "urgency": "Consult a doctor immediately." if severity == "Severe" else "Schedule a doctor's appointment.",
            "recommendations": [
                "Avoid putting weight on the affected area.",
                "Use ice packs to reduce swelling.",
                "Take over-the-counter pain relievers if necessary."
            ],
            "complexity": "High" if severity == "Severe" else "Moderate"
        }


def clean_llm_response(text):
    # Remove <think>...</think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned


def generate_chatbot_response(user_input):
    system_prompt = (
        "You are an expert orthopedic AI assistant. "
        "Help with fracture assessments, image interpretation, and treatment suggestions. "
        "Do NOT include <think> tags, internal thoughts, reasoning steps, or reflective statements. "
        "Only return the final response directly to the user."
    )

    payload = {
        "model": "qwen-2.5-32b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        raw_text = response.json()['choices'][0]['message']['content'].strip()
        return clean_llm_response(raw_text)
    else:
        return f"Error: {response.status_code} - {response.text}"

# ==== Conversational Endpoint (chat + optional image) ====
@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "")
    file = request.files.get("image", None)
    filename = None
    report_summary = ""
    annotated_url = ""

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image = Image.open(filepath)
            results = model(image)
            detections, highest_confidence, most_severe = [], 0, "No Fracture"

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

            report = generate_suggestion(most_severe, highest_confidence)
            report_summary = f"Severity: {report['severity_assessment']}, Urgency: {report['urgency']}, Recommendations: {', '.join(report['recommendations'])}"

            annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
            draw_annotations(filepath, results, annotated_path)
            annotated_url = f"/get_annotated/{filename}"

        except Exception as e:
            return jsonify({"error": f"Image analysis failed: {str(e)}"}), 500

    # Combine user message with report for the LLM
    final_prompt = f"{report_summary}\n\nUser says: {message}" if report_summary else message
    reply = generate_chatbot_response(final_prompt)

    return jsonify({
        "response": reply,
        "report_summary": report_summary,
        "annotated_image_url": annotated_url
    })


# ==== Download Annotated Image ====
@app.route("/get_annotated/<filename>")
def get_annotated_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)


# ==== Run the App ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
