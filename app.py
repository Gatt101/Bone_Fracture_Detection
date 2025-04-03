from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
from ultralytics import YOLO
import requests
import re
import json
from dotenv import load_dotenv

# ==== Load environment variables ====
load_dotenv()
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_dev_key")

# ==== Config ====
UPLOAD_FOLDER = "uploads"
ANNOTATED_FOLDER = "annotated_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# ==== YOLO Model ====
MODEL_PATH = os.getenv("MODEL_PATH", "mnt/data/best.pt")
model = YOLO(MODEL_PATH)

# ==== Severity Config ====
SEVERITY_THRESHOLD = float(os.getenv("SEVERITY_THRESHOLD", "0.5"))

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

# ==== Clean response ====
def clean_llm_response(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned

# ==== Medical Suggestion using Ollama ====
def generate_suggestion(severity, confidence, chat_history=None):
    context = ""
    if chat_history:
        context += "**Previous conversation:**\n"
        for msg in chat_history[-5:]:
            context += f"**{msg['role']}:** {msg['content']}\n"
        context += "\n"

    prompt = f"""
    {context}
    A patient has a bone fracture detected via X-ray.
    The model confidence is {confidence:.2f} and severity is classified as {severity}.

    Please provide a structured orthopedic medical recommendation in markdown format with these sections:
    - **Severity Assessment**
    - **Urgency Level**
    - **Recommendations** (as a bulleted list)
    - **Treatment Complexity** (High, Moderate, Low, None)

    Format your response in markdown.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "chatdoc:latest",
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            content = response.json().get("response", "").strip()
            return clean_llm_response(content)
        else:
            raise Exception(f"Ollama API Error: {response.status_code}")
    except Exception as e:
        return f"""## Error
{str(e)}

## Severity Assessment
{severity} Fracture Detected

## Urgency Level
{"Consult a doctor immediately." if severity == "Severe" else "Schedule a doctor's appointment."}

## Recommendations
- Avoid putting weight on the affected area
- Use ice packs to reduce swelling
- Take over-the-counter pain relievers if necessary

## Treatment Complexity
{"High" if severity == "Severe" else "Moderate"}"""

# ==== Chatbot with History using Ollama ====
def generate_chatbot_response(user_input, chat_history=None):
    context = ""
    if chat_history:
        for msg in chat_history[-10:]:
            context += f"**{msg['role'].capitalize()}:** {msg['content']}\n"
        context += "\n"

    prompt = f"""
    {context}
    {user_input}
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "chatdoc:latest",
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            content = response.json().get("response", "").strip()
            return clean_llm_response(content)
        else:
            return f"## Error\nOllama failed with status: {response.status_code}"
    except Exception as e:
        return f"## Error\nOllama call failed: {str(e)}"

# ==== Conversational Endpoint ====
@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "")
    file = request.files.get("image", None)
    chat_history = request.form.get("chat_history")
    chat_history = json.loads(chat_history) if chat_history else None

    filename = None
    report_summary, annotated_url = "", ""

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

            report = generate_suggestion(most_severe, highest_confidence, chat_history)
            report_summary = report

            annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
            draw_annotations(filepath, results, annotated_path)
            annotated_url = f"/get_annotated/{filename}"

        except Exception as e:
            return jsonify({"error": f"## Error\nImage analysis failed: {str(e)}"}), 500

    final_prompt = f"{report_summary}\n\nUser says: {message}" if report_summary else message
    reply = generate_chatbot_response(final_prompt, chat_history)

    return jsonify({
        "response": reply,
        "report_summary": report_summary,
        "annotated_image_url": annotated_url
    })

# ==== Download Annotated Image ====
@app.route("/get_annotated/<filename>")
def get_annotated_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

# ==== Clear Chat History ====
@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return jsonify({"message": "Chat history cleared."})

# ==== Run the App ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
