from flask import Flask, request, jsonify, send_file, send_from_directory, session
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
from markdown import markdown
from xhtml2pdf import pisa
import base64
from hospital_locator import find_nearby_orthopedic_hospitals

from io import BytesIO

# ==== Load environment variables ====
load_dotenv()
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
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

# ==== Groq LLM Setup ====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

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

# ==== Medical Suggestion ====
def generate_suggestion(severity, confidence, chat_history=None):
    try:
        context = ""
        if chat_history:
            context = "**Previous conversation:**\n"
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

        Format your response in markdown with proper headers and bullet points.
        """

        payload = {
            "model": "qwen-qwq-32b",
            "messages": [
                {"role": "system",
                 "content": "You are an expert orthopedic AI assistant providing structured medical reports in markdown format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 300
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            return content
        else:
            raise Exception(f"Groq API Error: {response.status_code}")

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

# ==== Chatbot with History ====
def generate_chatbot_response(user_input, chat_history=None):
    system_prompt = (
        "You are an expert orthopedic AI assistant. "
        "Help with fracture assessments, image interpretation, and treatment suggestions. "
        "Consider the conversation history to provide contextually relevant responses. "
        "Format your responses in markdown with proper headers, bullet points, and emphasis where appropriate. "
        "Do NOT include <think> tags, internal thoughts, or reasoning steps. "
        "Only return the final response directly to the user."
    )

    messages_for_llm = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages_for_llm.extend(chat_history[-10:])

    messages_for_llm.append({"role": "user", "content": user_input})

    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": messages_for_llm,
        "temperature": 0.7,
        "max_tokens": 300
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        raw_text = response.json()['choices'][0]['message']['content'].strip()
        cleaned = clean_llm_response(raw_text)
        return cleaned
    else:
        return f"## Error\nFailed to get response from server (Status: {response.status_code})"
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Retrieve form data
        message = request.form.get("message", "").strip()
        file = request.files.get("image", None)
        chat_history = request.form.get("chat_history")
        chat_history = json.loads(chat_history) if chat_history else None

        # Validate required fields
        if not message and not file:
            return jsonify({"error": "Message or image is required."}), 400

        filename = None
        report_summary, annotated_url = "", ""

        # Process the uploaded image if provided
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
                return jsonify({"error": f"Image analysis failed: {str(e)}"}), 500

        # Generate chatbot response
        final_prompt = f"{report_summary}\n\nUser says: {message}" if report_summary else message
        reply = generate_chatbot_response(final_prompt, chat_history)

        return jsonify({
            "response": reply,
            "report_summary": report_summary,
            "annotated_image_url": annotated_url
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# ==== Download Annotated Image ====
@app.route("/get_annotated/<filename>")
def get_annotated_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

@app.route("/nearby_hospitals", methods=["POST"])
def nearby_hospitals():
    try:
        data = request.get_json()
        lat = data.get("lat")
        lng = data.get("lng")

        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required."}), 400

        hospitals = find_nearby_orthopedic_hospitals(lat, lng)
        return jsonify({"hospitals": hospitals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==== Clear Chat History ====
@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return jsonify({"message": "Chat history cleared."})


# Revised PDF generation using xhtml2pdf for better formatting

def generate_pdf_from_markdown(md_content, image_path=None):
    # Convert markdown to HTML
    html_body = markdown(md_content)

    # Embed the image if provided
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode("utf-8")
            img_tag = (
                f'<div style="text-align:center; margin-top:20px;">'
                f'<img src="data:image/png;base64,{encoded_img}" '
                f'style="max-width:100%; height:auto; border:1px solid #ccc; padding:5px;"/>'
                f'</div>'
            )
        html_body += img_tag

    # Wrap in basic HTML structure with embedded CSS
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.5; padding: 20px; }}
        h1, h2, h3 {{ color: #2e3d49; }}
        ul {{ margin-left: 1em; }}
        .footer {{ text-align: center; font-size: 0.8em; color: #888; margin-top: 40px; }}
      </style>
    </head>
    <body>
      {html_body}
      <div class="footer">Generated by X-Ray Buddy</div>
    </body>
    </html>
    """

    # Generate the PDF
    pdf_path = "temp_report.pdf"
    with open(pdf_path, "wb") as f:
        pisa_status = pisa.CreatePDF(full_html, dest=f)
    if pisa_status.err:
        return None, "PDF generation failed. Ensure the HTML content is compatible with xhtml2pdf."
    return pdf_path, None


# Updated Flask route for PDF download
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    report_md = request.form.get("report_md", "")
    annotated_image = request.form.get("annotated_image", "")

    if not report_md:
        return jsonify({"error": "No report content provided."}), 400

    image_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_image) if annotated_image else None

    pdf_path, error = generate_pdf_from_markdown(report_md, image_path=image_path)
    if error:
        return jsonify({"error": error}), 500

    return send_file(
        pdf_path,
        as_attachment=True,
        mimetype='application/pdf',
        download_name='fracture_report.pdf'
    )



# ==== Run the App ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
