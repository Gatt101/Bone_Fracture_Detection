from datetime import datetime
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
from markdown import markdown as md_to_html
from xhtml2pdf import pisa
import base64
from hospital_locator import find_nearby_orthopedic_hospitals
from io import BytesIO

# ==== Constants ====
DEFAULT_REPORT_TEMPLATE = """# Fracture Analysis Report

## Severity Assessment
- Classification: {severity}
- Confidence Score: {confidence:.2f}/1.00
- Key Radiographic Features: 
  - Fracture detected with AI assistance
  - Clinical correlation recommended

## Urgency Level
- Priority Level: {priority}
- Recommended Action Timeline: {timeline}
- Triage Considerations: Requires professional evaluation

## Clinical Recommendations
1. Consult an orthopedic specialist
2. Obtain additional imaging if needed
3. Follow standard fracture precautions
4. Monitor for signs of complications

## Treatment Complexity
- Complexity Tier: To be determined by physician
- Potential Interventions: 
  - Will depend on clinical evaluation
  - May require immobilization
"""

# ==== Load environment variables ====
load_dotenv()
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5174"])
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


# Updated draw_annotations function
def draw_annotations(image, results):
    """Draw annotations on the image with better visualization"""
    img = image.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = round(box.conf[0].item(), 2)
            severity = "Severe" if confidence > SEVERITY_THRESHOLD else "Mild"
            color = (0, 0, 255) if severity == "Severe" else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create label text
            label = f"{severity} ({confidence:.2f})"

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Draw label background
            cv2.rectangle(img,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)

            # Draw label text
            cv2.putText(img, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return img


    # ==== Clean response ====
def clean_llm_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ==== Enhanced Medical Suggestion Function ====
def generate_suggestion(severity, confidence, chat_history=None):
    try:
        # Structured prompt with strict formatting
        prompt = f"""**Generate orthopedic report using this exact template:**

# Fracture Analysis Report

## Severity Assessment
- Classification: {severity}
- Confidence Score: {confidence:.2f}/1.00
- Key Radiographic Features: [List 3-4 specific findings]

## Urgency Level
- Priority Level: [Emergency/Urgent/Semi-Urgent]
- Recommended Action Timeline: [Immediate/Within 24h/Within 72h]
- Triage Considerations: [Brief justification]

## Clinical Recommendations
1. [Imaging protocol]
2. [Pain management]
3. [Mobility restrictions]
4. [Specialist referral]
5. [Follow-up schedule]

## Treatment Complexity
- Complexity Tier: [High/Medium/Low]
- Potential Interventions: 
  - [Primary procedure]
  - [Alternative options]
  - [Rehabilitation plan]

**Rules:**
1. Use medical terminology from AO Fracture Classification
2. Never leave section headers without content
3. Confidence score must use XX.XX% format
4. List items must use bullet points"""

        payload = {
            "model": "qwen-qwq-32b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an orthopedic radiologist. Generate COMPLETE medical reports using the provided template. Never omit sections. Use precise medical terminology."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']

            # Enhanced validation and cleanup
            required_sections = {
                "Severity Assessment": r"## Severity Assessment\n(.*?)(?=\n## |$)",
                "Urgency Level": r"## Urgency Level\n(.*?)(?=\n## |$)",
                "Clinical Recommendations": r"## Clinical Recommendations\n(.*?)(?=\n## |$)",
                "Treatment Complexity": r"## Treatment Complexity\n(.*?)(?=\n## |$)"
            }

            for section, pattern in required_sections.items():
                if not re.search(pattern, content, re.DOTALL):
                    content += f"\n## {section}\n- Content generation failed - manual review required"

            # Term standardization
            replacements = {
                r"\bSoupe\b": "Severe",
                r"\bCommituted\b": "Comminuted",
                r"\b(\d+) bony fragments\b": r"\1 bone fragments",
                r"Confidence Level:(\d+)%": r"Confidence Level: \1%"
            }

            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)

            return content

        # Fallback to default template if API fails
        priority = "Emergency" if severity == "Severe" else "Urgent"
        timeline = "Immediate" if severity == "Severe" else "Within 24 hours"
        return DEFAULT_REPORT_TEMPLATE.format(
            severity=severity,
            confidence=confidence,
            priority=priority,
            timeline=timeline
        )

    except Exception as e:
        priority = "Emergency" if severity == "Severe" else "Urgent"
        timeline = "Immediate" if severity == "Severe" else "Within 24 hours"
        return DEFAULT_REPORT_TEMPLATE.format(
            severity=severity,
            confidence=confidence,
            priority=priority,
            timeline=timeline
        )


# ==== Enhanced PDF Generation with Debugging ====
def generate_pdf_from_markdown(md_content, image_path=None):
    try:
        # Convert markdown to HTML with error handling
        try:
            body_html = md_to_html(
                md_content,
                extensions=['tables', 'fenced_code'],
                output_format='html5'
            )
        except Exception as e:
            return None, f"Markdown conversion failed: {str(e)}"

        # Simplified PDF-safe CSS
        css = """
        body {
            font-family: Helvetica;
            font-size: 12pt;
            line-height: 1.5;
            margin: 1cm;
        }
        h1 {
            color: #333;
            font-size: 18pt;
            margin-bottom: 0.5cm;
        }
        h2 {
            color: #444;
            font-size: 14pt;
            margin-top: 1cm;
        }
        ul {
            margin-left: 0.5cm;
        }
        li {
            margin-bottom: 0.3cm;
        }
        .image-container {
            text-align: center;
            margin: 1cm 0;
        }
        .annotated-image {
            max-width: 80%;
        }
        """

        # Build HTML template
        html_template = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>{css}</style>
        </head>
        <body>
            <h1>Fracture Analysis Report</h1>
            {body_html}"""

        # Handle image inclusion
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_template += f"""
                <div class="image-container">
                    <h2>Annotated Image</h2>
                    <img class="annotated-image" 
                         src="data:image/png;base64,{encoded_img}">
                    <p>AI-generated fracture annotations</p>
                </div>
                """
            except Exception as e:
                html_template += f"""
                <div style="color: red;">
                    Image loading error: {str(e)}
                </div>
                """
        else:
            html_template += """
            <div style="color: #666;">
                No annotated image available
            </div>
            """

        html_template += "</body></html>"

        # Generate PDF with error capture
        pdf_buffer = BytesIO()
        pisa_status = pisa.CreatePDF(
            html_template,
            dest=pdf_buffer,
            encoding='UTF-8',
            default_css=css
        )

        # Handle PDF generation errors
        if pisa_status.err:
            error_log = "\n".join([
                f"Error {i + 1}: {err}"
                for i, err in enumerate(pisa_status.log)
            ])
            return None, f"PDF generation failed: {error_log}"

        if pdf_buffer.tell() == 0:
            return None, "PDF buffer is empty"

        pdf_buffer.seek(0)
        return pdf_buffer, None

    except Exception as e:
        return None, f"PDF creation error: {str(e)}"


# ==== Updated Download Endpoint with Debugging ====
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    try:
        # Get input data
        report_md = request.form.get("report_md", "")
        annotated_img = request.form.get("annotated_image", "")

        if not report_md:
            return jsonify({"error": "No report content provided"}), 400

        # Validate and sanitize image path
        image_path = None
        if annotated_img:
            try:
                safe_filename = secure_filename(annotated_img)
                image_path = os.path.join(app.config['ANNOTATED_FOLDER'], safe_filename)
                if not os.path.exists(image_path):
                    app.logger.error(f"Image not found: {image_path}")
                    image_path = None
            except Exception as e:
                app.logger.error(f"Image path error: {str(e)}")
                image_path = None

        # Generate PDF with error handling
        pdf_buffer, error = generate_pdf_from_markdown(report_md, image_path)
        if error:
            app.logger.error(f"PDF Generation Error: {error}")
            return jsonify({
                "error": "PDF generation failed",
                "details": error
            }), 500

        # Validate PDF buffer
        if not pdf_buffer or pdf_buffer.getbuffer().nbytes == 0:
            return jsonify({"error": "Empty PDF generated"}), 500

        # Send file response
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Fracture_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )

    except Exception as e:
        app.logger.error(f"PDF Download Error: {str(e)}")
        return jsonify({
            "error": "PDF download failed",
            "details": str(e)
        }), 500


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
        # Validate request content type
        if not request.content_type.startswith('multipart/form-data'):
            return jsonify({"error": "Content-Type must be multipart/form-data"}), 400

        # Retrieve form data with proper error handling
        try:
            message = request.form.get("message", "").strip()
            file = request.files.get("image", None)
            chat_history = request.form.get("chat_history", "[]")
            chat_history = json.loads(chat_history)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid chat_history format"}), 400
        except Exception as e:
            return jsonify({"error": f"Form data processing error: {str(e)}"}), 400

        # Validate at least one input is provided
        if not message and not file:
            return jsonify({"error": "Either message or image must be provided"}), 400

        # Initialize response variables
        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_url": "",
            "detections": []
        }

        # Process image if provided
        if file:
            # Validate file
            if not file.filename:
                return jsonify({"error": "No file selected"}), 400

            try:
                # Secure filename and create paths
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)

                # Save original file
                file.save(upload_path)

                # Process image with model
                img = cv2.imread(upload_path)
                if img is None:
                    raise ValueError("Failed to read image file")

                results = model(img)
                detections = []
                highest_confidence = 0
                most_severe = "No Fracture"

                # Process detections
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = round(box.conf[0].item(), 2)
                        severity = "Severe" if confidence > SEVERITY_THRESHOLD else "Mild"

                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            most_severe = severity

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "severity": severity
                        })

                # Generate report if fractures detected
                if detections:
                    response_data["report_summary"] = generate_suggestion(
                        most_severe,
                        highest_confidence,
                        chat_history
                    )

                # Create and save annotated image
                annotated_img = draw_annotations(img, results)
                if not cv2.imwrite(annotated_path, annotated_img):
                    raise ValueError("Failed to save annotated image")

                response_data["annotated_image_url"] = f"/get_annotated/{filename}"
                response_data["detections"] = detections

            except Exception as e:
                # Clean up files if error occurred
                if 'upload_path' in locals() and os.path.exists(upload_path):
                    os.remove(upload_path)
                if 'annotated_path' in locals() and os.path.exists(annotated_path):
                    os.remove(annotated_path)

                return jsonify({
                    "error": f"Image processing failed: {str(e)}",
                    "details": "Please ensure you uploaded a valid image file"
                }), 400

        # Generate chatbot response
        prompt = message
        if response_data["report_summary"]:
            prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {message}"

        try:
            response_data["response"] = generate_chatbot_response(prompt, chat_history)
        except Exception as e:
            return jsonify({
                "error": f"Chat response generation failed: {str(e)}",
                "report_summary": response_data["report_summary"],
                "annotated_image_url": response_data["annotated_image_url"]
            }), 500

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500


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

# ==== Run the App ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)