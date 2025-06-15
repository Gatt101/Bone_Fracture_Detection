import logging
import os
import json
import cv2
import base64
import numpy as np
import tempfile
import shutil
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from app.utils.image_processing import draw_annotations
from app.utils.llm_utils import generate_suggestion, generate_chatbot_response
import pdfkit
from markdown2 import markdown

bp = Blueprint('chat', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Load YOLO model globally
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "mnt", "data", "best.pt"))
logger.info(f"Loading YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

def process_image_with_yolo(file):
    """Helper function to process image with YOLO model and return base64-encoded annotated image"""
    if model is None:
        raise ValueError("YOLO model is not available")
    
    if not file or not file.filename:
        raise ValueError("No file selected")
    
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Invalid file format. Please upload a PNG or JPEG image")
    
    name, _ = os.path.splitext(filename)
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    logger.info(f"Saving uploaded image to: {upload_path}")
    file.save(upload_path)
    
    img = cv2.imread(upload_path)
    if img is None:
        os.remove(upload_path)
        raise ValueError("Failed to read image file. Ensure the file is a valid image")
    
    logger.info("Processing image with YOLO model")
    results = model(img)
    detections = []
    highest_confidence = 0
    most_severe = "No Fracture"
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = round(box.conf[0].item(), 2)
            severity = "Severe" if confidence > float(os.getenv("SEVERITY_THRESHOLD", "0.5")) else "Mild"
            if confidence > highest_confidence:
                highest_confidence = confidence
                most_severe = severity
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "severity": severity
            })
    
    logger.info(f"YOLO detections: {len(detections)} found")
    
    logger.info("Annotating image")
    annotated_img = draw_annotations(img, results)
    
    # Encode annotated image as base64
    _, buffer = cv2.imencode('.png', annotated_img)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Clean up the uploaded image
    if os.path.exists(upload_path):
        logger.info(f"Cleaning up uploaded image: {upload_path}")
        os.remove(upload_path)
    
    return {
        "detections": detections,
        "highest_confidence": highest_confidence,
        "most_severe": most_severe,
        "annotated_image_base64": f"data:image/png;base64,{annotated_image_base64}",
        "upload_path": upload_path
    }

@bp.route("/chatimg", methods=["POST"])
def chat_with_image():
    """Endpoint for chat with mandatory image upload"""
    try:
        logger.info("Received chatimg request")
        
        if model is None:
            logger.error("YOLO model is not loaded.")
            return jsonify({"error": "YOLO model is not available. Please contact the administrator."}), 500

        if not request.content_type.startswith('multipart/form-data'):
            logger.error("Invalid Content-Type: Expected multipart/form-data")
            return jsonify({"error": "Content-Type must be multipart/form-data"}), 400

        try:
            message = request.form.get("message", "").strip()
            file = request.files.get("image", None)
            chat_history = request.form.get("chat_history", "[]")
            chat_history = json.loads(chat_history)
            logger.info(f"Received chatimg request: message='{message}', file={file.filename if file else None}")
        except json.JSONDecodeError:
            logger.error("Invalid chat_history format")
            return jsonify({"error": "Invalid chat_history format"}), 400
        except Exception as e:
            logger.error(f"Form data processing error: {str(e)}")
            return jsonify({"error": f"Form data processing error: {str(e)}"}), 400

        if not file:
            logger.error("No image provided to chatimg endpoint")
            return jsonify({"error": "Image is required for this endpoint"}), 400

        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_base64": "",
            "detections": []
        }

        try:
            image_data = process_image_with_yolo(file)
            response_data["detections"] = image_data["detections"]
            response_data["annotated_image_base64"] = image_data["annotated_image_base64"]
            logger.info(f"Sending annotated_image_base64 (first 50 chars): {response_data['annotated_image_base64'][:50]}")
            
            if image_data["detections"]:
                response_data["report_summary"] = generate_suggestion(
                    image_data["most_severe"],
                    image_data["highest_confidence"],
                    chat_history
                )
                logger.info(f"Generated report summary: {response_data['report_summary']}")

            prompt = message if message else "Analyze the provided X-ray image."
            if response_data["report_summary"]:
                prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {prompt}"

            logger.info("Generating chatbot response for chatimg")
            response_data["response"] = generate_chatbot_response(prompt, chat_history)
            logger.info(f"Chatbot response: {response_data['response']}")

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            if image_data.get("upload_path") and os.path.exists(image_data["upload_path"]):
                logger.info(f"Cleaning up file: {image_data['upload_path']}")
                os.remove(image_data["upload_path"])
            return jsonify({
                "error": f"Image processing failed: {str(e)}",
                "details": "Please ensure you uploaded a valid image file"
            }), 400

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in chatimg route: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

@bp.route("/chat", methods=["POST"])
def chat():
    """Endpoint for text-only chat or chat with optional image"""
    try:
        content_type = request.content_type or ""
        
        if content_type.startswith('multipart/form-data'):
            logger.info("Processing multipart form data (with optional image)")
            return _handle_multipart_chat()
        elif content_type.startswith('application/json'):
            logger.info("Processing JSON data (text only)")
            return _handle_text_only_chat()
        else:
            logger.error(f"Unsupported Content-Type: {content_type}")
            return jsonify({"error": "Content-Type must be multipart/form-data or application/json"}), 400

    except Exception as e:
        logger.error(f"Unexpected error in chat route: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

def _handle_multipart_chat():
    """Handle multipart form data with optional image upload"""
    try:
        try:
            message = request.form.get("message", "").strip()
            file = request.files.get("image", None)
            chat_history = request.form.get("chat_history", "[]")
            chat_history = json.loads(chat_history)
            logger.info(f"Received multipart request: message='{message}', file={file.filename if file else None}")
        except json.JSONDecodeError:
            logger.error("Invalid chat_history format")
            return jsonify({"error": "Invalid chat_history format"}), 400
        except Exception as e:
            logger.error(f"Form data processing error: {str(e)}")
            return jsonify({"error": f"Form data processing error: {str(e)}"}), 400

        if not message and not file:
            logger.error("No message or image provided")
            return jsonify({"error": "Either message or image must be provided"}), 400

        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_base64": "",
            "detections": []
        }

        if file:
            if model is None:
                logger.error("YOLO model is not loaded.")
                return jsonify({"error": "YOLO model is not available. Please contact the administrator."}), 500

            try:
                image_data = process_image_with_yolo(file)
                response_data["detections"] = image_data["detections"]
                response_data["annotated_image_base64"] = image_data["annotated_image_base64"]
                logger.info(f"Sending annotated_image_base64 (first 50 chars): {response_data['annotated_image_base64'][:50]}")
                
                if image_data["detections"]:
                    response_data["report_summary"] = generate_suggestion(
                        image_data["most_severe"],
                        image_data["highest_confidence"],
                        chat_history
                    )
                    logger.info(f"Generated report summary: {response_data['report_summary']}")

            except Exception as e:
                logger.error(f"Image processing failed: {str(e)}")
                if image_data.get("upload_path") and os.path.exists(image_data["upload_path"]):
                    logger.info(f"Cleaning up file: {image_data['upload_path']}")
                    os.remove(image_data["upload_path"])
                return jsonify({
                    "error": f"Image processing failed: {str(e)}",
                    "details": "Please ensure you uploaded a valid image file"
                }), 400

        prompt = message if message else "Analyze the provided X-ray image." if file else ""
        if response_data["report_summary"]:
            prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {prompt}"

        try:
            logger.info("Generating chatbot response")
            response_data["response"] = generate_chatbot_response(prompt, chat_history)
            logger.info(f"Chatbot response: {response_data['response']}")
        except Exception as e:
            logger.error(f"Chat response generation failed: {str(e)}")
            return jsonify({
                "error": f"Chat response generation failed: {str(e)}",
                "report_summary": response_data["report_summary"],
                "annotated_image_base64": response_data["annotated_image_base64"],
                "detections": response_data["detections"]
            }), 500

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in _handle_multipart_chat: {str(e)}")
        return jsonify({"error": "Failed to process multipart request", "details": str(e)}), 500

def _handle_text_only_chat():
    """Handle text-only chat requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = data.get("message", "").strip()
        chat_history = data.get("chat_history", [])
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        logger.info(f"Processing text-only chat: message='{message}'")
        
        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_base64": "",
            "detections": []
        }
        
        try:
            response_data["response"] = generate_chatbot_response(message, chat_history)
            logger.info(f"Chatbot response: {response_data['response']}")
        except Exception as e:
            logger.error(f"Chat response generation failed: {str(e)}")
            return jsonify({"error": f"Chat response generation failed: {str(e)}"}), 500
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in _handle_text_only_chat: {str(e)}")
        return jsonify({"error": "Failed to process text-only request", "details": str(e)}), 500

@bp.route("/download_pdf", methods=["POST"])
def download_pdf():
    """Generate a PDF from markdown report and optional base64-encoded image using pdfkit"""
    try:
        logger.info("Received download_pdf request")

        if not request.content_type.startswith('multipart/form-data'):
            logger.error("Invalid Content-Type: Expected multipart/form-data")
            return jsonify({"error": "Content-Type must be multipart/form-data"}), 400

        report_md = request.form.get("report_md", "").strip()
        image_base64 = request.form.get("image_base64", None)

        if not report_md:
            logger.error("No report markdown provided")
            return jsonify({"error": "Report markdown is required"}), 400

        # Create a temporary directory for PDF processing
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")

        # If an image is provided, decode and save it temporarily
        image_path = None
        if image_base64:
            try:
                logger.info(f"Received image_base64 (first 50 chars): {image_base64[:50]}")
                image_data = base64.b64decode(image_base64)
                image_path = os.path.join(temp_dir, "annotated_image.png")
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Saved temporary image to: {image_path}")
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {str(e)}")
                shutil.rmtree(temp_dir)
                return jsonify({"error": f"Failed to process base64 image: {str(e)}"}), 400
        else:
            logger.info("No image_base64 received in download_pdf request")

        # Convert markdown to HTML
        logger.info("Converting markdown to HTML")
        html_content = markdown(report_md, extras=["fenced-code-blocks", "tables", "header-ids"])

        # Create HTML document with a basic style
        html_document = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Fracture Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }}
                h1 {{
                    text-align: center;
                    color: #2c3e50;
                }}
                h2, h3, h4 {{
                    color: #34495e;
                }}
                img {{
                    max-width: 80%;
                    display: block;
                    margin: 20px auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                .generated-date {{
                    font-size: 0.9em;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fracture Analysis Report</h1>
                <p class="generated-date">Generated on: June 15, 2025, 01:59 PM IST</p>
            </div>
            {html_content}
        """

        # Add the image if it exists
        if image_path:
            logger.info("Embedding image path in HTML")
            html_document += f"""
            <h2>Annotated X-ray Image</h2>
            <img src="file://{image_path}" alt="Annotated X-ray">
            """

        html_document += """
        </body>
        </html>
        """

        # Write HTML to a temporary file
        html_path = os.path.join(temp_dir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_document)
        logger.info(f"Wrote HTML document to: {html_path}")

        # Convert HTML to PDF using pdfkit
        pdf_path = os.path.join(temp_dir, "report.pdf")
        try:
            pdfkit.from_file(
                html_path,
                pdf_path,
                options={
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                    'enable-local-file-access': None  # Enable access to local files
                }
            )
            logger.info("PDF generation successful")
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            shutil.rmtree(temp_dir)
            return jsonify({"error": "Failed to generate PDF", "details": str(e)}), 500

        # Check if PDF was generated
        if not os.path.exists(pdf_path):
            logger.error("PDF file was not generated")
            shutil.rmtree(temp_dir)
            return jsonify({"error": "PDF generation failed: Output file not found"}), 500

        # Send the PDF file as a response
        response = send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='fracture_report.pdf'
        )

        # Clean up temporary directory after sending the file
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

        return response

    except Exception as e:
        logger.error(f"Unexpected error in download_pdf route: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

@bp.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear chat history (assuming client manages history)"""
    logger.info("Clearing chat history")
    return jsonify({"message": "Chat history cleared (client-side management expected)."})