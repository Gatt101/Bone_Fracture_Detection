import logging
import os
import json
import cv2
from flask import Blueprint, request, jsonify, current_app, send_file, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from app.utils.image_processing import draw_annotations
from app.utils.llm_utils import generate_suggestion, generate_chatbot_response
from app.utils.cloudinary_utils import upload_image_to_cloudinary, upload_cv2_image_to_cloudinary
from markdown2 import markdown

bp = Blueprint('chat', __name__)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "mnt", "data", "best.pt"))
logger.info(f"Loading YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

def process_image_with_yolo(file):
    if model is None:
        raise ValueError("YOLO model is not available")

    if not file or not file.filename:
        raise ValueError("No file selected")

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Invalid file format. Please upload a PNG or JPEG image")

    # Save uploaded file temporarily
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
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
    annotated_img = draw_annotations(img, results)

    # Upload original image to Cloudinary
    original_cloudinary_result = None
    try:
        original_cloudinary_result = upload_image_to_cloudinary(
            upload_path, 
            folder="orthopedic-images/original"
        )
        logger.info(f"Uploaded original image to Cloudinary: {original_cloudinary_result['url']}")
    except Exception as e:
        logger.error(f"Failed to upload original image to Cloudinary: {str(e)}")

    # Upload annotated image to Cloudinary
    annotated_cloudinary_result = None
    try:
        annotated_filename = f"annotated_{filename}"
        annotated_cloudinary_result = upload_cv2_image_to_cloudinary(
            annotated_img,
            annotated_filename,
            folder="orthopedic-images/annotated"
        )
        logger.info(f"Uploaded annotated image to Cloudinary: {annotated_cloudinary_result['url']}")
    except Exception as e:
        logger.error(f"Failed to upload annotated image to Cloudinary: {str(e)}")

    # Also save annotated image locally as fallback
    annotated_folder = current_app.config['ANNOTATED_FOLDER']
    os.makedirs(annotated_folder, exist_ok=True)
    annotated_filename = f"annotated_{filename}"
    annotated_path = os.path.join(annotated_folder, annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)
    logger.info(f"Saved annotated image locally: {annotated_path}")

    # Clean up uploaded file
    os.remove(upload_path)

    return {
        "detections": detections,
        "highest_confidence": highest_confidence,
        "most_severe": most_severe,
        "annotated_image_filename": annotated_filename,
        "original_image_url": original_cloudinary_result['url'] if original_cloudinary_result else None,
        "annotated_image_url": annotated_cloudinary_result['url'] if annotated_cloudinary_result else None,
        "original_public_id": original_cloudinary_result['public_id'] if original_cloudinary_result else None,
        "annotated_public_id": annotated_cloudinary_result['public_id'] if annotated_cloudinary_result else None
    }

@bp.route("/chatimg", methods=["POST"])
def chat_with_image():
    try:
        logger.info("Received chatimg request")

        if model is None:
            return jsonify({"error": "YOLO model is not available. Please contact the administrator."}), 500

        if not request.content_type.startswith('multipart/form-data'):
            return jsonify({"error": "Content-Type must be multipart/form-data"}), 400

        message = request.form.get("message", "").strip()
        file = request.files.get("image", None)
        chat_history = json.loads(request.form.get("chat_history", "[]"))

        if not file:
            return jsonify({"error": "Image is required for this endpoint"}), 400

        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_url": "",
            "detections": []
        }

        image_data = process_image_with_yolo(file)
        response_data["detections"] = image_data["detections"]

        if image_data.get("annotated_image_filename"):
            # Use Cloudinary URL if available, otherwise fallback to local URL
            if image_data.get("annotated_image_url"):
                response_data["annotated_image_url"] = image_data["annotated_image_url"]
            else:
                response_data["annotated_image_url"] = url_for('serve_annotated_image',
                                                                filename=image_data["annotated_image_filename"],
                                                                _external=True)

        if image_data["detections"]:
            response_data["report_summary"] = generate_suggestion(
                image_data["most_severe"],
                image_data["highest_confidence"],
                chat_history
            )

        prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {message}" if response_data["report_summary"] else message
        response_data["response"] = generate_chatbot_response(prompt, chat_history)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"chatimg error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route("/chat", methods=["POST"])
def chat():
    content_type = request.content_type or ""

    if content_type.startswith('multipart/form-data'):
        return _handle_multipart_chat()
    elif content_type.startswith('application/json'):
        return _handle_text_only_chat()
    else:
        return jsonify({"error": "Unsupported Content-Type"}), 400

def _handle_multipart_chat():
    try:
        message = request.form.get("message", "").strip()
        file = request.files.get("image", None)
        chat_history = json.loads(request.form.get("chat_history", "[]"))

        if not message and not file:
            return jsonify({"error": "Either message or image must be provided"}), 400

        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_url": "",
            "detections": []
        }

        if file:
            image_data = process_image_with_yolo(file)
            response_data["detections"] = image_data["detections"]

            if image_data.get("annotated_image_filename"):
                # Use Cloudinary URL if available, otherwise fallback to local URL
                if image_data.get("annotated_image_url"):
                    response_data["annotated_image_url"] = image_data["annotated_image_url"]
                else:
                    response_data["annotated_image_url"] = url_for('serve_annotated_image',
                                                                    filename=image_data["annotated_image_filename"],
                                                                    _external=True)

            if image_data["detections"]:
                response_data["report_summary"] = generate_suggestion(
                    image_data["most_severe"],
                    image_data["highest_confidence"],
                    chat_history
                )

        prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {message}" if response_data["report_summary"] else message
        response_data["response"] = generate_chatbot_response(prompt, chat_history)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"multipart_chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def _handle_text_only_chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get("message", "").strip()
        chat_history = data.get("chat_history", [])

        if not message:
            return jsonify({"error": "Message is required"}), 400

        response_data = {
            "response": generate_chatbot_response(message, chat_history),
            "report_summary": "",
            "annotated_image_url": "",
            "detections": []
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"text_only_chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route("/clear_history", methods=["POST"])
def clear_history():
    logger.info("Clearing chat history")
    return jsonify({"message": "Chat history cleared (client-side management expected)."})
