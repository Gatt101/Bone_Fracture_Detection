import logging
import os
import json
import cv2
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from app.utils.image_processing import draw_annotations
from app.utils.llm_utils import generate_suggestion, generate_chatbot_response

bp = Blueprint('chat', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Load YOLO model globally and robustly
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "mnt", "data", "best.pt"))
logger.info(f"Loading YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None  # Explicitly set to None

def process_image_with_yolo(file):
    """Helper function to process image with YOLO model"""
    if model is None:
        raise ValueError("YOLO model is not available")
    
    if not file or not file.filename:
        raise ValueError("No file selected")
    
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Invalid file format. Please upload a PNG or JPEG image")
    
    name, _ = os.path.splitext(filename)
    annotated_filename = f"{name}_annotated.png"
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    annotated_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], annotated_filename)
    
    logger.info(f"Saving uploaded image to: {upload_path}")
    file.save(upload_path)
    
    img = cv2.imread(upload_path)
    if img is None:
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
    logger.info(f"Saving annotated image to: {annotated_path}")
    if not cv2.imwrite(annotated_path, annotated_img):
        raise ValueError("Failed to save annotated image")
    
    if not os.path.exists(annotated_path):
        raise ValueError("Annotated image not found after saving")
    
    return {
        "detections": detections,
        "highest_confidence": highest_confidence,
        "most_severe": most_severe,
        "annotated_filename": annotated_filename,
        "upload_path": upload_path,
        "annotated_path": annotated_path
    }

@bp.route("/chat+img", methods=["POST"])
def chat_with_image():
    """Dedicated endpoint for chat with image upload"""
    try:
        logger.info("Received chat+img request")
        
        if model is None:
            logger.error("YOLO model is not loaded. Cannot process image.")
            return jsonify({"error": "YOLO model is not available. Please contact the administrator."}), 500

        if not request.content_type.startswith('multipart/form-data'):
            logger.error("Invalid Content-Type: Expected multipart/form-data")
            return jsonify({"error": "Content-Type must be multipart/form-data"}), 400

        try:
            message = request.form.get("message", "").strip()
            file = request.files.get("image", None)
            chat_history = request.form.get("chat_history", "[]")
            chat_history = json.loads(chat_history)
            logger.info(f"Received chat+img request: message='{message}', file={file.filename if file else None}")
        except json.JSONDecodeError:
            logger.error("Invalid chat_history format")
            return jsonify({"error": "Invalid chat_history format"}), 400
        except Exception as e:
            logger.error(f"Form data processing error: {str(e)}")
            return jsonify({"error": f"Form data processing error: {str(e)}"}), 400

        if not file:
            logger.error("No image provided to chat+img endpoint")
            return jsonify({"error": "Image is required for this endpoint"}), 400

        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_url": "",
            "detections": []
        }

        upload_path = None
        annotated_path = None
        
        if file:
            if not file.filename:
                logger.error("No file selected")
                return jsonify({"error": "No file selected"}), 400

            try:
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logger.error(f"Invalid file format for {filename}. Expected PNG or JPEG")
                    return jsonify({"error": "Invalid file format. Please upload a PNG or JPEG image"}), 400

                name, _ = os.path.splitext(filename)
                annotated_filename = f"{name}_annotated.png"
                upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                annotated_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], annotated_filename)

                logger.info(f"Saving uploaded image to: {upload_path}")
                file.save(upload_path)

                img = cv2.imread(upload_path)
                if img is None:
                    logger.error(f"Failed to read image at {upload_path}")
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
                if detections:
                    response_data["report_summary"] = generate_suggestion(
                        most_severe,
                        highest_confidence,
                        chat_history
                    )
                    logger.info(f"Generated report summary: {response_data['report_summary']}")

                logger.info("Annotating image")
                annotated_img = draw_annotations(img, results)
                logger.info(f"Saving annotated image to: {annotated_path}")
                if not cv2.imwrite(annotated_path, annotated_img):
                    logger.error(f"Failed to save annotated image to {annotated_path}")
                    raise ValueError("Failed to save annotated image")

                if not os.path.exists(annotated_path):
                    logger.error(f"Annotated image not found at {annotated_path}")
                    raise ValueError("Annotated image not found after saving")

                response_data["annotated_image_url"] = f"/get_annotated/{annotated_filename}"
                response_data["detections"] = detections
                logger.info(f"Returning annotated_image_url: {response_data['annotated_image_url']}")

            except Exception as e:
                logger.error(f"Image processing failed: {str(e)}")
                if upload_path and os.path.exists(upload_path):
                    logger.info(f"Cleaning up uploaded file: {upload_path}")
                    os.remove(upload_path)
                if annotated_path and os.path.exists(annotated_path):
                    logger.info(f"Cleaning up annotated file: {annotated_path}")
                    os.remove(annotated_path)
                return jsonify({
                    "error": f"Image processing failed: {str(e)}",
                    "details": "Please ensure you uploaded a valid image file"
                }), 400

        prompt = message
        if response_data["report_summary"]:
            prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {message}"

        try:
            logger.info("Generating chatbot response for chat+img")
            response_data["response"] = generate_chatbot_response(prompt, chat_history)
            logger.info(f"Chatbot response: {response_data['response']}")
        except Exception as e:
            logger.error(f"Chat response generation failed: {str(e)}")
            return jsonify({
                "error": f"Chat response generation failed: {str(e)}",
                "report_summary": response_data["report_summary"],
                "annotated_image_url": response_data["annotated_image_url"]
            }), 500

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in chat+img route: {str(e)}")
        if upload_path and os.path.exists(upload_path):
            logger.info(f"Cleaning up uploaded file: {upload_path}")
            os.remove(upload_path)
        if annotated_path and os.path.exists(annotated_path):
            logger.info(f"Cleaning up annotated file: {annotated_path}")
            os.remove(annotated_path)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

@bp.route("/chat", methods=["POST"])
def chat():
    try:
        if model is None:
            logger.error("YOLO model is not loaded. Cannot process image.")
            return jsonify({"error": "YOLO model is not available. Please contact the administrator."}), 500

        # Check if this is a multipart request (with image) or JSON request (text only)
        content_type = request.content_type or ""
        
        if content_type.startswith('multipart/form-data'):
            logger.info("Processing multipart form data (with image)")
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
    """Handle multipart form data with potential image upload"""
    try:
        message = request.form.get("message", "").strip()
        file = request.files.get("image", None)
        chat_history = request.form.get("chat_history", "[]")
        chat_history = json.loads(chat_history)
        logger.info(f"Received multipart request: message='{message}', file={file.filename if file else None}")
        
        if not message and not file:
            logger.error("No message or image provided")
            return jsonify({"error": "Either message or image must be provided"}), 400

        response_data = {
            "response": "",
            "report_summary": "",
            "annotated_image_url": "",
            "detections": []
        }

        upload_path = None
        annotated_path = None
        
        if file:
            if not file.filename:
                logger.error("No file selected")
                return jsonify({"error": "No file selected"}), 400

            try:
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logger.error(f"Invalid file format for {filename}. Expected PNG or JPEG")
                    return jsonify({"error": "Invalid file format. Please upload a PNG or JPEG image"}), 400

                name, _ = os.path.splitext(filename)
                annotated_filename = f"{name}_annotated.png"
                upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                annotated_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], annotated_filename)

                logger.info(f"Saving uploaded image to: {upload_path}")
                file.save(upload_path)

                img = cv2.imread(upload_path)
                if img is None:
                    logger.error(f"Failed to read image at {upload_path}")
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
                if detections:
                    response_data["report_summary"] = generate_suggestion(
                        most_severe,
                        highest_confidence,
                        chat_history
                    )
                    logger.info(f"Generated report summary: {response_data['report_summary']}")

                logger.info("Annotating image")
                annotated_img = draw_annotations(img, results)
                logger.info(f"Saving annotated image to: {annotated_path}")
                if not cv2.imwrite(annotated_path, annotated_img):
                    logger.error(f"Failed to save annotated image to {annotated_path}")
                    raise ValueError("Failed to save annotated image")

                if not os.path.exists(annotated_path):
                    logger.error(f"Annotated image not found at {annotated_path}")
                    raise ValueError("Annotated image not found after saving")

                response_data["annotated_image_url"] = f"/get_annotated/{annotated_filename}"
                response_data["detections"] = detections
                logger.info(f"Returning annotated_image_url: {response_data['annotated_image_url']}")

            except Exception as e:
                logger.error(f"Image processing failed: {str(e)}")
                if upload_path and os.path.exists(upload_path):
                    logger.info(f"Cleaning up uploaded file: {upload_path}")
                    os.remove(upload_path)
                if annotated_path and os.path.exists(annotated_path):
                    logger.info(f"Cleaning up annotated file: {annotated_path}")
                    os.remove(annotated_path)
                return jsonify({
                    "error": f"Image processing failed: {str(e)}",
                    "details": "Please ensure you uploaded a valid image file"
                }), 400

        prompt = message
        if response_data["report_summary"]:
            prompt = f"Medical Report:\n{response_data['report_summary']}\n\nUser Question: {message}"

        try:
            logger.info("Generating chatbot response")
            response_data["response"] = generate_chatbot_response(prompt, chat_history)
            logger.info(f"Chatbot response: {response_data['response']}")
        except Exception as e:
            logger.error(f"Chat response generation failed: {str(e)}")
            return jsonify({
                "error": f"Chat response generation failed: {str(e)}",
                "report_summary": response_data["report_summary"],
                "annotated_image_url": response_data["annotated_image_url"]
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
            "annotated_image_url": "",
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

@bp.route("/clear_history", methods=["POST"])
def clear_history():
    logger.info("Clearing chat history")
    session.pop("history", None)
    return jsonify({"message": "Chat history cleared."})