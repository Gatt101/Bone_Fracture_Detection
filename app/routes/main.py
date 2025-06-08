import logging
from flask import Blueprint, render_template, send_from_directory, current_app, request, jsonify, send_file
from app.utils.pdf_generator import generate_pdf_from_markdown
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from io import BytesIO

bp = Blueprint('main', __name__)

# Configure logging
logger = logging.getLogger(__name__)

@bp.route('/')
def index():
    return render_template('frontend.html')

@bp.route("/get_annotated/<filename>")
def get_annotated_image(filename):
    try:
        safe_filename = secure_filename(filename)
        annotated_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], safe_filename)
        logger.info(f"Serving annotated image: {annotated_path}")
        if not os.path.exists(annotated_path):
            logger.error(f"Annotated image not found: {annotated_path}")
            return jsonify({"error": f"Image {safe_filename} not found"}), 404
        return send_from_directory(current_app.config['ANNOTATED_FOLDER'], safe_filename)
    except Exception as e:
        logger.error(f"Error serving annotated image {filename}: {str(e)}")
        return jsonify({"error": "Failed to serve image", "details": str(e)}), 500

@bp.route("/download_pdf", methods=["POST"])
def download_pdf():
    try:
        report_md = request.form.get("report_md", "")
        annotated_img = request.form.get("annotated_image", "")

        if not report_md:
            logger.error("No report content provided for PDF generation")
            return jsonify({"error": "No report content provided"}), 400

        image_path = None
        if annotated_img:
            safe_filename = secure_filename(annotated_img)
            image_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], safe_filename)
            logger.info(f"Checking for annotated image at: {image_path}")
            if not os.path.exists(image_path):
                logger.warning(f"Annotated image not found at: {image_path}")
                image_path = None
            else:
                logger.info(f"Found annotated image for PDF: {image_path}")

        logger.info("Generating PDF from markdown")
        pdf_buffer, error = generate_pdf_from_markdown(report_md, image_path)

        if error:
            logger.error(f"PDF generation failed: {error}")
            return jsonify({"error": "PDF generation failed", "details": error}), 500

        logger.info("PDF generated successfully")
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Fracture_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )

    except Exception as e:
        logger.error(f"Unexpected error in download_pdf: {str(e)}")
        return jsonify({"error": "Unexpected server error", "details": str(e)}), 500