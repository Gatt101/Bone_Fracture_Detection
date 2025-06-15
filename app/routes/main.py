import logging
from flask import Blueprint, render_template, send_from_directory, current_app, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from io import BytesIO
import tempfile
import base64
import shutil
from markdown2 import markdown

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
            # Try with 'annotated_' prefix if not already present
            if not safe_filename.startswith('annotated_'):
                alt_filename = f"annotated_{safe_filename}"
                alt_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], alt_filename)
                logger.info(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    return send_from_directory(current_app.config['ANNOTATED_FOLDER'], alt_filename)
            
            return jsonify({"error": f"Image {safe_filename} not found"}), 404
            
        return send_from_directory(current_app.config['ANNOTATED_FOLDER'], safe_filename)
    except Exception as e:
        logger.error(f"Error serving annotated image {filename}: {str(e)}")
        return jsonify({"error": "Failed to serve image", "details": str(e)}), 500

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

        # Create HTML document with improved styling
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
                    color: #333;
                }}
                h1 {{
                    text-align: center;
                    color: #2c3e50;
                    font-size: 24pt;
                    margin-bottom: 10px;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2, h3, h4 {{
                    color: #34495e;
                    margin-top: 20px;
                    margin-bottom: 10px;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                }}
                h2 {{
                    font-size: 18pt;
                }}
                h3 {{
                    font-size: 14pt;
                }}
                ul {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                img {{
                    max-width: 80%;
                    display: block;
                    margin: 20px auto;
                    border: 2px solid #ddd;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .header h1 {{
                    color: white;
                    border-bottom: none;
                    margin-bottom: 5px;
                }}
                .generated-date {{
                    font-size: 0.9em;
                    color: #ecf0f1;
                    margin-top: 10px;
                }}
                .content-section {{
                    background: #f8f9fa;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }}
                .severity-high {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .severity-medium {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .severity-low {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .image-caption {{
                    text-align: center;
                    font-style: italic;
                    color: #7f8c8d;
                    margin-top: 10px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fracture Analysis Report</h1>
                <p class="generated-date">Generated on: {datetime.now().strftime('%B %d, %Y, %I:%M %p')}</p>
            </div>
            <div class="content-section">
                {html_content}
            </div>
        """

        # Add the image if it exists
        if image_path:
            logger.info("Embedding image path in HTML")
            html_document += f"""
            <div class="content-section">
                <h2>Annotated X-ray Image</h2>
                <img src="file://{image_path}" alt="Annotated X-ray">
                <p class="image-caption">AI-generated fracture region visualization with confidence scoring</p>
            </div>
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

        # Convert HTML to PDF using xhtml2pdf (since we removed pdfkit)
        from app.utils.pdf_generator import generate_pdf_from_markdown
        
        # Generate PDF using our existing function
        pdf_buffer, error = generate_pdf_from_markdown(report_md, image_path)
        
        if error:
            logger.error(f"PDF generation failed: {error}")
            shutil.rmtree(temp_dir)
            return jsonify({"error": "Failed to generate PDF", "details": error}), 500

        if not pdf_buffer:
            logger.error("PDF buffer is empty")
            shutil.rmtree(temp_dir)
            return jsonify({"error": "PDF generation failed: Empty buffer"}), 500

        # Send the PDF file as a response
        response = send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'fracture_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
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