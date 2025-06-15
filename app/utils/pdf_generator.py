from markdown import markdown as md_to_html
from xhtml2pdf import pisa
import base64
from io import BytesIO
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_pdf_from_markdown(md_content, image_path=None):
    try:
        # Convert Markdown to HTML (with safety)
        body_html = md_to_html(md_content)

        # Improved PDF-safe CSS
        css = """
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.6;
            margin: 40px;
            color: #333;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 24pt;
            margin-bottom: 10px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2, h3, h4 {
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h2 {
            font-size: 18pt;
        }
        h3 {
            font-size: 14pt;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        img {
            max-width: 80%;
            height: auto;
            margin: 20px auto;
            display: block;
            border: 2px solid #ddd;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            background: #667eea;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .header h1 {
            color: white;
            border-bottom: none;
            margin-bottom: 5px;
        }
        .generated-date {
            font-size: 0.9em;
            color: #ecf0f1;
            margin-top: 10px;
        }
        .content-section {
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .image-caption {
            text-align: center;
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
            font-size: 10pt;
        }
        """

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Fracture Analysis Report</title>
            <style>{css}</style>
        </head>
        <body>
            <div class="header">
                <h1>Fracture Analysis Report</h1>
                <p class="generated-date">Generated on: {datetime.now().strftime('%B %d, %Y, %I:%M %p')}</p>
            </div>
            <div class="content-section">
                {body_html}
            </div>
        """

        # Embed image (if provided)
        if image_path:
            logger.info(f"Attempting to embed image: {image_path}")
            
            # Check if image_path is a filename or full path
            if not os.path.isabs(image_path):
                # If it's just a filename, try to find it in the annotated_images folder
                from flask import current_app
                if current_app:
                    image_path = os.path.join(current_app.config['ANNOTATED_FOLDER'], image_path)
                    logger.info(f"Resolved image path to: {image_path}")
            
            if os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as img_file:
                        img_data = img_file.read()
                        encoded_img = base64.b64encode(img_data).decode('utf-8')
                        
                        # Determine image format
                        if image_path.lower().endswith('.png'):
                            mime_type = 'image/png'
                        elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                            mime_type = 'image/jpeg'
                        else:
                            mime_type = 'image/png'  # Default
                        
                        html_template += f"""
                        <div class="content-section">
                            <h2>Annotated X-ray Image</h2>
                            <img src="data:{mime_type};base64,{encoded_img}" alt="Annotated X-ray" />
                            <div class="image-caption">AI-generated fracture region visualization with confidence scoring</div>
                        </div>
                        """
                        logger.info(f"Successfully embedded image: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to embed image {image_path}: {str(e)}")
                    html_template += f"<p><b>⚠ Image loading failed:</b> {str(e)}</p>"
            else:
                logger.warning(f"Image file not found: {image_path}")
                html_template += f"<p><b>⚠ Image not found:</b> {os.path.basename(image_path)}</p>"

        html_template += "</body></html>"

        # Generate PDF
        pdf_buffer = BytesIO()
        pisa_status = pisa.CreatePDF(html_template, dest=pdf_buffer)
        
        if pisa_status.err:
            logger.error(f"PDF generation error: {pisa_status.err}")
            return None, f"Failed to generate PDF: {pisa_status.err}"
        
        if pdf_buffer.tell() == 0:
            logger.error("PDF buffer is empty")
            return None, "Failed to generate PDF: Empty buffer"

        pdf_buffer.seek(0)
        logger.info("PDF generated successfully")
        return pdf_buffer, None

    except Exception as e:
        logger.error(f"PDF creation exception: {str(e)}")
        return None, f"PDF creation exception: {str(e)}" 