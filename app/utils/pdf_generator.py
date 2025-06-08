from markdown import markdown as md_to_html
from xhtml2pdf import pisa
import base64
from io import BytesIO
import os

def generate_pdf_from_markdown(md_content, image_path=None):
    try:
        # Convert Markdown to HTML (with safety)
        body_html = md_to_html(md_content)

        # PDF-safe minimal CSS
        css = """
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.6;
            margin: 40px;
        }
        h1, h2 {
            color: #2c3e50;
            margin-top: 24px;
            margin-bottom: 12px;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        img {
            max-width: 90%;
            height: auto;
            margin-top: 20px;
        }
        """

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>{css}</style>
        </head>
        <body>
            <h1>Orthopedic Fracture Analysis Report</h1>
            {body_html}
        """

        # Embed image (if provided)
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                    html_template += f"""
                    <hr>
                    <h2>Annotated X-ray Image</h2>
                    <img src="data:image/png;base64,{encoded_img}" alt="Annotated X-ray" />
                    <p>AI-generated fracture region visualization</p>
                    """
            except Exception as e:
                html_template += f"<p><b>⚠ Image loading failed:</b> {str(e)}</p>"

        html_template += "</body></html>"

        # Generate PDF
        pdf_buffer = BytesIO()
        pisa_status = pisa.CreatePDF(html_template, dest=pdf_buffer)
        if pisa_status.err or pdf_buffer.tell() == 0:
            return None, "Failed to generate PDF. Check content or image formatting."

        pdf_buffer.seek(0)
        return pdf_buffer, None

    except Exception as e:
        return None, f"PDF creation exception: {str(e)}" 