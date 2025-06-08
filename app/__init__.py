import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app():
    # Load environment variables
    load_dotenv()
    
    app = Flask(__name__)
    # Configure CORS with explicit route patterns
    CORS(app, 
         supports_credentials=True, 
         resources={
             r"/get_annotated/*": {"origins": ["https://orthopedic-agent-rhjh9rdg8-gatt101s-projects.vercel.app"]},
             r"/chat": {"origins": ["https://orthopedic-agent-rhjh9rdg8-gatt101s-projects.vercel.app"]},
             r"/download_pdf": {"origins": ["https://orthopedic-agent-rhjh9rdg8-gatt101s-projects.vercel.app"]}
         },
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "OPTIONS"])
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_dev_key")
    
    # Configure upload folders with absolute paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, "uploads")
    app.config['ANNOTATED_FOLDER'] = os.path.join(base_dir, "annotated_images")
    
    # Create directories if they don't exist
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)
        logger.info(f"Created UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
        logger.info(f"Created ANNOTATED_FOLDER: {app.config['ANNOTATED_FOLDER']}")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise
    
    # Register blueprints
    from app.routes import main, chat, hospital
    app.register_blueprint(main.bp)
    app.register_blueprint(chat.bp)
    app.register_blueprint(hospital.bp)
    
    return app