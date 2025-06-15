import os
import logging
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_session import Session
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app():
    # Load environment variables
    load_dotenv()
    
    app = Flask(__name__)
    
    # Validate required environment variables
    required_env_vars = ["FLASK_SECRET_KEY", "MODEL_PATH", "SEVERITY_THRESHOLD"]
    for var in required_env_vars:
        if not os.getenv(var):
            logger.error(f"Missing required environment variable: {var}")
            raise ValueError(f"Environment variable {var} is not set")

    # Configure Flask app
    app.secret_key = os.getenv("FLASK_SECRET_KEY")
    app.config['SESSION_TYPE'] = os.getenv("SESSION_TYPE", "filesystem")
    app.config['SESSION_FILE_DIR'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "sessions")
    app.config['SESSION_PERMANENT'] = False
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # Default: 16MB

    # Initialize session
    try:
        os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
        logger.info(f"Created session directory: {app.config['SESSION_FILE_DIR']}")
        Session(app)
    except Exception as e:
        logger.error(f"Failed to initialize session directory: {str(e)}")
        raise

    # Configure CORS with explicit route patterns
    allowed_origins = [
        "https://orthopedic-agent.vercel.app",
        "https://orthopedic-agent-rhjh9rdg8-gatt101s-projects.vercel.app",
        "http://localhost:5173"
    ]
    
    CORS(app, 
         supports_credentials=True, 
         resources={
             r"/chat": {"origins": allowed_origins},
             r"/chatimg": {"origins": allowed_origins},
             r"/download_pdf": {"origins": allowed_origins},
             r"/annotated_images/*": {"origins": allowed_origins}
         },
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "OPTIONS"])
    
    # Configure upload folder with absolute path
    base_dir = os.path.abspath(os.path.dirname(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, "uploads")
    app.config['ANNOTATED_FOLDER'] = os.path.join(base_dir, "annotated_images")

    # Create upload and annotated directories if they don't exist
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logger.info(f"Created UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
        
        os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)
        logger.info(f"Created ANNOTATED_FOLDER: {app.config['ANNOTATED_FOLDER']}")
    except Exception as e:
        logger.error(f"Failed to create required directories: {str(e)}")
        raise

    # Route to serve annotated images
    @app.route('/annotated_images/<filename>')
    def serve_annotated_image(filename):
        return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

    # Register blueprints
    try:
        from app.routes import main, chat, hospital
        app.register_blueprint(main.bp)
        app.register_blueprint(chat.bp)
        app.register_blueprint(hospital.bp)
    except ImportError as e:
        logger.error(f"Failed to register blueprints: {str(e)}")
        raise

    return app
