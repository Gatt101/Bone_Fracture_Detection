import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import logging
import cv2
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

def upload_image_to_cloudinary(image_path, folder="orthopedic-images", public_id=None):
    """
    Upload an image to Cloudinary
    
    Args:
        image_path (str): Path to the image file
        folder (str): Cloudinary folder name
        public_id (str): Optional public ID for the image
    
    Returns:
        dict: Cloudinary upload response with URL and public_id
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Generate public_id if not provided
        if not public_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            public_id = f"{folder}/{name}_{timestamp}"
        
        logger.info(f"Uploading image to Cloudinary: {image_path}")
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            image_path,
            public_id=public_id,
            folder=folder,
            resource_type="image",
            overwrite=True,
            invalidate=True
        )
        
        logger.info(f"Successfully uploaded to Cloudinary: {result['secure_url']}")
        return {
            "url": result['secure_url'],
            "public_id": result['public_id'],
            "asset_id": result['asset_id']
        }
        
    except Exception as e:
        logger.error(f"Failed to upload image to Cloudinary: {str(e)}")
        raise

def upload_cv2_image_to_cloudinary(cv2_image, filename, folder="orthopedic-images"):
    """
    Upload a CV2 image (numpy array) to Cloudinary
    
    Args:
        cv2_image: CV2 image (numpy array)
        filename (str): Name for the file
        folder (str): Cloudinary folder name
    
    Returns:
        dict: Cloudinary upload response with URL and public_id
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Save CV2 image to temporary file
        cv2.imwrite(temp_path, cv2_image)
        
        # Upload to Cloudinary
        result = upload_image_to_cloudinary(temp_path, folder, filename)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to upload CV2 image to Cloudinary: {str(e)}")
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

def delete_image_from_cloudinary(public_id):
    """
    Delete an image from Cloudinary
    
    Args:
        public_id (str): Public ID of the image to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Deleting image from Cloudinary: {public_id}")
        result = cloudinary.uploader.destroy(public_id)
        
        if result.get('result') == 'ok':
            logger.info(f"Successfully deleted image: {public_id}")
            return True
        else:
            logger.error(f"Failed to delete image: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting image from Cloudinary: {str(e)}")
        return False

def get_cloudinary_url(public_id, transformation=None):
    """
    Generate a Cloudinary URL with optional transformations
    
    Args:
        public_id (str): Public ID of the image
        transformation (dict): Optional transformation parameters
    
    Returns:
        str: Cloudinary URL
    """
    try:
        if transformation:
            url = cloudinary.CloudinaryImage(public_id).build_url(**transformation)
        else:
            url = cloudinary.CloudinaryImage(public_id).build_url()
        
        return url
        
    except Exception as e:
        logger.error(f"Error generating Cloudinary URL: {str(e)}")
        raise 