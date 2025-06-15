#!/usr/bin/env python3
"""
Test script to verify Cloudinary integration
"""
import os
import sys
import tempfile
import cv2
import numpy as np

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.cloudinary_utils import upload_cv2_image_to_cloudinary

def test_cloudinary_upload():
    """Test Cloudinary upload functionality"""
    
    print("Testing Cloudinary integration...")
    
    # Create a test image
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    test_image[:] = (255, 255, 255)  # White background
    
    # Add some test content
    cv2.putText(test_image, "Test Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.rectangle(test_image, (100, 100), (300, 200), (0, 255, 0), 2)
    
    try:
        # Upload test image to Cloudinary
        result = upload_cv2_image_to_cloudinary(
            test_image,
            "test_image.png",
            folder="orthopedic-images/test"
        )
        
        print("✅ Cloudinary upload successful!")
        print(f"📁 Public ID: {result['public_id']}")
        print(f"🔗 URL: {result['url']}")
        print(f"🆔 Asset ID: {result['asset_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cloudinary upload failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_cloudinary_upload()
    if success:
        print("\n🎉 Cloudinary integration test passed!")
        print("Your Cloudinary setup is working correctly.")
    else:
        print("\n💥 Cloudinary integration test failed!")
        print("Please check your Cloudinary credentials in .env file.")
        sys.exit(1) 