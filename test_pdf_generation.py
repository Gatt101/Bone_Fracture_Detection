#!/usr/bin/env python3
"""
Test script to verify PDF generation functionality with improved styling
"""
import os
import sys
from io import BytesIO

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.pdf_generator import generate_pdf_from_markdown

def test_pdf_generation():
    """Test PDF generation with sample markdown content"""
    
    # Sample markdown content
    sample_markdown = """# Fracture Analysis Report

## Severity Assessment
- Classification: Severe
- Confidence Score: 85.50%
- Key Radiographic Features: 
  - Significant fracture pattern detected
  - Clinical correlation recommended

## Urgency Level
- Priority Level: Emergency
- Recommended Action Timeline: Immediate
- Triage Considerations: Requires urgent orthopedic evaluation

## Clinical Recommendations
1. Comprehensive imaging workup
2. Appropriate analgesia protocol
3. Immobilization measures
4. Immediate orthopedic consultation
5. Follow-up within 24-48 hours

## Treatment Complexity
- Complexity Tier: High
- Potential Interventions:
  - Surgical stabilization options
  - Non-operative management alternatives
  - Post-treatment rehabilitation program
"""

    print("Testing PDF generation with improved styling...")
    
    # Test without image
    pdf_buffer, error = generate_pdf_from_markdown(sample_markdown)
    
    if error:
        print(f"❌ PDF generation failed: {error}")
        return False
    
    if pdf_buffer and pdf_buffer.tell() > 0:
        print("✅ PDF generated successfully with improved styling")
        
        # Save test PDF
        with open("test_report_improved.pdf", "wb") as f:
            f.write(pdf_buffer.getvalue())
        print("📄 Test PDF saved as 'test_report_improved.pdf'")
        return True
    else:
        print("❌ PDF buffer is empty")
        return False

if __name__ == "__main__":
    success = test_pdf_generation()
    if success:
        print("\n🎉 PDF generation test passed!")
        print("The PDF should now have:")
        print("- Improved header with gradient background")
        print("- Better typography and spacing")
        print("- Content sections with left border")
        print("- Professional styling")
    else:
        print("\n💥 PDF generation test failed!")
        sys.exit(1) 