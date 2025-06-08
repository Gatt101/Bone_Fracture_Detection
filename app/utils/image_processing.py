import cv2
import os

def draw_annotations(image, results):
    """Draw annotations on the image with better visualization"""
    img = image.copy()
    severity_threshold = float(os.getenv("SEVERITY_THRESHOLD", "0.5"))

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = round(box.conf[0].item(), 2)
            severity = "Severe" if confidence > severity_threshold else "Mild"
            color = (0, 0, 255) if severity == "Severe" else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create label text
            label = f"{severity} ({confidence:.2f})"

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Draw label background
            cv2.rectangle(img,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)

            # Draw label text
            cv2.putText(img, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return img 