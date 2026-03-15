import cv2
import numpy as np
 
def inspect_surface(image_path_or_array, threshold=65,
                    min_defect_area=50):
    """
    Inspect a composite surface image for defects.
    
    Returns:
        result: dict with pass/fail, defect count, annotated image
    """
    # Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array.copy()
    
    if img is None:
        return {"status": "ERROR", "message": "Could not load image"}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold: defects are brighter than the dark composite surface
    _, binary = cv2.threshold(blurred, threshold, 255,
                               cv2.THRESH_BINARY)
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours (each contour = a potential defect)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    defects = []
    annotated = img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_defect_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Classify defect type by shape
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        if aspect_ratio > 4:
            defect_type = "Scratch"
            color = (0, 0, 255)     # Red
        elif circularity > 0.6:
            defect_type = "Dent/Porosity"
            color = (0, 165, 255)   # Orange
        else:
            defect_type = "Delamination"
            color = (0, 255, 255)   # Yellow
        
        defects.append({
            "type": defect_type,
            "area_px": int(area),
            "location": (x, y, w, h),
            "circularity": round(circularity, 3),
        })
        
        # Draw on annotated image
        cv2.drawContours(annotated, [cnt], -1, color, 2)
        cv2.rectangle(annotated, (x-2, y-2), (x+w+2, y+h+2),
                      color, 1)
        cv2.putText(annotated, defect_type, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # PASS / FAIL decision
    status = "FAIL" if len(defects) > 0 else "PASS"
    
    return {
        "status": status,
        "defect_count": len(defects),
        "defects": defects,
        "annotated_image": annotated,
        "binary_mask": binary,
        "total_defect_area_px": sum(d["area_px"] for d in defects),
    }
 
if __name__ == "__main__":
    import os
    print("Testing detector on sample images...")
    
    # Test on a good image
    good_result = inspect_surface("test_images/good/good_001.png")
    print(f"\nGood image: {good_result['status']} "
          f"({good_result['defect_count']} defects)")
    
    # Test on a defect image
    bad_result = inspect_surface("test_images/defect/defect_001.png")
    print(f"Defect image: {bad_result['status']} "
          f"({bad_result['defect_count']} defects)")
    for d in bad_result["defects"]:
        print(f"  - {d['type']}: area={d['area_px']}px, "
              f"circularity={d['circularity']}")
    
    # Save annotated result
    cv2.imwrite("test_annotated.png",
                bad_result["annotated_image"])
    print("\nSaved annotated image: test_annotated.png")
    
    # Run accuracy test on all images
    correct = 0
    total = 0
    for label, folder in [("good","test_images/good"),
                           ("defect","test_images/defect")]:
        for f in sorted(os.listdir(folder)):
            path = os.path.join(folder, f)
            result = inspect_surface(path)
            expected = "PASS" if label == "good" else "FAIL"
            if result["status"] == expected:
                correct += 1
            total += 1
    print(f"\nAccuracy: {correct}/{total} = "
          f"{correct/total*100:.0f}%")
