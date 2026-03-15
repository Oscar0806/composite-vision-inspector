import cv2
import numpy as np
import os
 
os.makedirs("test_images/good", exist_ok=True)
os.makedirs("test_images/defect", exist_ok=True)
 
def make_composite_surface(size=400):
    """Generate a realistic-looking composite surface texture."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 40
    # Add carbon fiber weave pattern
    for i in range(0, size, 4):
        color_var = np.random.randint(30, 55)
        cv2.line(img, (i, 0), (i, size), (color_var,color_var,color_var), 1)
        cv2.line(img, (0, i), (size, i), (color_var,color_var,color_var), 1)
    # Add slight noise for realism
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img
 
def add_scratch(img):
    """Add a scratch defect."""
    h, w = img.shape[:2]
    x1 = np.random.randint(50, w-100)
    y1 = np.random.randint(50, h-100)
    length = np.random.randint(80, 200)
    angle = np.random.randint(-30, 30)
    x2 = x1 + int(length * np.cos(np.radians(angle)))
    y2 = y1 + int(length * np.sin(np.radians(angle)))
    thickness = np.random.randint(1, 3)
    brightness = np.random.randint(80, 140)
    cv2.line(img, (x1,y1), (x2,y2), (brightness,brightness,brightness),
             thickness)
    return img
 
def add_dent(img):
    """Add a circular dent/porosity defect."""
    h, w = img.shape[:2]
    cx = np.random.randint(60, w-60)
    cy = np.random.randint(60, h-60)
    radius = np.random.randint(10, 35)
    brightness = np.random.randint(60, 100)
    cv2.circle(img, (cx,cy), radius, (brightness,brightness,brightness), -1)
    # Add subtle ring around dent
    cv2.circle(img, (cx,cy), radius+3,
               (brightness-10,brightness-10,brightness-10), 1)
    return img
 
def add_delamination(img):
    """Add a delamination/blister defect (irregular bright patch)."""
    h, w = img.shape[:2]
    cx = np.random.randint(80, w-80)
    cy = np.random.randint(80, h-80)
    axes = (np.random.randint(30, 70), np.random.randint(20, 40))
    angle = np.random.randint(0, 180)
    brightness = np.random.randint(70, 110)
    cv2.ellipse(img, (cx,cy), axes, angle, 0, 360,
                (brightness,brightness,brightness), -1)
    return img
 
# Generate 20 good images
print("Generating good images...")
for i in range(20):
    img = make_composite_surface()
    cv2.imwrite(f"test_images/good/good_{i+1:03d}.png", img)
print(f"  Saved 20 images to test_images/good/")
 
# Generate 20 defect images (mixed defect types)
print("Generating defect images...")
defect_funcs = [add_scratch, add_dent, add_delamination]
for i in range(20):
    img = make_composite_surface()
    # Add 1-3 random defects
    n_defects = np.random.randint(1, 4)
    for _ in range(n_defects):
        func = np.random.choice(defect_funcs)
        img = func(img)
    cv2.imwrite(f"test_images/defect/defect_{i+1:03d}.png", img)
print(f"  Saved 20 images to test_images/defect/")
print("Done! Total: 40 test images")
