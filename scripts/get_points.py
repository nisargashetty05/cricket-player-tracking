import cv2
import numpy as np

# Global variables to store scale
scale_factor = 1.0

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Scale the click BACK to original size
        real_x = int(x / scale_factor)
        real_y = int(y / scale_factor)
        
        print(f"✅ Clicked at: [{real_x}, {real_y}]")
        
        # Draw on the small image so you can see where you clicked
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_display, str(real_x) + ',' + str(real_y), (x,y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Click 4 Corners', img_display)

# Load video
cap = cv2.VideoCapture('data/cricket_match.mp4')
ret, frame = cap.read() # Read the first frame

if ret:
    # --- RESIZE LOGIC ---
    original_height, original_width = frame.shape[:2]
    
    # Force the display width to 960px (fits on almost all laptops)
    target_width = 960
    scale_factor = target_width / original_width
    
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    img_display = cv2.resize(frame, (new_width, new_height))
    # --------------------

    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("1. A smaller window will open.")
    print("2. Click the 4 corners of the pitch in CLOCKWISE order:")
    print("   Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("3. Watch this terminal for the coordinates.")
    print("4. Press any key to close when done.")
    print("="*50 + "\n")

    cv2.imshow('Click 4 Corners', img_display)
    cv2.setMouseCallback('Click 4 Corners', click_event)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not read video file.")