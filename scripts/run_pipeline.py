import cv2
import sys
import os
import numpy as np

# Add parent directory to path so we can import from pipeline folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.track import PlayerTracker
from pipeline.transformer import ViewTransformer

def main():
    # --- FILE PATHS ---
    input_video_path = "data/cricket_match.mp4"
    output_video_path = "output/tracked_video_with_map.mp4"
    
    # Try different extensions just in case
    possible_map_names = ["data/Ground_image.png", "data/Ground_image.jpg", "data/Ground_image.jpeg"]
    map_image_path = None
    
    # Automatically find the image file
    for path in possible_map_names:
        if os.path.exists(path):
            map_image_path = path
            break

    # --- CONFIGURATION: PASTE YOUR POINTS HERE ---
    # REPLACE these numbers with the ones you got from get_points.py!
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    SOURCE_POINTS = [
        [255, 140],   # Point 1
        [1890, 145],  # Point 2
        [2050, 980],  # Point 3
        [120, 980]    # Point 4
    ]
    # ---------------------------------------------

    # Initialize Modules
    tracker = PlayerTracker()
    transformer = ViewTransformer(SOURCE_POINTS)

    # Setup Video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Map Output Dimensions
    map_width = 400
    map_height = 600
    out_width = width + map_width

    # 1. LOAD THE MAP IMAGE (Or create a green fallback)
    if map_image_path:
        print(f"Loading ground image from {map_image_path}...")
        bg_img = cv2.imread(map_image_path)
        bg_img = cv2.resize(bg_img, (map_width, map_height))
    else:
        print("Warning: Ground image not found. Using green box.")
        bg_img = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        cv2.rectangle(bg_img, (0, 0), (map_width, map_height), (34, 139, 34), -1)

    # Setup Output Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, max(height, map_height)))

    id_map = {}
    next_id = 1 

    print("Processing video... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        # 2. Reset the map for this frame (Crucial Step!)
        pitch_map = bg_img.copy()

        # 3. Track Players
        tracks = tracker.track_frame(frame)

        for track in tracks:
            raw_id, x1, y1, x2, y2 = track
            
            # ID Cleaning Logic
            if raw_id not in id_map:
                id_map[raw_id] = next_id
                next_id += 1
            clean_id = id_map[raw_id]
            
            # Draw Box on Video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {clean_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Transform to Map Coordinates
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            map_x, map_y = transformer.transform_point((foot_x, foot_y))
            
            # Draw Red Dot on Map
            # We removed the 'if' check so dots draw even if slightly outside
            cv2.circle(pitch_map, (map_x, map_y), 8, (0, 0, 255), -1) 
            cv2.circle(pitch_map, (map_x, map_y), 9, (255, 255, 255), 1) # White border
            cv2.putText(pitch_map, str(clean_id), (map_x + 10, map_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 4. Combine Video + Map side-by-side
        final_frame = np.zeros((max(height, map_height), out_width, 3), dtype=np.uint8)
        
        # Paste Video
        final_frame[:height, :width] = frame
        
        # Paste Map
        final_frame[:map_height, width:] = pitch_map

        out.write(final_frame)
        
        # Smart Resize for Display (Fits screen)
        scale_percent = 50 
        new_width = int(final_frame.shape[1] * scale_percent / 100)
        new_height = int(final_frame.shape[0] * scale_percent / 100)
        display_frame = cv2.resize(final_frame, (new_width, new_height))
        
        cv2.imshow("Cricket Tracking + Real Map", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Video saved to 'output' folder.")

if __name__ == "__main__":
    main()