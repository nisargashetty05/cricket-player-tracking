import cv2
import sys
import os
import numpy as np

# Add parent directory to path so we can import from pipeline folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.track import PlayerTracker
from pipeline.transformer import ViewTransformer
from pipeline.utils import BoundaryFilter

def main():
    # --- FILE PATHS ---
    input_video_path = "data/cricket_match.mp4"
    output_video_path = "output/tracked_video_with_map.mp4"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    # Use yolov8s (small) model for better detection of distant players
    tracker = PlayerTracker(model_path='yolov8s.pt')
    transformer = ViewTransformer(SOURCE_POINTS)
    
    # Initialize boundary filter to exclude detections outside the pitch
    boundary_filter = BoundaryFilter(SOURCE_POINTS, expand_ratio=0.15)
    print("Boundary filter initialized - will exclude players outside pitch area")

    # Setup Video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if width == 0 or height == 0:
        print("Error: Invalid video dimensions. Please check the video file.")
        cap.release()
        return
    
    # Map Output Dimensions
    map_width = 400
    map_height = 600
    out_width = width + map_width

    # 1. LOAD THE MAP IMAGE (Or create a green fallback)
    if map_image_path:
        print(f"Loading ground image from {map_image_path}...")
        bg_img = cv2.imread(map_image_path)
        if bg_img is not None:
            bg_img = cv2.resize(bg_img, (map_width, map_height))
        else:
            print("Warning: Could not load ground image. Using green box.")
            bg_img = np.zeros((map_height, map_width, 3), dtype=np.uint8)
            cv2.rectangle(bg_img, (0, 0), (map_width, map_height), (34, 139, 34), -1)
    else:
        print("Warning: Ground image not found. Using green box.")
        bg_img = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        cv2.rectangle(bg_img, (0, 0), (map_width, map_height), (34, 139, 34), -1)

    # Setup Output Video
    # Use integer codec for mp4v (0x7634706D) or let OpenCV choose with -1
    fourcc = 0x7634706D  # 'mp4v' codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, max(height, map_height)))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_video_path}")
        cap.release()
        return

    id_map = {}
    next_id = 1 
    
    # Trajectory storage: {player_id: [(map_x, map_y), ...]}
    trajectories = {}
    max_trajectory_length = 50  # Maximum number of points to store per player

    # Progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    print(f"Processing video ({total_frames} frames)... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        frame_count += 1
        
        # Progress indicator (every 30 frames)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"\rProgress: {frame_count}/{total_frames} ({progress:.1f}%) | Players tracked: {len(id_map)}", end="")

        # 2. Reset the map for this frame (Crucial Step!)
        pitch_map = bg_img.copy()

        # 3. Track Players
        tracks = tracker.track_frame(frame)
        
        # 4. Filter tracks to only include players inside the pitch boundary
        tracks = boundary_filter.filter_tracks(tracks)

        for track in tracks:
            # New format: [id, x1, y1, x2, y2, confidence]
            raw_id, x1, y1, x2, y2 = track[0], track[1], track[2], track[3], track[4]
            
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
            
            # Update trajectory for this player
            if clean_id not in trajectories:
                trajectories[clean_id] = []
            trajectories[clean_id].append((map_x, map_y))
            
            # Limit trajectory length
            if len(trajectories[clean_id]) > max_trajectory_length:
                trajectories[clean_id].pop(0)
            
            # Draw trajectory path (fading effect)
            if len(trajectories[clean_id]) > 1:
                points = np.array(trajectories[clean_id], dtype=np.int32)
                # Generate a unique, visible color for each player using HSV
                # This ensures colors are always bright and distinguishable
                hue = (clean_id * 45) % 180  # Spread hues evenly
                # Create HSV color and convert to BGR
                hsv_color = np.uint8([[[hue, 255, 230]]])  # Full saturation, high value
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                
                # Draw trajectory line connecting all points
                for i in range(1, len(points)):
                    # Fade older points (lighter color for older segments)
                    alpha = i / len(points)
                    faded_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                    cv2.line(pitch_map, tuple(points[i-1]), tuple(points[i]), faded_color, 2)
            
            # Draw Red Dot on Map (current position)
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
    print(f"\n\nDone! Processed {frame_count} frames.")
    print(f"Total unique players tracked: {len(id_map)}")
    print(f"Video saved to: {output_video_path}")

if __name__ == "__main__":
    main()