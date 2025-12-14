import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.track import PlayerTracker

def main():
    # 1. Setup Video Paths
    input_video_path = "data/cricket_match.mp4"
    output_video_path = "output/tracked_video.mp4"

    if not os.path.exists(input_video_path):
        print(f"Error: Video not found at {input_video_path}")
        return

    # 2. Initialize Tracker
    tracker = PlayerTracker()

    # 3. Open Video
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video... Output will be saved to {output_video_path}")
    print("Press 'q' to stop viewing early.")

    # This dictionary will hold the translation: {random_id: simple_id}
    id_map = {}
    next_id = 1 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        # 4. Track Players
        tracks = tracker.track_frame(frame)

        # 5. Draw IDs and Boxes
        for track in tracks:
            raw_id, x1, y1, x2, y2 = track
            
            # If we haven't seen this 'raw_id' before, give it a new simple number
            if raw_id not in id_map:
                id_map[raw_id] = next_id
                next_id += 1
            
            # Use the simple number for display
            clean_id = id_map[raw_id]
            
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Unique ID (Using clean_id now!)
            label = f"ID: {clean_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)

        out.write(frame)

        # Show the frame
        display_frame = cv2.resize(frame, (1024, 600))
        cv2.imshow("Cricket Player Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Video saved in 'output' folder.")

if __name__ == "__main__":
    main()