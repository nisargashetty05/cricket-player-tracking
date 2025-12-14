from ultralytics import YOLO  # type: ignore
import cv2
import os

class PlayerTracker:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the Tracker with the YOLO model.
        Uses custom ByteTrack configuration for improved ID consistency.
        """
        print(f"Loading YOLOv8 model with Tracking: {model_path}...")
        self.model = YOLO(model_path)
        
        # Path to custom tracker config (optimized for cricket)
        self.tracker_config = os.path.join(
            os.path.dirname(__file__), 
            'custom_bytetrack.yaml'
        )
        
        # Fallback to default if custom config doesn't exist
        if not os.path.exists(self.tracker_config):
            print("Warning: Custom tracker config not found, using default bytetrack.yaml")
            self.tracker_config = "bytetrack.yaml"
        else:
            print(f"Using custom tracker config: {self.tracker_config}")

    def track_frame(self, frame):
        """
        Takes a frame, tracks players, and returns the results.
        Output: A list of tracks: [id, x1, y1, x2, y2]
        
        Optimized parameters for cricket player tracking:
        - persist=True: Maintain track state across frames (crucial for ID consistency)
        - conf=0.25: Lower threshold to detect partially occluded/distant players
        - iou=0.45: Better handling of overlapping players
        - imgsz=1280: Higher resolution for better detection accuracy
        """
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_config,
            conf=0.25,       # Lower confidence to catch more players
            iou=0.45,        # Adjusted IoU for better overlap handling
            classes=[0],     # Only track persons (class 0)
            imgsz=1280,      # Higher resolution for better accuracy
            verbose=False
        )
        
        tracked_objects = []

        # Parse the results
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
                
            # Get the boxes and IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # type: ignore
            track_ids = result.boxes.id.cpu().numpy()  # type: ignore
            confidences = result.boxes.conf.cpu().numpy()  # type: ignore
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = box
                # Include confidence for potential filtering
                tracked_objects.append([
                    int(track_id), 
                    int(x1), int(y1), 
                    int(x2), int(y2),
                    float(conf)
                ])
        
        return tracked_objects