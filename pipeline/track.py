from ultralytics import YOLO
import cv2

class PlayerTracker:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the Tracker with the YOLO model.
        """
        print(f"Loading YOLOv8 model with Tracking: {model_path}...")
        self.model = YOLO(model_path)

    def track_frame(self, frame):
        """
        Takes a frame, tracks players, and returns the results.
        Output: A list of tracks: [id, x1, y1, x2, y2]
        """
        # Run YOLOv8 tracking
        # persist=True tells the model to remember this frame for the next one (crucial for ID consistency)
        # classes=[0] tracks only Persons (Class 0)
        # conf=0.3: Only track players if the model is 30% sure (removes garbage detections)
        # iou=0.5: Helps handle overlaps better
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                           conf=0.3, iou=0.5, classes=[0], verbose=False)
        
        tracked_objects = []

        # Parse the results
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
                
            # Get the boxes and IDs
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                tracked_objects.append([int(track_id), int(x1), int(y1), int(x2), int(y2)])
        
        return tracked_objects