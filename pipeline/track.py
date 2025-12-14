from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path='yolov8m.pt'): # Changed to 'm' (Medium)
        print(f"Loading smarter YOLO model: {model_path}...")
        self.model = YOLO(model_path)

    def track_frame(self, frame):
        # Run tracking with tweaked settings for sports
        # conf=0.25: Lowers the bar so it detects players further away
        # persist=True: CRITICAL for keeping the same ID on the same player
        # classes=[0]: Only detect "Person" (ignore bats, stumps, etc.)
        results = self.model.track(frame, persist=True, conf=0.25, classes=[0], verbose=False)
        
        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                tracks.append((track_id, int(x1), int(y1), int(x2), int(y2)))
        
        return tracks