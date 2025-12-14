from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_version='yolov8n.pt'):
        """
        Initialize the YOLO model.
        We use 'yolov8n.pt' (nano) because it downloads automatically and runs fast on CPU.
        """
        print(f"Loading YOLO model: {model_version}...")
        self.model = YOLO(model_version)

    def detect(self, frame):
        """
        Input: A single video frame (image).
        Output: A list of bounding boxes for detected people: [x1, y1, x2, y2, score]
        """
        # Run the model on the frame
        # verbose=False keeps the terminal output clean
        results = self.model(frame, classes=[0], verbose=False) 
        # classes=[0] forces it to ONLY look for people (Class ID 0)

        detections = []
        
        # Extract the bounding box data
        for result in results:
            for box in result.boxes:
                # Get coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Get confidence score
                conf = float(box.conf[0])
                
                detections.append([x1, y1, x2, y2, conf])
        
        return detections