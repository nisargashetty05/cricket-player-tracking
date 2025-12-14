import cv2
import numpy as np

class ViewTransformer:
    def __init__(self, source_points):
        """
        source_points: List of 4 (x, y) tuples from the video.
        """
        # Define the SOURCE (Video Coordinates)
        self.src_points = np.float32(source_points)
        
        # Define the DESTINATION (2D Map Coordinates)
        # We assume a map size of 400x600 pixels
        # We map the pitch to fit nicely inside with some padding
        padding = 50
        width = 300
        height = 500
        
        self.dst_points = np.float32([
            [padding, padding],             # Top-Left on Map
            [padding + width, padding],     # Top-Right on Map
            [padding + width, padding + height], # Bottom-Right on Map
            [padding, padding + height]     # Bottom-Left on Map
        ])
        
        # Calculate the Perspective Matrix
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def transform_point(self, point):
        """
        Converts a point (x, y) from video to map.
        """
        # Reshape point for matrix multiplication
        p = np.array([point], dtype='float32')
        p = np.array([p])
        
        # Apply matrix
        transformed = cv2.perspectiveTransform(p, self.matrix)
        
        # Return as integer (x, y)
        return int(transformed[0][0][0]), int(transformed[0][0][1])