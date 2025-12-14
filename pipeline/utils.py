"""
Utility functions for cricket player tracking pipeline.
"""
import cv2
import numpy as np
from typing import Tuple, List


def generate_color(player_id: int) -> Tuple[int, int, int]:
    """
    Generate a unique, visible BGR color for a player ID.
    Uses HSV color space to ensure colors are always bright and distinguishable.
    
    Args:
        player_id: Unique player identifier
        
    Returns:
        BGR color tuple (B, G, R)
    """
    hue = (player_id * 45) % 180  # Spread hues evenly
    hsv_color = np.uint8([[[hue, 255, 230]]])  # Full saturation, high value
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))


def draw_styled_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                    player_id: int, color: Tuple[int, int, int] = None,
                    thickness: int = 2) -> None:
    """
    Draw a styled bounding box with player ID.
    
    Args:
        frame: Image to draw on
        x1, y1, x2, y2: Bounding box coordinates
        player_id: Player ID to display
        color: Optional BGR color, auto-generated if None
        thickness: Line thickness
    """
    if color is None:
        color = generate_color(player_id)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw ID label with background
    label = f"ID: {player_id}"
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class BoundaryFilter:
    """
    Filters detections to only include players within the cricket pitch boundary.
    Uses point-in-polygon test with expanded boundary for better coverage.
    """
    
    def __init__(self, boundary_points: List[List[int]], expand_ratio: float = 0.15):
        """
        Initialize with boundary points.
        
        Args:
            boundary_points: List of 4 corner points [[x,y], ...] defining the pitch
            expand_ratio: How much to expand the boundary (0.15 = 15% expansion)
        """
        self.original_points = np.array(boundary_points, dtype=np.float32)
        
        # Expand the boundary to include players near the edges
        self.boundary = self._expand_polygon(self.original_points, expand_ratio)
        
    def _expand_polygon(self, points: np.ndarray, ratio: float) -> np.ndarray:
        """Expand polygon outward from its center."""
        center = points.mean(axis=0)
        expanded = []
        for point in points:
            direction = point - center
            new_point = point + direction * ratio
            expanded.append(new_point)
        return np.array(expanded, dtype=np.int32)
    
    def is_inside(self, x: int, y: int) -> bool:
        """
        Check if a point is inside the boundary polygon.
        
        Args:
            x, y: Point coordinates (typically player foot position)
            
        Returns:
            True if point is inside the boundary
        """
        result = cv2.pointPolygonTest(self.boundary, (float(x), float(y)), False)
        return result >= 0
    
    def filter_tracks(self, tracks: List, use_foot_position: bool = True) -> List:
        """
        Filter tracks to only include those inside the boundary.
        
        Args:
            tracks: List of tracks [id, x1, y1, x2, y2, ...]
            use_foot_position: If True, use bottom-center of box (foot position)
            
        Returns:
            Filtered list of tracks inside the boundary
        """
        filtered = []
        for track in tracks:
            x1, y1, x2, y2 = track[1], track[2], track[3], track[4]
            
            if use_foot_position:
                # Use foot position (bottom-center of bounding box)
                check_x = int((x1 + x2) / 2)
                check_y = int(y2)
            else:
                # Use center of bounding box
                check_x = int((x1 + x2) / 2)
                check_y = int((y1 + y2) / 2)
            
            if self.is_inside(check_x, check_y):
                filtered.append(track)
                
        return filtered


class FPSCounter:
    """Simple FPS counter for performance monitoring."""
    
    def __init__(self, avg_frames: int = 30):
        self.prev_time = cv2.getTickCount()
        self.fps_history = []
        self.avg_frames = avg_frames
    
    def update(self) -> float:
        """Update and return current FPS."""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_time) / cv2.getTickFrequency()
        self.prev_time = current_time
        
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.fps_history.append(current_fps)
            if len(self.fps_history) > self.avg_frames:
                self.fps_history.pop(0)
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """Get average FPS."""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

