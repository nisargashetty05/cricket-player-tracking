# Cricket Player Detection and Tracking

## Project Overview
This project implements a computer vision pipeline to detect cricket players and track them with unique IDs across video frames. It utilizes **YOLOv8** for detection and **ByteTrack** for multi-object tracking.

## Features
* **Player Detection:** Identifies all visible players in the footage using YOLOv8.
* **Multi-Object Tracking:** Assigns and maintains unique IDs for each player using ByteTrack algorithm.
* **Visual Output:** Generates a processed video with bounding boxes and ID labels.
* **Trajectory Visualization:** Displays movement paths showing player trajectories over time on the top-view map.

## Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # On Windows Git Bash
    ```

3.  Install dependencies:
    ```bash
    pip install -r Requirements.txt
    ```

## Usage
1.  Place your raw video file in `data/cricket_match.mp4`.
    link for cricket_match = "https://drive.google.com/file/d/1NL9gd7WZpKxCm0Kg-WkEeEfhy_HBCHbh/view?usp=drive_link"
2.  Run the pipeline:
    ```bash
    python scripts/run_pipeline.py
    ```
3.  The processed video will be saved to `output/tracked_video_with_map.mp4`.

## Optional Enhancements Implemented

### 1. Bird's Eye View (Top-View Projection)
A Perspective Transformation module maps player positions from the video onto a 2D top-down view of the cricket pitch.

**How it works:**
1.  **Calibration:** A helper script (`scripts/get_points.py`) allows the user to click 4 corners of the pitch in the video.
2.  **Transformation:** Using OpenCV's `getPerspectiveTransform`, the pixel coordinates are mapped to a 2D plane via homography.
3.  **Visualization:** Players are plotted as red dots on a `Ground_image.png` template for realistic tactical analysis.

### 2. Trajectory Visualization
The system displays movement paths showing player trajectories over time on the top-view map.

**Features:**
- **Movement Paths:** Trajectory lines connect previous positions, showing player movement patterns
- **Color-Coded Trails:** Each player has a unique color for their trajectory path
- **Fading Effect:** Older trajectory segments fade out, making recent movement more prominent
- **Configurable Length:** Trajectory history limited to last 50 positions per player (configurable)

This enhancement enables:
- Analysis of player movement patterns
- Understanding of field coverage
- Tactical positioning insights

## Dependencies
* Python 3.x
* Ultralytics (YOLOv8)
* OpenCV
* NumPy