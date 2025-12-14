# Cricket Player Detection and Tracking

## Project Overview
This project implements a computer vision pipeline to detect cricket players and track them with unique IDs across video frames. It utilizes **YOLOv8** for detection and **BoT-SORT** for multi-object tracking.

## Features
* **Player Detection:** Identifies all visible players in the footage.
* **Multi-Object Tracking:** Assigns and maintains unique IDs for each player.
* **Visual Output:** Generates a processed video with bounding boxes and ID labels.

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
    pip install -r requirements.txt
    ```

## Usage
1.  Place your raw video file in `data/cricket_match.mp4`.
    link for cricket_match = "https://drive.google.com/file/d/1NL9gd7WZpKxCm0Kg-WkEeEfhy_HBCHbh/view?usp=drive_link"
2.  Run the pipeline:
    ```bash
    python scripts/run_pipeline.py
    ```
3.  The processed video will be saved to `output/tracked_video.mp4`.

## Optional Enhancements implemented
### Bird's Eye View (Minimap)
I implemented a Perspective Transformation module to map player positions from the video onto a 2D top-down view of the cricket pitch.

**How it works:**
1.  **Calibration:** A helper script (`scripts/get_points.py`) allows the user to click 4 corners of the pitch in the video.
2.  **Transformation:** Using OpenCV's `getPerspectiveTransform`, the pixel coordinates are mapped to a 2D plane.
3.  **Visualization:** Players are plotted as red dots on a `Ground_image.png` template for realistic tactical analysis.

## Dependencies
* Python 3.x
* Ultralytics (YOLOv8)
* OpenCV
* NumPy