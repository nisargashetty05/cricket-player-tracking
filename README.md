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
2.  Run the pipeline:
    ```bash
    python scripts/run_pipeline.py
    ```
3.  The processed video will be saved to `output/tracked_video.mp4`.

## Dependencies
* Python 3.x
* Ultralytics (YOLOv8)
* OpenCV
* NumPy