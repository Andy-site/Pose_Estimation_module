# Pose Detection Module

This repository contains a Python module for real-time pose detection using the MediaPipe library. The module detects human poses and calculates angles between specific body joints.

## Features
<img width="723" alt="pose_landmarks_index" src="https://github.com/user-attachments/assets/2434a9c1-a536-4f80-8cd6-1887ffd62751">

- Detects poses from a video input or camera feed.
- Calculates angles between specified body landmarks (e.g., elbow, shoulder).
- Displays the detected landmarks and angles on the video feed.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe 
```
Usage

Place a video file named 1.mp4 in the PoseVideos directory, or modify the VideoCapture argument in main() to your video file path.

Run the module:

```bash

    python pose_detection.py
```
##Code Explanation

    PoseDetector Class: Contains methods for detecting poses, finding landmark positions, and calculating angles between landmarks.
    findPose(img): Processes the input image and draws the detected landmarks.
    findPosition(img): Returns the positions of the detected landmarks.
    findAngle(img, p1, p2, p3): Calculates the angle formed by three specified landmarks.
