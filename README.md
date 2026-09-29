# Smart Attendance System with Face Recognition & Anti-Spoofing

An automated computer vision attendance pipeline using Python, OpenCV, and MediaPipe.

## Milestone 1: Real-Time Face & Landmark Tracking
- Real-time video processing pipeline via OpenCV at ~55 FPS.
- Sub-millisecond face bounding box and 6-point facial landmark detection using MediaPipe.
- Normalized 640x480 resolution stream for smooth inference.

## Setup & Running
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_camera.py