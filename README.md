
# Hand-Controlled Robotic Arm 

## Introduction
This Python script uses OpenCV and MediaPipe to track hand movements and control a robotic arm based on those movements. It allows for hand gesture recognition to move the robotic arm to predefined positions or dynamically track the wrist to guide the arm.

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- numpy
- xArm Python SDK

## Installation
Ensure Python 3.8 or higher is installed on your system. You can then install the required libraries using pip:
```bash
pip install opencv-python mediapipe numpy xArm-Python-SDK
```

## Usage
To run the script, navigate to the script's directory in the terminal and execute:
```bash
python gesture.py
```
Ensure the xArm and your webcam are properly configured and connected to your computer. The script should automatically begin tracking your hand movements and move the robotic arm accordingly.

## Function Descriptions
- **map_coordinates_to_angles(x, y, width, height)**: Converts the webcam coordinates to angles for the robotic arm.
- **is_hand_closed(landmarks)**: Determines if the hand gesture is closed based on finger positions.

## Troubleshooting
- If the camera feed does not appear, ensure that your webcam is properly connected and accessible.
- If the arm does not respond to gestures, check that the arm's IP is correctly configured and that it is connected to the same network as your computer.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

##Acknowledgement:
Credit to Pavly Hamin & Dhyey
