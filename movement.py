import cv2
import mediapipe as mp
from xarm.wrapper import XArmAPI
import numpy as np

DEGREES_TO_RADIANS = np.pi / 180
MIN_ANGLE_DEG, MAX_ANGLE_DEG = -360, 360

arm = XArmAPI('192.168.1.220')
arm.motion_enable(enable=True)
arm.set_mode(0) 
arm.set_state(state=0)  

workspace_height = 720
workspace_width = 1024

def map_coordinates_to_angles(x, y, width, height):
    x_normalized = (x - width / 2) / (width / 2) * 100
    y_normalized = (y - height / 2) / (height / 2) * 100

    min_angle_deg_x, max_angle_deg_x = -360, 360  
    min_angle_deg_y, max_angle_deg_y = -100, 100

    x_angle = np.interp(x_normalized, [-1, 1], [min_angle_deg_x, max_angle_deg_x]) * DEGREES_TO_RADIANS
    y_angle = np.interp(y_normalized, [-1, 1], [min_angle_deg_y, max_angle_deg_y]) * DEGREES_TO_RADIANS

    print(x,y)

    return x_angle, y_angle

# for corner, (x, y) in corners.items():
#     x_angle, y_angle = map_coordinates_to_angles(x, y, workspace_width, workspace_height)
#     print(f"{corner} - X={x_angle} rad, Y={y_angle} rad")

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
hand_connections = mp.solutions.hands.HAND_CONNECTIONS

SPEED = 100

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, hand_connections)
                wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * workspace_width)
                wrist_y = int(wrist.y * workspace_height)
                x_angle, y_angle = map_coordinates_to_angles(wrist_x, wrist_y, workspace_width, workspace_height)
                try:
                    arm.set_servo_angle(angle=[x_angle, y_angle], speed=SPEED, wait=False)
                    print(f"Moving to angles: X={x_angle} rad, Y={y_angle} rad")
                except Exception as e:
                    print(f"Failed to move arm: {e}")

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    arm.disconnect()