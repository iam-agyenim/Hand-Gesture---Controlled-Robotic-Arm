import cv2
import mediapipe as mp
from xarm.wrapper import XArmAPI
import numpy as np

DEGREES_TO_RADIANS = np.pi / 180
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
    return x_angle, y_angle

def is_hand_closed(landmarks):
    fingertip_indices = [
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mp.solutions.hands.HandLandmark.PINKY_TIP
    ]
    lower_joint_indices = [
        mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
        mp.solutions.hands.HandLandmark.PINKY_PIP
    ]
    closed_fingers = 0
    for fingertip, lower_joint in zip(fingertip_indices, lower_joint_indices):
        fingertip_pos = landmarks.landmark[fingertip]
        lower_joint_pos = landmarks.landmark[lower_joint]
        if fingertip_pos.y > lower_joint_pos.y:
            closed_fingers += 1
    return closed_fingers >= 3

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
hand_connections = mp.solutions.hands.HAND_CONNECTIONS

SPEED = 40

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
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(image, hand_landmarks, hand_connections)
                if is_hand_closed(hand_landmarks):
                    label = handedness.classification[0].label
                    if label == 'Left':
                        angle = -180 * DEGREES_TO_RADIANS 
                        try:
                            arm.set_servo_angle(angle=[-180, 0, 0, 0, 0, 0, 0], speed=SPEED, wait=True)
                            print(f"Hand {label} closed: moving arm to {angle} radians.")
                        except Exception as e:
                            print(f"Failed to move arm: {e}")
                    elif label == 'Right':
                        angle = 180 * DEGREES_TO_RADIANS 
                        try:
                            arm.set_servo_angle(angle=[180, 0, 0, 0, 0, 0, 0], speed=SPEED, wait=True)
                            print(f"Hand {label} closed: moving arm to {angle} radians.")
                        except Exception as e:
                            print(f"Failed to move arm: {e}")
                else:
                    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    wrist_x = int(wrist.x * workspace_width)
                    wrist_y = int(wrist.y * workspace_height)
                    x_angle, y_angle = map_coordinates_to_angles(wrist_x, wrist_y, workspace_width, workspace_height)
                    try:
                        arm.set_servo_angle(angle=[x_angle, y_angle], speed=SPEED, wait=False)
                        print(f"Open hand: Moving to angles: X={x_angle} rad, Y={y_angle} rad")
                    except Exception as e:
                        print(f"Failed to move arm: {e}")

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    arm.disconnect()
