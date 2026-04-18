import cv2
import mediapipe as mp
from xarm.wrapper import XArmAPI
import numpy as np

DEGREES_TO_RADIANS = np.pi / 180
MIN_ROTATION_DEG, MAX_ROTATION_DEG = -360, 360

arm = XArmAPI('192.168.1.220')
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

workspace_height = 720
workspace_width = 1024

def calculate_hand_rotation(landmarks):
    wrist = landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    index_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

    dx = index_mcp.x - pinky_mcp.x
    dy = index_mcp.y - pinky_mcp.y
    rotation_rad = np.arctan2(dy, dx)
    rotation_deg = np.degrees(rotation_rad)

    return rotation_deg

def map_rotation_to_servo_angle(rotation_deg):
    servo_angle = np.interp(rotation_deg, [-180, 180], [MIN_ROTATION_DEG, MAX_ROTATION_DEG])
    return servo_angle

def is_hand_open(landmarks):
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
    open_fingers = 0
    for fingertip, lower_joint in zip(fingertip_indices, lower_joint_indices):
        fingertip_pos = landmarks.landmark[fingertip]
        lower_joint_pos = landmarks.landmark[lower_joint]
        if fingertip_pos.y < lower_joint_pos.y:
            open_fingers += 1
    return open_fingers >= 3

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
hand_connections = mp.solutions.hands.HAND_CONNECTIONS

SPEED = 60

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

                if is_hand_open(hand_landmarks):
                    rotation_deg = calculate_hand_rotation(hand_landmarks)
                    servo_angle = map_rotation_to_servo_angle(rotation_deg)

                    try:
                        arm.set_servo_angle(angle=[0, 0, 0, 0, 0, servo_angle, 0], speed=SPEED, wait=False)
                        print(f"Rotation: {rotation_deg:.1f} deg -> Servo angle: {servo_angle:.1f} deg")
                    except Exception as e:
                        print(f"Failed to rotate arm: {e}")

                    cv2.putText(image, f"Rotation: {rotation_deg:.1f} deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Hand closed - rotation paused",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Arm Rotation Control', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    arm.disconnect()
