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

SPEED = 50
GRIPPER_SPEED = 5000

PICK_HEIGHT = 150
PLACE_HEIGHT = 150
SAFE_HEIGHT = 300
PLACE_POSITION = [300, 200, PLACE_HEIGHT]

arm.set_gripper_enable(True)
arm.set_gripper_speed(GRIPPER_SPEED)

def map_coordinates_to_position(x, y, width, height):
    x_pos = np.interp(x, [0, width], [-500, 500])
    y_pos = np.interp(y, [0, height], [200, 600])
    return x_pos, y_pos

def count_extended_fingers(landmarks):
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
    thumb_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]

    count = 0
    if thumb_tip.x < thumb_ip.x:
        count += 1

    for fingertip, lower_joint in zip(fingertip_indices, lower_joint_indices):
        if landmarks.landmark[fingertip].y < landmarks.landmark[lower_joint].y:
            count += 1
    return count

def is_hand_closed(landmarks):
    return count_extended_fingers(landmarks) <= 1

def is_hand_open(landmarks):
    return count_extended_fingers(landmarks) >= 4

def is_pinch_gesture(landmarks):
    thumb_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05

def is_peace_sign(landmarks):
    index_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP]

    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_down = ring_tip.y > ring_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y

    return index_up and middle_up and ring_down and pinky_down

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
hand_connections = mp.solutions.hands.HAND_CONNECTIONS

STATE_IDLE = 0
STATE_NAVIGATE = 1
STATE_PICK = 2
STATE_PLACE = 3

state = STATE_IDLE
target_x, target_y = 0, 0

cap = cv2.VideoCapture(0)

print("=== Farm Robotic Arm - Pick & Place ===")
print("Gestures:")
print("  Open hand    -> Navigate (move arm to wrist position)")
print("  Pinch        -> Pick (lower arm and grip)")
print("  Peace sign   -> Place (move to drop-off and release)")
print("  Closed fist  -> Return to home / idle")
print("Press 'q' to quit.")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status_text = "IDLE"
        status_color = (200, 200, 200)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, hand_connections)

                wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * workspace_width)
                wrist_y = int(wrist.y * workspace_height)

                if is_pinch_gesture(hand_landmarks):
                    if state != STATE_PICK:
                        state = STATE_PICK
                        try:
                            arm.set_position(x=target_x, y=target_y, z=PICK_HEIGHT, speed=SPEED, wait=True)
                            arm.set_gripper_position(0, wait=True)
                            arm.set_position(x=target_x, y=target_y, z=SAFE_HEIGHT, speed=SPEED, wait=True)
                            print(f"Picked at X={target_x:.0f}, Y={target_y:.0f}")
                        except Exception as e:
                            print(f"Failed to pick: {e}")
                    status_text = "PICK"
                    status_color = (0, 165, 255)

                elif is_peace_sign(hand_landmarks):
                    if state != STATE_PLACE:
                        state = STATE_PLACE
                        try:
                            arm.set_position(x=PLACE_POSITION[0], y=PLACE_POSITION[1],
                                             z=SAFE_HEIGHT, speed=SPEED, wait=True)
                            arm.set_position(x=PLACE_POSITION[0], y=PLACE_POSITION[1],
                                             z=PLACE_HEIGHT, speed=SPEED, wait=True)
                            arm.set_gripper_position(800, wait=True)
                            arm.set_position(x=PLACE_POSITION[0], y=PLACE_POSITION[1],
                                             z=SAFE_HEIGHT, speed=SPEED, wait=True)
                            print("Placed at drop-off location.")
                        except Exception as e:
                            print(f"Failed to place: {e}")
                    status_text = "PLACE"
                    status_color = (0, 255, 0)

                elif is_hand_open(hand_landmarks):
                    state = STATE_NAVIGATE
                    x_pos, y_pos = map_coordinates_to_position(wrist_x, wrist_y,
                                                               workspace_width, workspace_height)
                    target_x, target_y = x_pos, y_pos
                    try:
                        arm.set_position(x=x_pos, y=y_pos, z=SAFE_HEIGHT, speed=SPEED, wait=False)
                        print(f"Navigating to X={x_pos:.0f}, Y={y_pos:.0f}")
                    except Exception as e:
                        print(f"Failed to navigate: {e}")
                    status_text = "NAVIGATE"
                    status_color = (255, 255, 0)

                elif is_hand_closed(hand_landmarks):
                    if state != STATE_IDLE:
                        state = STATE_IDLE
                        try:
                            arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0, 0], speed=SPEED, wait=True)
                            arm.set_gripper_position(800, wait=True)
                            print("Returned to home position.")
                        except Exception as e:
                            print(f"Failed to return home: {e}")
                    status_text = "HOME"
                    status_color = (0, 0, 255)

        cv2.putText(image, f"State: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(image, f"Target: ({target_x:.0f}, {target_y:.0f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Farm Arm - Pick & Place', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    arm.set_gripper_position(800, wait=True)
    cap.release()
    cv2.destroyAllWindows()
    arm.disconnect()
