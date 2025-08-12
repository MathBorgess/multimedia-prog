#!/usr/bin/env python3
"""
Test script for thumbs up gesture detection.
This script shows the camera feed and detects thumbs up gestures.
"""

import cv2
import mediapipe as mp
import time
from color_touch import ColorTouch


def test_thumbs_up():
    """Test the thumbs up gesture detection."""

    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Create a color touch instance to use its gesture detection
    game = ColorTouch()

    print("Testing thumbs up detection...")
    print("Show thumbs up gesture to the camera")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
                # Get hand label
                hand_label = handedness.classification[0].label

                # Get thumb tip position
                thumb_lm = hand_landmarks.landmark[4]
                thumb_x, thumb_y = int(thumb_lm.x * w), int(thumb_lm.y * h)

                # Detect thumbs up using the game's method
                is_thumbs_up = game._detect_thumbs_up(hand_landmarks)

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw detection result
                if is_thumbs_up:
                    cv2.circle(frame, (thumb_x, thumb_y), 30, (0, 255, 0), 5)
                    cv2.putText(frame, f"{hand_label} THUMBS UP!", (thumb_x - 100, thumb_y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, "👍", (thumb_x - 20, thumb_y - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                else:
                    cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 0, 255), 3)
                    cv2.putText(frame, f"{hand_label} Hand", (thumb_x - 50, thumb_y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add instructions
        cv2.putText(frame, "Show thumbs up gesture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Thumbs Up Test', frame)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Test completed!")


if __name__ == "__main__":
    test_thumbs_up()
