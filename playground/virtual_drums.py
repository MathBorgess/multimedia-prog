import asyncio
import logging
import time
from typing import Tuple, Dict
import cv2
import mediapipe as mp
from pygame import mixer
from playground.drum_kit import DrumKit
from config.config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)


class VirtualDrums:
    """Main class for the virtual drums application."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.cap = None
        self.kit = None
        self.prev_positions: Dict[int, Tuple[int, int, float]] = {}

    def _check_camera_permissions(self) -> None:
        """Check camera permissions on macOS."""
        import platform
        if platform.system() == "Darwin":  # macOS
            logger.info(
                "Running on macOS - if this is the first time running, you may need to grant camera permissions")
            logger.info(
                "If prompted, please allow camera access in System Preferences > Security & Privacy > Camera")

    def setup(self) -> None:
        """Initialize pygame, MediaPipe, and camera."""
        # Check camera permissions first
        self._check_camera_permissions()

        try:
            mixer.init()
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            raise

        try:
            self.hands = self.mp_hands.Hands(**CONFIG['hands_config'])
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise

        # Try to initialize camera with retry logic
        self._init_camera_with_retry()

    def _init_camera_with_retry(self) -> None:
        """Initialize camera with retry mechanism."""
        max_retries = 3
        # Try config index first, then fallback
        camera_indices = [CONFIG['camera_index'], 0, 1]

        for retry in range(max_retries):
            for cam_idx in camera_indices:
                try:
                    logger.info(
                        f"Attempting to initialize camera {cam_idx} (attempt {retry + 1}/{max_retries})")

                    # Release any existing capture
                    if self.cap:
                        self.cap.release()

                    self.cap = cv2.VideoCapture(cam_idx)

                    if not self.cap.isOpened():
                        logger.warning(
                            f"Camera index {cam_idx} could not be opened")
                        continue

                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                    # Try to read a frame
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning(
                            f"Camera index {cam_idx} opened but cannot read frames")
                        continue

                    h, w = frame.shape[:2]
                    self.kit = DrumKit((w, h))
                    logger.info(
                        f"Camera {cam_idx} initialized successfully with resolution {w}x{h}")
                    return

                except Exception as e:
                    logger.warning(
                        f"Failed to initialize camera {cam_idx}: {e}")
                    continue

            if retry < max_retries - 1:
                import time
                logger.info(f"Waiting 2 seconds before retry...")
                time.sleep(2)

        raise RuntimeError(
            "Could not initialize any camera after multiple attempts. Please check camera permissions and availability.")

    def update_loop(self) -> None:
        """Process one frame of the video feed."""
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized or closed.")
            return

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            # Try to reinitialize camera if frame reading fails
            try:
                self._reinit_camera()
            except Exception as e:
                logger.error(f"Failed to reinitialize camera: {e}")
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                lm = hand_landmarks.landmark[8]  # Index finger tip
                w, h = frame.shape[1], frame.shape[0]
                x, y = int(lm.x * w), int(lm.y * h)
                t = asyncio.get_event_loop().time()

                # Compute vertical velocity
                vel = 0.0
                if idx in self.prev_positions:
                    px, py, pt = self.prev_positions[idx]
                    dt = t - pt
                    vel = (y - py) / dt if dt > 0 else 0.0
                self.prev_positions[idx] = (x, y, t)

                # Process interaction
                self.kit.interact((x, y), vel)

                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        self.kit.draw(frame)
        cv2.imshow('Virtual Drums', frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Exit requested by user.")
            raise SystemExit

    def _reinit_camera(self) -> None:
        """Reinitialize camera if it fails during runtime."""
        logger.info("Attempting to reinitialize camera...")
        if self.cap:
            self.cap.release()
        self._init_camera_with_retry()

    def cleanup(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()
        mixer.quit()
        logger.info("Resources cleaned up.")
