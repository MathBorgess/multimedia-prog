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
        self.active_camera_index = None
        self.available_cameras = []  # Store list of working cameras

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

        # First, discover all available cameras
        if not self.available_cameras:
            self._discover_available_cameras()

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
                        f"✓ SUCCESS: Using Camera {cam_idx} with resolution {w}x{h}")
                    logger.info(f"Available cameras: {self.available_cameras}")
                    logger.info("Press 'c' to switch camera, 'q' to quit")

                    # Store the camera index being used for reference
                    self.active_camera_index = cam_idx
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

        # Add camera info overlay
        if self.active_camera_index is not None:
            cv2.putText(frame, f"Camera: {self.active_camera_index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to switch camera, 'q' to quit", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Virtual Drums', frame)

        # Check for keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Exit requested by user.")
            raise SystemExit
        elif key == ord('c'):
            logger.info("Camera switch requested by user.")
            self._switch_camera()

    def _reinit_camera(self) -> None:
        """Reinitialize camera if it fails during runtime."""
        logger.info("Attempting to reinitialize camera...")
        old_camera = self.active_camera_index
        if self.cap:
            self.cap.release()
        self._init_camera_with_retry()
        if self.active_camera_index != old_camera:
            logger.info(
                f"Camera switched from {old_camera} to {self.active_camera_index}")

    def _discover_available_cameras(self) -> None:
        """Discover all available cameras."""
        logger.info("Discovering available cameras...")
        self.available_cameras = []

        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.available_cameras.append(i)
                    logger.info(f"Found working camera at index {i}")
            cap.release()

        if not self.available_cameras:
            logger.warning("No working cameras found!")
        else:
            logger.info(f"Total available cameras: {self.available_cameras}")

    def _switch_camera(self) -> None:
        """Switch to the next available camera."""
        if len(self.available_cameras) <= 1:
            logger.info("Only one camera available, cannot switch.")
            return

        try:
            current_index = self.available_cameras.index(
                self.active_camera_index)
            next_index = (current_index + 1) % len(self.available_cameras)
            next_camera = self.available_cameras[next_index]

            logger.info(
                f"Switching from camera {self.active_camera_index} to camera {next_camera}")

            # Release current camera
            if self.cap:
                self.cap.release()

            # Initialize new camera
            self.cap = cv2.VideoCapture(next_camera)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {next_camera}")
                # Try to go back to previous camera
                self._reinit_camera()
                return

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Test if camera works
            ret, frame = self.cap.read()
            if not ret:
                logger.error(
                    f"Camera {next_camera} opened but cannot read frames")
                self._reinit_camera()
                return

            # Update kit with new resolution if needed
            h, w = frame.shape[:2]
            self.kit = DrumKit((w, h))

            self.active_camera_index = next_camera
            logger.info(f"✓ Successfully switched to camera {next_camera}")

        except (ValueError, Exception) as e:
            logger.error(f"Error switching camera: {e}")
            # Try to reinitialize current camera
            self._reinit_camera()

    def cleanup(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()
        mixer.quit()
        logger.info("Resources cleaned up.")
