import asyncio
import logging
import time
import random
from typing import Tuple, Dict, List, Optional
import cv2
import mediapipe as mp
import numpy as np
import pygame
from config.config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)


class MemoryGame:
    """Main class for the memory and coordination game."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.cap = None
        self.active_camera_index = None
        self.available_cameras = []

        # Game state
        self.game_state = "INIT"  # INIT, SHOW_SEQUENCE, WAIT_INPUT, CHECKING, SUCCESS, FAILURE
        self.sequence = []
        self.current_sequence_index = 0
        self.score = 0
        self.sequence_length = CONFIG['sequence_start_length']

        # Game areas (3x3 grid)
        self.game_areas = []
        self.area_colors = []
        self.area_hands = []  # Which hand should be used for each area

        # Timing
        self.sequence_start_time = 0
        self.last_sequence_sound = 0
        self.last_interaction_time = 0
        self.interaction_cooldown = 0.5  # seconds

        # Colors for the game
        self.colors = CONFIG['game_colors']

        # Hand detection
        self.prev_positions: Dict[str, Tuple[int, int, float]] = {}
        self.hand_labels = []

        # Sound system
        self.sound_enabled = CONFIG['enable_sounds']
        self.sounds = {}

        # Visual overlay for camera transparency
        self.overlay = None

    def _check_camera_permissions(self) -> None:
        """Check camera permissions on macOS."""
        import platform
        if platform.system() == "Darwin":  # macOS
            logger.info(
                "Running on macOS - if this is the first time running, you may need to grant camera permissions")
            logger.info(
                "If prompted, please allow camera access in System Preferences > Security & Privacy > Camera")

    def setup(self) -> None:
        """Initialize MediaPipe, camera, and sound system."""
        self._check_camera_permissions()

        # Initialize sound system
        if self.sound_enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-
                                  16, channels=1, buffer=512)
                self._load_sounds()
                logger.info("Sound system initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize sound system: {e}")
                self.sound_enabled = False

        try:
            self.hands = self.mp_hands.Hands(**CONFIG['hands_config'])
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise

        # Initialize camera
        self._init_camera_with_retry()
        self._setup_game_areas()
        self._start_new_game()

    def _load_sounds(self) -> None:
        """Load sound files."""
        for sound_name, sound_file in CONFIG['sound_files'].items():
            try:
                self.sounds[sound_name] = pygame.mixer.Sound(sound_file)
                logger.info(f"Loaded sound: {sound_name}")
            except Exception as e:
                logger.warning(f"Failed to load sound {sound_name}: {e}")

    def _play_sound(self, sound_name: str) -> None:
        """Play a sound effect."""
        if self.sound_enabled and sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except Exception as e:
                logger.warning(f"Failed to play sound {sound_name}: {e}")

    def _init_camera_with_retry(self) -> None:
        """Initialize camera with retry mechanism."""
        max_retries = 3
        camera_indices = [CONFIG['camera_index'], 0, 1]

        if not self.available_cameras:
            self._discover_available_cameras()

        for retry in range(max_retries):
            for cam_idx in camera_indices:
                try:
                    logger.info(
                        f"Attempting to initialize camera {cam_idx} (attempt {retry + 1}/{max_retries})")

                    if self.cap:
                        self.cap.release()

                    self.cap = cv2.VideoCapture(cam_idx)

                    if not self.cap.isOpened():
                        logger.warning(
                            f"Camera index {cam_idx} could not be opened")
                        continue

                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                    # Test frame reading
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning(
                            f"Camera index {cam_idx} opened but cannot read frames")
                        continue

                    h, w = frame.shape[:2]
                    logger.info(
                        f"✓ SUCCESS: Using Camera {cam_idx} with resolution {w}x{h}")
                    logger.info(f"Available cameras: {self.available_cameras}")

                    self.active_camera_index = cam_idx
                    return

                except Exception as e:
                    logger.warning(
                        f"Failed to initialize camera {cam_idx}: {e}")
                    continue

            if retry < max_retries - 1:
                logger.info("Waiting 2 seconds before retry...")
                time.sleep(2)

        raise RuntimeError(
            "Could not initialize any camera after multiple attempts.")

    def _discover_available_cameras(self) -> None:
        """Discover all available cameras."""
        logger.info("Discovering available cameras...")
        self.available_cameras = []

        for i in range(5):
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

    def _setup_game_areas(self) -> None:
        """Setup the 3x3 grid of game areas."""
        # These will be calculated based on frame dimensions in each frame
        pass

    def _calculate_game_areas(self, frame_width: int, frame_height: int) -> None:
        """Calculate the 3x3 grid areas based on frame dimensions."""
        self.game_areas = []
        area_width = frame_width // 3
        area_height = frame_height // 3

        for row in range(3):
            for col in range(3):
                x = col * area_width
                y = row * area_height
                self.game_areas.append((x, y, area_width, area_height))

    def _start_new_game(self) -> None:
        """Start a new game round."""
        self.sequence = []
        self.current_sequence_index = 0
        self.game_state = "SHOW_SEQUENCE"
        self.sequence_start_time = time.time()

        # Generate random sequence
        for _ in range(self.sequence_length):
            self.sequence.append(random.randint(0, len(self.colors) - 1))

        # Setup random colors and hands for areas
        self._setup_random_areas()

        logger.info(
            f"Starting new game - Sequence length: {self.sequence_length}")
        logger.info(f"Sequence: {self.sequence}")

    def _setup_random_areas(self) -> None:
        """Setup random colors and hand indicators for the 9 areas."""
        self.area_colors = []
        self.area_hands = []

        for _ in range(9):
            # Random color index
            color_idx = random.randint(0, len(self.colors) - 1)
            self.area_colors.append(color_idx)

            # Random hand (0 = left, 1 = right)
            hand = random.randint(0, 1)
            self.area_hands.append(hand)

    def _draw_border_flash(self, frame: np.ndarray, color_index: int, elapsed_time: float) -> None:
        """Draw flashing border with the current sequence color."""
        flash_duration = CONFIG['border_flash_duration']
        if elapsed_time > flash_duration:
            return

        color = self.colors[color_index]
        h, w = frame.shape[:2]
        thickness = CONFIG['border_thickness']

        # Flash effect (alternate visibility)
        flash_cycle = 0.3  # seconds
        if (elapsed_time % (flash_cycle * 2)) < flash_cycle:
            # Draw border
            cv2.rectangle(frame, (0, 0), (w, thickness), color, -1)  # Top
            cv2.rectangle(frame, (0, h-thickness), (w, h), color, -1)  # Bottom
            cv2.rectangle(frame, (0, 0), (thickness, h), color, -1)  # Left
            cv2.rectangle(frame, (w-thickness, 0), (w, h), color, -1)  # Right

    def _draw_game_areas(self, frame: np.ndarray) -> None:
        """Draw the 3x3 grid with semi-transparent colors and hand indicators."""
        if not self.game_areas:
            return

        # Create overlay for transparency
        overlay = frame.copy()

        for i, (x, y, w, h) in enumerate(self.game_areas):
            # Draw area with color on overlay
            color = self.colors[self.area_colors[i]]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

        # Blend overlay with original frame for transparency
        alpha = CONFIG['area_transparency']
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw borders and hand indicators on top
        for i, (x, y, w, h) in enumerate(self.game_areas):
            # Draw border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

            # Draw hand indicator
            hand_text = "L" if self.area_hands[i] == 0 else "R"
            text_size = cv2.getTextSize(
                hand_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            # Draw text with outline for better visibility
            cv2.putText(frame, hand_text, (text_x, text_y),
                        # Black outline
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            cv2.putText(frame, hand_text, (text_x, text_y),
                        # White text
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def _check_area_interaction(self, hand_pos: Tuple[int, int], hand_label: str) -> Optional[int]:
        """Check if hand position is touching any area and return area index."""
        x, y = hand_pos

        for i, (area_x, area_y, area_w, area_h) in enumerate(self.game_areas):
            if (area_x <= x <= area_x + area_w and
                    area_y <= y <= area_y + area_h):

                # Check if correct hand is used
                expected_hand = "Left" if self.area_hands[i] == 0 else "Right"
                if hand_label == expected_hand:
                    return i

        return None

    def _process_game_logic(self, current_time: float) -> None:
        """Process the main game logic based on current state."""
        if self.game_state == "SHOW_SEQUENCE":
            elapsed = current_time - self.sequence_start_time
            color_duration = CONFIG['border_flash_duration'] + \
                CONFIG['color_pause_duration']

            # Play sequence sound for each color
            current_color_idx = int(elapsed // color_duration)
            if (current_color_idx < len(self.sequence) and
                    current_time - self.last_sequence_sound > color_duration):
                self._play_sound('sequence')
                self.last_sequence_sound = current_time

            if elapsed >= len(self.sequence) * color_duration:
                # Sequence display finished
                self.game_state = "WAIT_INPUT"
                self.current_sequence_index = 0
                logger.info(
                    "Sequence display finished. Waiting for user input.")

        elif self.game_state == "WAIT_INPUT":
            # Waiting for user to complete the sequence
            if self.current_sequence_index >= len(self.sequence):
                self.game_state = "SUCCESS"

        elif self.game_state == "SUCCESS":
            # Player completed sequence correctly
            self._play_sound('success')
            self.score += 1
            self.sequence_length += 1
            logger.info(
                f"Success! Score: {self.score}, Next sequence length: {self.sequence_length}")
            time.sleep(CONFIG['success_pause_duration'])
            self._start_new_game()

        elif self.game_state == "FAILURE":
            # Player made a mistake
            self._play_sound('error')
            logger.info("Game Over! Restarting...")
            self.score = 0
            self.sequence_length = CONFIG['sequence_start_length']
            time.sleep(CONFIG['failure_pause_duration'])
            self._start_new_game()

    def update_loop(self) -> None:
        """Process one frame of the video feed."""
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized or closed.")
            return

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Calculate game areas if not done
        if not self.game_areas:
            self._calculate_game_areas(w, h)

        # Process hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        current_time = time.time()
        detected_hands = []

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
                # Get hand label
                hand_label = handedness.classification[0].label

                # Get index finger tip position
                lm = hand_landmarks.landmark[8]
                x, y = int(lm.x * w), int(lm.y * h)

                detected_hands.append((x, y, hand_label))

                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Check for area interactions during input phase
                if self.game_state == "WAIT_INPUT" and current_time - self.last_interaction_time > self.interaction_cooldown:
                    area_idx = self._check_area_interaction((x, y), hand_label)
                    if area_idx is not None:
                        # Check if this is the correct color in sequence
                        expected_color = self.sequence[self.current_sequence_index]
                        actual_color = self.area_colors[area_idx]

                        if expected_color == actual_color:
                            self.current_sequence_index += 1
                            self.last_interaction_time = current_time
                            logger.info(
                                f"Correct! Progress: {self.current_sequence_index}/{len(self.sequence)}")
                        else:
                            self.game_state = "FAILURE"
                            self.last_interaction_time = current_time
                            logger.info("Wrong color! Game over.")

        # Draw game elements based on state
        if self.game_state == "SHOW_SEQUENCE":
            elapsed = current_time - self.sequence_start_time
            color_duration = CONFIG['border_flash_duration'] + \
                CONFIG['color_pause_duration']
            current_color_idx = int(elapsed // color_duration)

            if current_color_idx < len(self.sequence):
                color_elapsed = elapsed - (current_color_idx * color_duration)
                self._draw_border_flash(
                    frame, self.sequence[current_color_idx], color_elapsed)

        elif self.game_state in ["WAIT_INPUT", "SUCCESS", "FAILURE"]:
            self._draw_game_areas(frame)

        # Draw UI
        self._draw_ui(frame)

        # Process game logic
        self._process_game_logic(current_time)

        cv2.imshow('Memory & Coordination Game', frame)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Exit requested by user.")
            raise SystemExit

    def _draw_ui(self, frame: np.ndarray) -> None:
        """Draw game UI elements."""
        # Score
        cv2.putText(frame, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Sequence length
        cv2.putText(frame, f"Length: {self.sequence_length}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Game state
        state_text = {
            "SHOW_SEQUENCE": "Memorize the sequence!",
            "WAIT_INPUT": f"Touch areas in order ({self.current_sequence_index + 1}/{len(self.sequence)})",
            "SUCCESS": "Success! Next level...",
            "FAILURE": "Game Over! Restarting..."
        }.get(self.game_state, "")

        if state_text:
            cv2.putText(frame, state_text, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def cleanup(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()
        if self.sound_enabled:
            pygame.mixer.quit()
        logger.info("Resources cleaned up.")
