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
        self.interaction_cooldown = 1.0  # Increased to prevent double detection
        self.last_touched_area = -1  # Track last touched area to prevent repeated touches

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

        # Visual feedback for interactions
        self.selected_areas = {}  # area_idx: {'start_time': time, 'type': 'correct'/'wrong'}
        self.hover_areas = {}     # area_idx: time when started hovering
        self.error_flash_start = 0  # Time when error flash started

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

        # Clear visual feedback
        self.selected_areas.clear()
        self.hover_areas.clear()
        self.error_flash_start = 0
        self.last_touched_area = -1  # Reset touched area tracking

        # Generate random sequence
        for _ in range(self.sequence_length):
            self.sequence.append(random.randint(0, len(self.colors) - 1))

        # Setup random colors and hands for areas
        self._setup_random_areas()

        logger.info(
            f"Starting new game - Sequence length: {self.sequence_length}")
        logger.info(f"Sequence: {self.sequence}")

    def _setup_random_areas(self) -> None:
        """Setup colors and hand indicators for the 9 areas, ensuring sequence colors are available."""
        self.area_colors = []
        self.area_hands = []

        # First, ensure all colors from the sequence are represented in the areas
        # Get unique colors from sequence
        sequence_colors = list(set(self.sequence))
        available_positions = list(range(9))

        # Place sequence colors in random positions
        for color_idx in sequence_colors:
            if available_positions:
                pos = random.choice(available_positions)
                available_positions.remove(pos)
                # Temporarily store color at this position
                while len(self.area_colors) <= pos:
                    self.area_colors.append(-1)
                self.area_colors[pos] = color_idx

        # Fill remaining positions with random colors (can include duplicates)
        for pos in available_positions:
            color_idx = random.randint(0, len(self.colors) - 1)
            while len(self.area_colors) <= pos:
                self.area_colors.append(-1)
            self.area_colors[pos] = color_idx

        # Fill any gaps with random colors
        for i in range(9):
            if i >= len(self.area_colors) or self.area_colors[i] == -1:
                while len(self.area_colors) <= i:
                    self.area_colors.append(-1)
                self.area_colors[i] = random.randint(0, len(self.colors) - 1)

        # Setup hand indicators for all 9 areas
        for _ in range(9):
            # Random hand (0 = left, 1 = right)
            hand = random.randint(0, 1)
            self.area_hands.append(hand)

        logger.info(f"Area colors: {self.area_colors}")
        logger.info(f"Sequence colors needed: {sequence_colors}")

    def _draw_border_flash(self, frame: np.ndarray, color_index: int, elapsed_time: float) -> None:
        """Draw flashing border with the current sequence color and highlight matching areas."""
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

        # Highlight areas that match the current sequence color
        self._highlight_matching_areas(frame, color_index, elapsed_time)

    def _highlight_matching_areas(self, frame: np.ndarray, color_index: int, elapsed_time: float) -> None:
        """Highlight areas that match the current sequence color during sequence display."""
        if not self.game_areas:
            return

        flash_cycle = 0.4  # seconds for area highlighting
        show_highlight = (elapsed_time % (flash_cycle * 2)) < flash_cycle

        if show_highlight:
            # Find areas with matching color
            for i, (x, y, w, h) in enumerate(self.game_areas):
                if i < len(self.area_colors) and self.area_colors[i] == color_index:
                    # Create a bright outline around matching areas
                    highlight_color = (255, 255, 255)  # White highlight
                    thickness = 8
                    cv2.rectangle(frame, (x-thickness//2, y-thickness//2),
                                  (x + w + thickness//2, y + h + thickness//2),
                                  highlight_color, thickness)

                    # Add a subtle glow effect
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h),
                                  highlight_color, -1)
                    cv2.addWeighted(frame, 0.9, overlay, 0.1, 0, frame)

    def _draw_sequence_progress(self, frame: np.ndarray, current_idx: int, total: int, current_color: int) -> None:
        """Draw sequence progress and current color indicator."""
        h, w = frame.shape[:2]

        # Draw progress text
        progress_text = f"Memorize: {current_idx + 1}/{total}"
        text_size = cv2.getTextSize(
            progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 50

        # Background for text
        padding = 10
        cv2.rectangle(frame, (text_x - padding, text_y - text_size[1] - padding),
                      (text_x + text_size[0] + padding, text_y + padding),
                      (0, 0, 0), -1)

        # Text
        cv2.putText(frame, progress_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw current color indicator
        color = self.colors[current_color]
        color_rect_size = 40
        color_x = text_x + text_size[0] + 20
        color_y = text_y - color_rect_size + 10

        cv2.rectangle(frame, (color_x, color_y),
                      (color_x + color_rect_size, color_y + color_rect_size),
                      color, -1)
        cv2.rectangle(frame, (color_x, color_y),
                      (color_x + color_rect_size, color_y + color_rect_size),
                      (255, 255, 255), 2)

    def _draw_game_areas(self, frame: np.ndarray) -> None:
        """Draw the 3x3 grid with semi-transparent colors, hand indicators, and visual feedback."""
        if not self.game_areas:
            return

        current_time = time.time()

        # Create overlay for transparency
        overlay = frame.copy()

        for i, (x, y, w, h) in enumerate(self.game_areas):
            # Base color
            color = self.colors[self.area_colors[i]]
            base_alpha = CONFIG['area_transparency']

            # Check for hover effect
            if i in self.hover_areas:
                hover_duration = current_time - self.hover_areas[i]
                if hover_duration < 0.3:  # Hover effect duration
                    # Add glow effect for hovering
                    glow_alpha = 0.3 * (1 - hover_duration / 0.3)  # Fade out
                    glow_color = tuple(min(255, c + 100)
                                       for c in color)  # Brighter color
                    cv2.rectangle(overlay, (x-5, y-5),
                                  (x + w + 5, y + h + 5), glow_color, -1)
                    base_alpha = min(0.8, base_alpha + glow_alpha)
                else:
                    # Remove expired hover
                    del self.hover_areas[i]

            # Check for selection feedback
            if i in self.selected_areas:
                selection = self.selected_areas[i]
                selection_duration = current_time - selection['start_time']

                if selection_duration < 0.5:  # Selection effect duration
                    if selection['type'] == 'correct':
                        # Green success glow
                        glow_intensity = 1 - (selection_duration / 0.5)
                        success_color = (0, 255, 0)  # Green
                        # Create pulsing effect
                        pulse = 0.5 + 0.5 * \
                            abs(np.sin(selection_duration * 10))
                        glow_color = tuple(
                            int(c * pulse + success_color[j] * (1-pulse)) for j, c in enumerate(color))
                        cv2.rectangle(overlay, (x-8, y-8),
                                      (x + w + 8, y + h + 8), glow_color, -1)
                        base_alpha = 0.9
                    else:  # wrong
                        # Red error flash
                        flash_intensity = 1 - (selection_duration / 0.5)
                        error_color = (0, 0, 255)  # Red
                        # Create flashing effect
                        flash = 0.5 + 0.5 * \
                            abs(np.sin(selection_duration * 20))
                        flash_color = tuple(
                            int(error_color[j] * flash + c * (1-flash)) for j, c in enumerate(color))
                        cv2.rectangle(overlay, (x-8, y-8),
                                      (x + w + 8, y + h + 8), flash_color, -1)
                        base_alpha = 0.9
                else:
                    # Remove expired selection
                    del self.selected_areas[i]

            # Draw area with color on overlay
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

        # Blend overlay with original frame for transparency
        cv2.addWeighted(overlay, base_alpha, frame, 1 - base_alpha, 0, frame)

        # Draw borders and hand indicators on top
        for i, (x, y, w, h) in enumerate(self.game_areas):
            # Draw border with dynamic thickness based on feedback
            border_thickness = 3
            border_color = (255, 255, 255)

            # Enhanced border for active areas
            if i in self.selected_areas or i in self.hover_areas:
                border_thickness = 5
                if i in self.selected_areas:
                    selection = self.selected_areas[i]
                    if selection['type'] == 'correct':
                        border_color = (0, 255, 0)  # Green border for correct
                    else:
                        border_color = (0, 0, 255)  # Red border for wrong
                else:
                    border_color = (255, 255, 0)  # Yellow border for hover

            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          border_color, border_thickness)

            # Draw hand indicator with better visibility
            hand_text = "L" if self.area_hands[i] == 0 else "R"
            text_size = cv2.getTextSize(
                hand_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            # Draw text with outline for better visibility
            cv2.putText(frame, hand_text, (text_x, text_y),
                        # Black outline
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
            cv2.putText(frame, hand_text, (text_x, text_y),
                        # White text
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            # Draw area number for debugging (can be removed later)
            number_text = str(i)
            number_size = cv2.getTextSize(
                number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, number_text, (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _check_area_interaction(self, hand_pos: Tuple[int, int], hand_label: str) -> Optional[int]:
        """Check if hand position is touching any area and return area index."""
        x, y = hand_pos
        current_time = time.time()

        for i, (area_x, area_y, area_w, area_h) in enumerate(self.game_areas):
            if (area_x <= x <= area_x + area_w and
                    area_y <= y <= area_y + area_h):

                # Add hover effect for visual feedback
                if i not in self.hover_areas:
                    self.hover_areas[i] = current_time

                # Check if correct hand is used
                expected_hand = "Left" if self.area_hands[i] == 0 else "Right"
                if hand_label == expected_hand:
                    # Additional check to prevent double detection of same area
                    if i == self.last_touched_area and current_time - self.last_interaction_time < 2.0:
                        return None  # Ignore repeated touches of same area too quickly
                    return i

        return None

    def _draw_error_notification(self, frame: np.ndarray) -> None:
        """Draw animated error notification overlay."""
        if self.error_flash_start <= 0:
            return

        current_time = time.time()
        error_duration = current_time - self.error_flash_start

        if error_duration > 1.0:  # Error notification duration
            self.error_flash_start = 0
            return

        h, w = frame.shape[:2]

        # Create pulsing red overlay
        overlay = frame.copy()

        # Calculate flash intensity (fade out over time)
        intensity = max(0, 1 - error_duration)

        # Create pulsing effect
        pulse_frequency = 8  # pulses per second
        pulse = 0.3 + 0.7 * \
            abs(np.sin(error_duration * pulse_frequency * 2 * np.pi))

        # Red overlay with pulsing intensity
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :] = (0, 0, 255)  # Red color

        alpha = intensity * pulse * 0.3  # Max 30% opacity
        cv2.addWeighted(frame, 1 - alpha, red_overlay, alpha, 0, frame)

        # Draw error text
        error_text = "WRONG COLOR!"
        text_size = cv2.getTextSize(
            error_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2

        # Text with pulsing effect
        text_alpha = intensity * pulse
        if text_alpha > 0.3:
            # Black outline
            cv2.putText(frame, error_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
            # White text
            cv2.putText(frame, error_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def _draw_success_notification(self, frame: np.ndarray, progress: str) -> None:
        """Draw success notification with progress."""
        h, w = frame.shape[:2]

        # Draw success message
        success_text = f"CORRECT! ({progress})"
        text_size = cv2.getTextSize(
            success_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 100

        # Green background for text
        bg_padding = 20
        cv2.rectangle(frame,
                      (text_x - bg_padding, text_y -
                       text_size[1] - bg_padding),
                      (text_x + text_size[0] +
                       bg_padding, text_y + bg_padding),
                      (0, 150, 0), -1)

        # Black outline
        cv2.putText(frame, success_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        # White text
        cv2.putText(frame, success_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

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
        any_hand_touching_areas = False

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

                # Check if hand is currently touching any area
                for i, (area_x, area_y, area_w, area_h) in enumerate(self.game_areas):
                    if (area_x <= x <= area_x + area_w and area_y <= y <= area_y + area_h):
                        any_hand_touching_areas = True
                        break

                # Check for area interactions during input phase
                if self.game_state == "WAIT_INPUT" and current_time - self.last_interaction_time > self.interaction_cooldown:
                    area_idx = self._check_area_interaction((x, y), hand_label)
                    if area_idx is not None:
                        # Check if this is the correct color in sequence
                        expected_color = self.sequence[self.current_sequence_index]
                        actual_color = self.area_colors[area_idx]

                        # Update interaction tracking
                        self.last_interaction_time = current_time
                        self.last_touched_area = area_idx

                        if expected_color == actual_color:
                            # Correct selection
                            self.current_sequence_index += 1

                            # Add visual feedback for correct selection
                            self.selected_areas[area_idx] = {
                                'start_time': current_time,
                                'type': 'correct'
                            }

                            # Play success sound with slight delay for better feedback
                            self._play_sound('success')

                            logger.info(
                                f"Correct! Progress: {self.current_sequence_index}/{len(self.sequence)}")
                        else:
                            # Wrong selection
                            self.game_state = "FAILURE"

                            # Add visual feedback for wrong selection
                            self.selected_areas[area_idx] = {
                                'start_time': current_time,
                                'type': 'wrong'
                            }

                            # Start error notification
                            self.error_flash_start = current_time

                            # Play error sound immediately
                            self._play_sound('error')

                            logger.info("Wrong color! Game over.")

        # Reset last touched area if no hand is currently touching any area
        if not any_hand_touching_areas and current_time - self.last_interaction_time > 0.5:
            self.last_touched_area = -1

        # Draw game elements based on state
        if self.game_state == "SHOW_SEQUENCE":
            elapsed = current_time - self.sequence_start_time
            color_duration = CONFIG['border_flash_duration'] + \
                CONFIG['color_pause_duration']
            current_color_idx = int(elapsed // color_duration)

            if current_color_idx < len(self.sequence):
                color_elapsed = elapsed - (current_color_idx * color_duration)
                sequence_color = self.sequence[current_color_idx]
                self._draw_border_flash(frame, sequence_color, color_elapsed)

                # Show sequence progress
                self._draw_sequence_progress(
                    frame, current_color_idx, len(self.sequence), sequence_color)

        elif self.game_state in ["WAIT_INPUT", "SUCCESS", "FAILURE"]:
            self._draw_game_areas(frame)

            # Draw success notification during input if user is making progress
            if self.game_state == "WAIT_INPUT" and self.current_sequence_index > 0:
                progress_text = f"{self.current_sequence_index}/{len(self.sequence)}"
                # Only show for recent correct selections
                show_success = False
                for area_idx, selection in self.selected_areas.items():
                    if (selection['type'] == 'correct' and
                            current_time - selection['start_time'] < 0.5):
                        show_success = True
                        break

                if show_success:
                    self._draw_success_notification(frame, progress_text)

            # Draw error notification
            self._draw_error_notification(frame)

        # Draw UI
        self._draw_ui(frame, current_time)

        # Process game logic
        self._process_game_logic(current_time)

        cv2.imshow('Memory & Coordination Game', frame)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Exit requested by user.")
            raise SystemExit

    def _draw_ui(self, frame: np.ndarray, current_time: float) -> None:
        """Draw game UI elements."""
        # Score
        cv2.putText(frame, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Sequence length
        cv2.putText(frame, f"Length: {self.sequence_length}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Game state with more helpful instructions
        state_text = {
            "SHOW_SEQUENCE": "Memorize the sequence! Watch the flashing border colors.",
            "WAIT_INPUT": f"Touch matching areas with correct hand! ({self.current_sequence_index + 1}/{len(self.sequence)})",
            "SUCCESS": "Success! Next level...",
            "FAILURE": "Game Over! Restarting..."
        }.get(self.game_state, "")

        if state_text:
            # Multi-line text support
            lines = state_text.split('!')
            y_offset = frame.shape[0] - 60
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    cv2.putText(frame, line.strip() + ('!' if i < len(lines) - 1 else ''),
                                (10, y_offset + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show current sequence during input phase
        if self.game_state == "WAIT_INPUT" and self.sequence:
            self._draw_sequence_reminder(frame)

        # Show cooldown indicator if active
        if self.game_state == "WAIT_INPUT":
            self._draw_interaction_cooldown(frame, current_time)

    def _draw_sequence_reminder(self, frame: np.ndarray) -> None:
        """Draw a small reminder of the sequence colors during input phase."""
        if not self.sequence:
            return

        h, w = frame.shape[:2]

        # Draw sequence colors as small rectangles
        rect_size = 25
        spacing = 5
        start_x = w - (len(self.sequence) * (rect_size + spacing)) - 20
        start_y = 20

        # Background
        bg_width = len(self.sequence) * (rect_size + spacing) + 10
        cv2.rectangle(frame, (start_x - 10, start_y - 5),
                      (start_x + bg_width, start_y + rect_size + 10),
                      (0, 0, 0), -1)

        # Draw each color in sequence
        for i, color_idx in enumerate(self.sequence):
            x = start_x + i * (rect_size + spacing)
            y = start_y

            color = self.colors[color_idx]

            # Highlight current position in sequence
            if i == self.current_sequence_index:
                # Bright border for current position
                cv2.rectangle(frame, (x - 3, y - 3),
                              (x + rect_size + 3, y + rect_size + 3),
                              (255, 255, 0), 3)  # Yellow highlight
            elif i < self.current_sequence_index:
                # Green border for completed positions
                cv2.rectangle(frame, (x - 2, y - 2),
                              (x + rect_size + 2, y + rect_size + 2),
                              (0, 255, 0), 2)  # Green border

            # Draw color rectangle
            cv2.rectangle(frame, (x, y), (x + rect_size,
                          y + rect_size), color, -1)
            cv2.rectangle(frame, (x, y), (x + rect_size,
                          y + rect_size), (255, 255, 255), 1)

    def _draw_interaction_cooldown(self, frame: np.ndarray, current_time: float) -> None:
        """Draw cooldown indicator when interaction is temporarily disabled."""
        if current_time - self.last_interaction_time < self.interaction_cooldown:
            h, w = frame.shape[:2]

            # Calculate remaining cooldown time
            remaining_time = self.interaction_cooldown - \
                (current_time - self.last_interaction_time)
            cooldown_text = f"Wait: {remaining_time:.1f}s"

            # Position near the center bottom
            text_size = cv2.getTextSize(
                cooldown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 120

            # Semi-transparent background
            padding = 10
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (text_x - padding, text_y - text_size[1] - padding),
                          (text_x + text_size[0] + padding, text_y + padding),
                          (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

            # Orange text for cooldown
            cv2.putText(frame, cooldown_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

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
