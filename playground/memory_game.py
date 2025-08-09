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
        # START_SCREEN, INIT, SHOW_SEQUENCE, WAIT_INPUT, CHECKING, SUCCESS, FAILURE, COUNTDOWN
        self.game_state = "START_SCREEN"
        self.sequence = []
        self.current_sequence_index = 0
        self.score = 0
        self.sequence_length = CONFIG['sequence_start_length']
        self.init_start_time = 0  # For initial instructions display

        # Start screen balloons for difficulty selection
        # Will store (x, y, radius, time_limit, color)
        self.difficulty_balloons = []
        self.balloon_hover_effects = {}  # balloon_idx: start_time

        # Game areas (circular areas randomly positioned)
        self.game_areas = []  # Will store (x, y, radius) for circular areas
        self.area_colors = []
        self.area_hands = []  # Which hand should be used for each area
        self.num_areas = 9  # Number of circular areas

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

        # Error and countdown handling
        self.countdown_start_time = 0
        self.countdown_duration = CONFIG.get(
            'countdown_duration', 5)  # 5 seconds countdown default
        self.error_message = ""
        self.error_type = "general"  # general, wrong_color, wrong_hand, wrong_area

        # Sound system
        self.sound_enabled = CONFIG['enable_sounds']
        self.sounds = {}

        # Visual overlay for camera transparency
        self.overlay = None

        # Visual feedback for interactions
        self.selected_areas = {}  # area_idx: {'start_time': time, 'type': 'correct'/'wrong'}
        self.hover_areas = {}     # area_idx: time when started hovering
        self.error_flash_start = 0  # Time when error flash started

        # Timed selection system
        # Time in seconds to hold hand over area to select
        self.selection_time_threshold = 0.75
        # area_idx: {'start_time': time, 'hand_label': str}
        self.area_selection_progress = {}

        # Selection time limit - user has limited time to make each selection
        self.selection_time_limit = 3.0  # seconds per selection
        self.selection_deadline = 0  # when current selection expires
        self.show_deadline_warning = False

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

        # Set initial timer for instructions display
        self.init_start_time = time.time()

        # Don't start game automatically - wait for difficulty selection
        # self._start_new_game()

    def _setup_difficulty_balloons(self, frame_width: int, frame_height: int) -> None:
        """Setup the difficulty selection balloons on start screen."""
        self.difficulty_balloons = []

        # Balloon configurations: (time_limit, label, color)
        balloon_configs = [
            (7, "EASY\n7s", (0, 255, 0)),      # Green - 7 seconds
            (5, "MEDIUM\n5s", (0, 165, 255)),  # Orange - 5 seconds
            (2, "HARD\n2s", (0, 0, 255))       # Red - 2 seconds
        ]

        balloon_radius = 80
        balloon_spacing = frame_width // 4
        balloon_y = frame_height // 2

        for i, (time_limit, label, color) in enumerate(balloon_configs):
            balloon_x = balloon_spacing * (i + 1)
            self.difficulty_balloons.append(
                (balloon_x, balloon_y, balloon_radius, time_limit, label, color))

        logger.info(
            f"Created {len(self.difficulty_balloons)} difficulty balloons")

    def _draw_start_screen(self, frame: np.ndarray, current_time: float) -> None:
        """Draw the start screen with difficulty selection balloons."""
        h, w = frame.shape[:2]

        # Setup balloons if not done yet
        if not self.difficulty_balloons:
            self._setup_difficulty_balloons(w, h)

        # Semi-transparent background
        overlay = np.zeros_like(frame)
        overlay[:, :] = (20, 20, 20)  # Dark background
        alpha = 0.8
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)

        # Title
        title_text = "MEMORY & COORDINATION GAME"
        title_size = cv2.getTextSize(
            title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (w - title_size[0]) // 2
        title_y = 80

        cv2.putText(frame, title_text, (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Instructions
        instruction_text = "Select Difficulty - Touch a balloon with correct hand"
        inst_size = cv2.getTextSize(
            instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        inst_x = (w - inst_size[0]) // 2
        inst_y = title_y + 60

        cv2.putText(frame, instruction_text, (inst_x, inst_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw balloons
        for i, (balloon_x, balloon_y, radius, time_limit, label, color) in enumerate(self.difficulty_balloons):
            # Check for hover effect
            current_color = color
            current_radius = radius

            if i in self.balloon_hover_effects:
                hover_duration = current_time - self.balloon_hover_effects[i]
                if hover_duration < 0.5:
                    # Pulsing effect
                    pulse = 1.0 + 0.3 * abs(np.sin(hover_duration * 10))
                    current_radius = int(radius * pulse)
                    current_color = tuple(min(255, int(c * 1.2))
                                          for c in color)
                else:
                    del self.balloon_hover_effects[i]

            # Draw balloon (circle)
            cv2.circle(frame, (balloon_x, balloon_y),
                       current_radius, current_color, -1)
            cv2.circle(frame, (balloon_x, balloon_y),
                       current_radius, (255, 255, 255), 3)

            # Draw balloon string
            string_start_y = balloon_y + current_radius
            string_end_y = string_start_y + 30
            cv2.line(frame, (balloon_x, string_start_y),
                     (balloon_x, string_end_y), (100, 100, 100), 2)

            # Draw label on balloon
            lines = label.split('\n')
            total_height = len(lines) * 25
            start_y = balloon_y - total_height // 2

            for j, line in enumerate(lines):
                text_size = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = balloon_x - text_size[0] // 2
                text_y = start_y + j * 25 + text_size[1]

                # Text with outline
                cv2.putText(frame, line, (text_x, text_y),
                            # Black outline
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, line, (text_x, text_y),
                            # White text
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw hand indicator
            # Left for easy, Right for hard, both for medium
            hand_text = "L" if i == 0 else ("R" if i == 2 else "L/R")
            hand_size = cv2.getTextSize(
                hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            hand_x = balloon_x - hand_size[0] // 2
            hand_y = string_end_y + 25

            cv2.putText(frame, hand_text, (hand_x, hand_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def _check_balloon_selection(self, hand_pos: Tuple[int, int], hand_label: str, current_time: float) -> Optional[int]:
        """Check if hand is touching a difficulty balloon."""
        if not self.difficulty_balloons:
            return None

        x, y = hand_pos

        for i, (balloon_x, balloon_y, radius, time_limit, label, color) in enumerate(self.difficulty_balloons):
            distance = ((x - balloon_x) ** 2 + (y - balloon_y) ** 2) ** 0.5

            if distance <= radius:
                # Add hover effect
                if i not in self.balloon_hover_effects:
                    self.balloon_hover_effects[i] = current_time

                # Check correct hand for balloon
                required_hand = None
                if i == 0:  # Easy - Left hand
                    required_hand = "Left"
                elif i == 2:  # Hard - Right hand
                    required_hand = "Right"
                # Medium (i == 1) accepts both hands

                if required_hand is None or hand_label == required_hand:
                    return i

        return None

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
        """Calculate circular areas randomly positioned around the screen with max 60% overlap."""
        self.game_areas = []

        # Area parameters
        min_radius = 50
        max_radius = 80
        margin = 100  # Margin from screen edges
        max_overlap_percentage = 0.6  # Maximum 60% overlap allowed

        # Generate random positions for circular areas
        attempts = 0
        max_attempts = 2000

        while len(self.game_areas) < self.num_areas and attempts < max_attempts:
            attempts += 1

            # Random position within screen bounds (considering margin)
            x = random.randint(margin, frame_width - margin)
            y = random.randint(margin, frame_height - margin)
            radius = random.randint(min_radius, max_radius)

            # Check overlap with existing areas
            valid_position = True
            for existing_x, existing_y, existing_radius in self.game_areas:
                distance = ((x - existing_x) ** 2 +
                            (y - existing_y) ** 2) ** 0.5

                # Calculate maximum allowed overlap area
                area1 = np.pi * radius ** 2
                area2 = np.pi * existing_radius ** 2
                smaller_area = min(area1, area2)
                max_overlap_area = smaller_area * max_overlap_percentage

                # Calculate actual overlap area using intersection of circles formula
                if distance < (radius + existing_radius):
                    if distance <= abs(radius - existing_radius):
                        # One circle is completely inside the other
                        overlap_area = smaller_area
                    else:
                        # Partial overlap - use circle intersection formula
                        r1, r2 = radius, existing_radius
                        d = distance

                        # Calculate intersection area
                        part1 = r1**2 * \
                            np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
                        part2 = r2**2 * \
                            np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
                        part3 = 0.5 * \
                            np.sqrt((-d + r1 + r2) * (d + r1 - r2)
                                    * (d - r1 + r2) * (d + r1 + r2))
                        overlap_area = part1 + part2 - part3

                    # Check if overlap exceeds maximum allowed
                    if overlap_area > max_overlap_area:
                        valid_position = False
                        break

            if valid_position:
                self.game_areas.append((x, y, radius))

        # If we couldn't place all areas, fill remaining with fallback positions
        if len(self.game_areas) < self.num_areas:
            logger.warning(
                f"Could only place {len(self.game_areas)}/{self.num_areas} areas with overlap constraint")
            # Use a simple grid fallback for remaining areas
            grid_size = 3
            area_spacing_x = frame_width // (grid_size + 1)
            area_spacing_y = frame_height // (grid_size + 1)

            for i in range(len(self.game_areas), self.num_areas):
                row = (i - len(self.game_areas)) // grid_size
                col = (i - len(self.game_areas)) % grid_size
                x = area_spacing_x * (col + 1)
                y = area_spacing_y * (row + 1)
                radius = 60
                self.game_areas.append((x, y, radius))

        logger.info(
            f"Generated {len(self.game_areas)} circular game areas with max 60% overlap")

    def _start_new_game(self) -> None:
        """Start a new game round."""
        self.sequence = []
        self.current_sequence_index = 0
        self.game_state = "SHOW_SEQUENCE"
        self.sequence_start_time = time.time()

        # Clear visual feedback
        self.selected_areas.clear()
        self.hover_areas.clear()
        self.area_selection_progress.clear()  # Clear selection progress
        self.error_flash_start = 0
        self.last_touched_area = -1  # Reset touched area tracking

        # Clear error messages
        self.error_message = ""
        self.error_type = "general"
        self.countdown_start_time = 0

        # Reset selection timing
        self.selection_deadline = 0
        self.show_deadline_warning = False

        # Generate random sequence
        for _ in range(self.sequence_length):
            self.sequence.append(random.randint(0, len(self.colors) - 1))

        # Setup random colors and hands for areas
        self._setup_random_areas()

        # Force recalculation of areas for new layout
        self.game_areas = []

        logger.info(
            f"Starting new game - Sequence length: {self.sequence_length}")
        logger.info(f"Sequence: {self.sequence}")

    def _get_color_name(self, color_index: int) -> str:
        """Get human-readable color name from color index."""
        color_names = [
            "Red", "Green", "Blue", "Yellow", "Magenta",
            "Cyan", "Orange", "Purple", "Pink"
        ]
        if 0 <= color_index < len(color_names):
            return color_names[color_index]
        return f"Color {color_index}"

    def _setup_random_areas(self) -> None:
        """Setup colors and hand indicators for the circular areas, ensuring sequence colors are available."""
        self.area_colors = []
        self.area_hands = []

        # First, ensure all colors from the sequence are represented in the areas
        # Get unique colors from sequence
        sequence_colors = list(set(self.sequence))
        available_positions = list(range(self.num_areas))

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
        for i in range(self.num_areas):
            if i >= len(self.area_colors) or self.area_colors[i] == -1:
                while len(self.area_colors) <= i:
                    self.area_colors.append(-1)
                self.area_colors[i] = random.randint(0, len(self.colors) - 1)

        # Setup hand indicators for all areas
        for _ in range(self.num_areas):
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

        # Note: Removed area highlighting to maintain game difficulty

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
        """Draw the circular areas with semi-transparent colors, hand indicators, and visual feedback."""
        if not self.game_areas:
            return

        current_time = time.time()

        # Create overlay for transparency
        overlay = frame.copy()

        for i, (center_x, center_y, radius) in enumerate(self.game_areas):
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
                    cv2.circle(overlay, (center_x, center_y),
                               radius + 8, glow_color, -1)
                    base_alpha = min(0.8, base_alpha + glow_alpha)
                else:
                    # Remove expired hover
                    del self.hover_areas[i]

            # Check for timed selection progress
            if i in self.area_selection_progress:
                progress = self.area_selection_progress[i]
                time_elapsed = current_time - progress['start_time']
                selection_progress = min(
                    time_elapsed / self.selection_time_threshold, 1.0)

                # Create brightening effect as selection progresses
                brightness_multiplier = 1.0 + \
                    (selection_progress * 1.5)  # Up to 2.5x brighter
                bright_color = tuple(
                    min(255, int(c * brightness_multiplier)) for c in color)

                # Add pulsing effect when near completion
                if selection_progress > 0.7:
                    pulse_intensity = abs(
                        np.sin(time_elapsed * 15)) * 0.3 + 0.7
                    bright_color = tuple(
                        min(255, int(c * brightness_multiplier * pulse_intensity)) for c in color)

                # Override base color and alpha for selection progress
                color = bright_color
                base_alpha = min(
                    0.9, CONFIG['area_transparency'] + selection_progress * 0.4)

                # Add a white border that gets thicker as selection progresses
                border_thickness = int(3 + selection_progress * 5)
                cv2.circle(overlay, (center_x, center_y), radius +
                           border_thickness, (255, 255, 255), border_thickness)

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
                        cv2.circle(overlay, (center_x, center_y),
                                   radius + 12, glow_color, -1)
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
                        cv2.circle(overlay, (center_x, center_y),
                                   radius + 12, flash_color, -1)
                        base_alpha = 0.9
                else:
                    # Remove expired selection
                    del self.selected_areas[i]

            # Draw circle with color on overlay
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)

        # Blend overlay with original frame for transparency
        cv2.addWeighted(overlay, base_alpha, frame, 1 - base_alpha, 0, frame)

        # Draw borders and hand indicators on top
        for i, (center_x, center_y, radius) in enumerate(self.game_areas):
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

            cv2.circle(frame, (center_x, center_y), radius,
                       border_color, border_thickness)

            # Draw hand indicator with better visibility
            hand_text = "L" if self.area_hands[i] == 0 else "R"
            text_size = cv2.getTextSize(
                hand_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2

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
            cv2.putText(frame, number_text, (center_x - 10, center_y - radius + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _check_area_interaction(self, hand_pos: Tuple[int, int], hand_label: str) -> Optional[int]:
        """Check if hand position is touching any circular area and return area index."""
        x, y = hand_pos
        current_time = time.time()

        for i, (center_x, center_y, radius) in enumerate(self.game_areas):
            # Calculate distance from hand to circle center
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance <= radius:
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

    def _check_area_touched(self, hand_pos: Tuple[int, int]) -> Optional[int]:
        """Check if hand position is touching any circular area regardless of hand type."""
        x, y = hand_pos

        for i, (center_x, center_y, radius) in enumerate(self.game_areas):
            # Calculate distance from hand to circle center
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if distance <= radius:
                return i
        return None

    def _check_timed_area_selection(self, hand_pos: Tuple[int, int], hand_label: str, current_time: float) -> Optional[int]:
        """Check for timed selection - hand must stay over circular area for a duration to select it."""
        x, y = hand_pos

        # Check which area the hand is currently over
        current_area = None
        for i, (center_x, center_y, radius) in enumerate(self.game_areas):
            # Calculate distance from hand to circle center
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if distance <= radius:
                expected_hand = "Left" if self.area_hands[i] == 0 else "Right"
                if hand_label == expected_hand:
                    current_area = i
                    break

        # Clean up progress for areas no longer being hovered
        areas_to_remove = []
        for area_idx in self.area_selection_progress:
            if area_idx != current_area:
                areas_to_remove.append(area_idx)

        for area_idx in areas_to_remove:
            del self.area_selection_progress[area_idx]

        # If hand is over a valid area
        if current_area is not None:
            if current_area not in self.area_selection_progress:
                # Start timing selection
                self.area_selection_progress[current_area] = {
                    'start_time': current_time,
                    'hand_label': hand_label
                }
            else:
                # Check if enough time has passed
                progress = self.area_selection_progress[current_area]
                time_elapsed = current_time - progress['start_time']

                if time_elapsed >= self.selection_time_threshold:
                    # Selection completed - remove from progress and return area
                    del self.area_selection_progress[current_area]
                    return current_area

        return None

    def _draw_error_notification(self, frame: np.ndarray) -> None:
        """Draw animated error notification overlay."""
        if self.error_flash_start <= 0:
            return

        current_time = time.time()
        error_duration = current_time - self.error_flash_start

        if error_duration > 4.0:  # Extended error notification duration
            self.error_flash_start = 0
            return

        h, w = frame.shape[:2]

        # Create pulsing red overlay
        overlay = frame.copy()

        # Calculate flash intensity (fade out over time)
        intensity = max(0, 1 - error_duration / 2.0)

        # Create pulsing effect
        pulse_frequency = 6  # pulses per second
        pulse = 0.3 + 0.7 * \
            abs(np.sin(error_duration * pulse_frequency * 2 * np.pi))

        # Red overlay with pulsing intensity
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :] = (0, 0, 255)  # Red color

        alpha = intensity * pulse * 0.25  # Max 25% opacity
        cv2.addWeighted(frame, 1 - alpha, red_overlay, alpha, 0, frame)

        # Draw error text - use custom message if available
        error_text = self.error_message if self.error_message else "WRONG!"

        # Split long messages into multiple lines
        lines = []
        if len(error_text) > 25:
            words = error_text.split()
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= 25:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
        else:
            lines = [error_text]

        # Draw each line
        text_alpha = intensity * pulse
        if text_alpha > 0.3:
            font_scale = 1.5 if len(lines) > 1 else 2
            thickness = 3 if len(lines) > 1 else 6
            line_height = 60

            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2 - (len(lines) - 1) * \
                    line_height // 2 + i * line_height

                # Black outline
                cv2.putText(frame, line, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                # White text
                cv2.putText(frame, line, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def _draw_selection_timer(self, frame: np.ndarray, current_time: float) -> None:
        """Draw the selection timer when waiting for input."""
        if self.game_state != "WAIT_INPUT" or self.selection_deadline == 0:
            return

        remaining_time = max(0, self.selection_deadline - current_time)
        h, w = frame.shape[:2]

        # Timer position (top center)
        timer_x = w // 2
        timer_y = 120

        # Timer circle background
        circle_radius = 40
        circle_color = (100, 100, 100)  # Gray background

        # Warning color when time is running out
        if remaining_time <= 1.0:
            # Pulsing red when less than 1 second
            pulse = 0.5 + 0.5 * abs(np.sin(current_time * 8))
            circle_color = (0, 0, int(255 * pulse))
        elif remaining_time <= 2.0:
            # Orange when less than 2 seconds
            circle_color = (0, 165, 255)

        # Draw circle background
        cv2.circle(frame, (timer_x, timer_y), circle_radius, circle_color, -1)
        cv2.circle(frame, (timer_x, timer_y),
                   circle_radius, (255, 255, 255), 3)

        # Draw timer arc showing remaining time
        if remaining_time > 0:
            progress = remaining_time / self.selection_time_limit
            start_angle = -90  # Start from top
            end_angle = start_angle + (360 * progress)  # Progress clockwise

            arc_color = (0, 255, 0)  # Green
            if remaining_time <= 1.0:
                arc_color = (0, 0, 255)  # Red
            elif remaining_time <= 2.0:
                arc_color = (0, 165, 255)  # Orange

            cv2.ellipse(frame, (timer_x, timer_y), (circle_radius - 5, circle_radius - 5),
                        0, start_angle, end_angle, arc_color, 6)

        # Timer text
        timer_text = f"{remaining_time:.1f}s"
        text_size = cv2.getTextSize(
            timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = timer_x - text_size[0] // 2
        text_y = timer_y + text_size[1] // 2

        cv2.putText(frame, timer_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # "Select Color" instruction
        instruction_text = f"Select Next Color ({self.current_sequence_index + 1}/{len(self.sequence)})"
        inst_size = cv2.getTextSize(
            instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        inst_x = timer_x - inst_size[0] // 2
        inst_y = timer_y + circle_radius + 30

        cv2.putText(frame, instruction_text, (inst_x, inst_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_selection_progress_indicator(self, frame: np.ndarray, current_time: float) -> None:
        """Draw progress indicator for any circular area currently being selected."""
        if not self.area_selection_progress:
            return

        h, w = frame.shape[:2]

        for area_idx, progress_info in self.area_selection_progress.items():
            if area_idx < len(self.game_areas):
                center_x, center_y, radius = self.game_areas[area_idx]

                # Calculate progress
                time_elapsed = current_time - progress_info['start_time']
                progress = min(
                    time_elapsed / self.selection_time_threshold, 1.0)

                # Draw progress arc around the circle
                start_angle = -90  # Start from top
                end_angle = start_angle + \
                    (360 * progress)  # Progress clockwise

                # Draw background arc (full circle)
                cv2.ellipse(frame, (center_x, center_y), (radius + 15, radius + 15),
                            0, 0, 360, (50, 50, 50), 8)

                # Draw progress arc
                if progress > 0:
                    # Color changes from yellow to green as it progresses
                    color = (0, int(255 * progress), int(255 *
                             (1 - progress)))  # Yellow to green
                    cv2.ellipse(frame, (center_x, center_y), (radius + 15, radius + 15),
                                0, start_angle, end_angle, color, 8)

                # Show percentage text
                percentage_text = f"{int(progress * 100)}%"
                text_size = cv2.getTextSize(
                    percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y - radius - 30

                # Background for text
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, percentage_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_countdown_notification(self, frame: np.ndarray) -> None:
        """Draw countdown notification for next session."""
        if self.game_state != "COUNTDOWN" or self.countdown_start_time <= 0:
            return

        current_time = time.time()
        elapsed = current_time - self.countdown_start_time
        remaining = max(0, self.countdown_duration - elapsed)

        if remaining <= 0:
            return

        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = np.zeros_like(frame)
        overlay[:, :] = (50, 50, 50)  # Dark gray
        alpha = 0.7
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)

        # Countdown number
        countdown_num = int(remaining) + 1
        countdown_text = str(countdown_num)

        # Large countdown number
        font_scale = 8
        thickness = 12
        text_size = cv2.getTextSize(
            countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2

        # Pulsing effect for countdown
        pulse = 0.8 + 0.2 * abs(np.sin(elapsed * 3 * np.pi))
        pulse_scale = font_scale * pulse
        pulse_thickness = int(thickness * pulse)

        # Red countdown number with white outline
        cv2.putText(frame, countdown_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, pulse_scale, (0, 0, 0), pulse_thickness + 4)
        cv2.putText(frame, countdown_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, pulse_scale, (0, 0, 255), pulse_thickness)

        # "Restarting in..." text
        restart_text = "Restarting in..."
        restart_size = cv2.getTextSize(
            restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        restart_x = (w - restart_size[0]) // 2
        restart_y = text_y - 100

        cv2.putText(frame, restart_text, (restart_x, restart_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(frame, restart_text, (restart_x, restart_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Error message at bottom
        if self.error_message:
            error_lines = []
            if len(self.error_message) > 40:
                words = self.error_message.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= 40:
                        current_line += (" " if current_line else "") + word
                    else:
                        if current_line:
                            error_lines.append(current_line)
                        current_line = word
                if current_line:
                    error_lines.append(current_line)
            else:
                error_lines = [self.error_message]

            for i, line in enumerate(error_lines):
                error_size = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                error_x = (w - error_size[0]) // 2
                error_y = text_y + 120 + i * 30

                cv2.putText(frame, line, (error_x, error_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                cv2.putText(frame, line, (error_x, error_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 200), 2)

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
        if self.game_state == "INIT":
            # Show instructions for a few seconds before starting
            if current_time - self.init_start_time > 2.0:  # 2 seconds of instructions
                # Start the actual game
                self._start_new_game()

        elif self.game_state == "SHOW_SEQUENCE":
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
            # Set deadline when entering input phase for the first time
            if self.selection_deadline == 0:
                self.selection_deadline = current_time + self.selection_time_limit
                logger.info(
                    f"Selection deadline set: {self.selection_time_limit}s from now")

            # Check if time limit exceeded
            remaining_time = self.selection_deadline - current_time
            if remaining_time <= 0:
                # Time's up!
                self.error_type = "timeout"
                self.error_message = f"Time's Up! You had {self.selection_time_limit}s to select"
                self.game_state = "FAILURE"
                self.error_flash_start = current_time
                self._play_sound('error')
                logger.info("Selection timeout - game over")
            elif remaining_time <= 1.0:
                # Show warning when less than 1 second remains
                self.show_deadline_warning = True

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
            logger.info("Game Over! Starting countdown to restart...")
            self.game_state = "COUNTDOWN"
            self.countdown_start_time = time.time()

        elif self.game_state == "COUNTDOWN":
            # Countdown before restarting
            elapsed = time.time() - self.countdown_start_time
            if elapsed >= self.countdown_duration:
                self.score = 0
                self.sequence_length = CONFIG['sequence_start_length']
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

                # Handle start screen balloon selection
                if self.game_state == "START_SCREEN":
                    balloon_idx = self._check_balloon_selection(
                        (x, y), hand_label, current_time)
                    if balloon_idx is not None:
                        # Get selected time limit
                        _, _, _, time_limit, _, _ = self.difficulty_balloons[balloon_idx]
                        self.selection_time_limit = time_limit
                        logger.info(
                            f"Difficulty selected: {time_limit}s per selection")

                        # Start the game
                        self.game_state = "INIT"
                        self.init_start_time = current_time
                        self._play_sound('sequence')

                # Check if hand is currently touching any area (only during game)
                if self.game_state in ["WAIT_INPUT", "SUCCESS", "FAILURE"]:
                    for i, (center_x, center_y, radius) in enumerate(self.game_areas):
                        distance = ((x - center_x) ** 2 +
                                    (y - center_y) ** 2) ** 0.5
                        if distance <= radius:
                            any_hand_touching_areas = True
                            break

                # Check for area interactions during input phase
                if self.game_state == "WAIT_INPUT":
                    # Always update hover effects for visual feedback
                    self._check_area_interaction((x, y), hand_label)

                    # Check for timed selection (only if cooldown has passed)
                    if current_time - self.last_interaction_time > self.interaction_cooldown:
                        area_idx = self._check_timed_area_selection(
                            (x, y), hand_label, current_time)
                        if area_idx is not None:
                            # Check if this is the correct color in sequence
                            expected_color = self.sequence[self.current_sequence_index]
                            actual_color = self.area_colors[area_idx]

                            # Debug logging
                            expected_color_name = self._get_color_name(
                                expected_color)
                            actual_color_name = self._get_color_name(
                                actual_color)
                            logger.info(
                                f"Selection attempt: Position {self.current_sequence_index + 1}/{len(self.sequence)}, Expected: {expected_color_name}, Got: {actual_color_name}")

                            # Update interaction tracking
                            self.last_interaction_time = current_time
                            self.last_touched_area = area_idx

                            if expected_color == actual_color:
                                # Correct selection
                                self.current_sequence_index += 1

                                # Reset selection deadline for next selection
                                if self.current_sequence_index < len(self.sequence):
                                    self.selection_deadline = current_time + self.selection_time_limit
                                    self.show_deadline_warning = False
                                    logger.info(
                                        f"Correct selection! Deadline reset for next selection")
                                else:
                                    # Sequence complete, no more deadline
                                    self.selection_deadline = 0
                                    self.show_deadline_warning = False

                                # Add visual feedback for correct selection
                                self.selected_areas[area_idx] = {
                                    'start_time': current_time,
                                    'type': 'correct'
                                }

                                # Play success sound
                                self._play_sound('success')

                                logger.info(
                                    f"Correct! Progress: {self.current_sequence_index}/{len(self.sequence)}")
                            else:
                                # Wrong selection
                                self.error_type = "wrong_color"
                                self.error_message = f"Wrong Color! Expected {expected_color_name}, got {actual_color_name} (Position {self.current_sequence_index + 1})"

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

                                logger.info(f"Error: {self.error_message}")

        # Reset last touched area if no hand is currently touching any area
        if not any_hand_touching_areas and current_time - self.last_interaction_time > 0.5:
            # Reset last touched area if no hand is currently touching any area
            self.last_touched_area = -1
        if not any_hand_touching_areas and current_time - self.last_interaction_time > 0.5:
            self.last_touched_area = -1

        # Draw game elements based on state
        if self.game_state == "START_SCREEN":
            self._draw_start_screen(frame, current_time)

        elif self.game_state == "SHOW_SEQUENCE":
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

            # Draw selection timer during input phase
            self._draw_selection_timer(frame, current_time)

            # Draw selection progress indicators (progress bars above areas being selected)
            self._draw_selection_progress_indicator(frame, current_time)

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

        # Draw countdown notification
        self._draw_countdown_notification(frame)

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
        elif key == ord('r') and self.game_state in ["FAILURE", "COUNTDOWN"]:
            # Allow manual restart during failure/countdown
            logger.info("Manual restart requested by user.")
            self.score = 0
            self.sequence_length = CONFIG['sequence_start_length']
            # Return to start screen for difficulty selection
            self.game_state = "START_SCREEN"
            self.difficulty_balloons = []
            self.balloon_hover_effects = {}
        elif key == ord('s'):
            # Skip countdown with 's' key
            if self.game_state == "COUNTDOWN":
                logger.info("Countdown skipped by user.")
                self.score = 0
                self.sequence_length = CONFIG['sequence_start_length']
                # Return to start screen for difficulty selection
                self.game_state = "START_SCREEN"
                self.difficulty_balloons = []
                self.balloon_hover_effects = {}

    def _draw_ui(self, frame: np.ndarray, current_time: float) -> None:
        """Draw game UI elements."""
        h, w = frame.shape[:2]

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

        # Instructions for new players (show only during INIT state)
        if self.game_state == "INIT":
            instructions = [
                f"Starting game with {self.selection_time_limit}s per selection",
                "1. Watch the sequence of colors",
                "2. Touch areas with the correct hand (L/R shown)",
                "3. Follow the sequence in order",
                "Get ready!"
            ]

            # Semi-transparent background
            overlay = np.zeros_like(frame)
            overlay[:, :] = (0, 0, 0)
            alpha = 0.6
            cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)

            for i, instruction in enumerate(instructions):
                if i == 0:  # Title
                    font_scale = 1.2
                    thickness = 3
                    color = (0, 255, 255)
                else:
                    font_scale = 0.8
                    thickness = 2
                    color = (255, 255, 255)

                text_size = cv2.getTextSize(
                    instruction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2 - 100 + i * 40

                cv2.putText(frame, instruction, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

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
