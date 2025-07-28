import logging
from typing import Tuple
import cv2
import numpy as np
from pygame import mixer

logger = logging.getLogger(__name__)

class Drum:
    """Represents a single drum with position, sound, and hit detection."""
    def __init__(self, name: str, pos: Tuple[int, int], radius: int, sound_path: str, cooldown: float):
        self.name = name
        self.pos = pos
        self.radius = radius
        self.cooldown = cooldown
        self.last_hit = 0.0
        try:
            self.sound = mixer.Sound(sound_path)
        except Exception as e:
            logger.error(f"Failed to load sound {sound_path}: {e}")
            raise

    def draw(self, frame: np.ndarray, current_time: float) -> None:
        """Draw the drum on the frame, changing color if recently hit."""
        color = (0, 255, 0) if current_time - self.last_hit < self.cooldown else (0, 0, 255)
        cv2.circle(frame, self.pos, self.radius, color, 4)
        cv2.putText(
            frame, self.name, (self.pos[0] - self.radius, self.pos[1] - self.radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    def check_hit(self, hand_pos: Tuple[int, int], hand_vel: float, current_time: float, velocity_threshold: float) -> None:
        """Check if the drum is hit based on hand position and velocity."""
        dist = np.linalg.norm(np.array(hand_pos) - np.array(self.pos))
        if dist < self.radius and hand_vel > velocity_threshold:
            if current_time - self.last_hit > self.cooldown:
                try:
                    self.sound.play()
                    self.last_hit = current_time
                except Exception as e:
                    logger.error(f"Error playing sound for {self.name}: {e}")