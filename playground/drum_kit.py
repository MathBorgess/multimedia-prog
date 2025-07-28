from typing import Tuple, List
import numpy as np
import asyncio

from playground.drum import Drum
from config.config import CONFIG


class DrumKit:
    """Manages a collection of drums."""

    def __init__(self, frame_dim: Tuple[int, int]):
        self.drums: List[Drum] = []
        w, h = frame_dim
        for drum_config in CONFIG['drums']:
            pos = (int(w * drum_config['pos'][0]),
                   int(h * drum_config['pos'][1]))
            self.drums.append(Drum(
                drum_config['name'], pos, drum_config['radius'],
                drum_config['sound'], CONFIG['drum_cooldown']
            ))

    def draw(self, frame: np.ndarray) -> None:
        """Draw all drums on the frame."""
        current_time = asyncio.get_event_loop().time()
        for drum in self.drums:
            drum.draw(frame, current_time)

    def interact(self, hand_pos: Tuple[int, int], hand_vel: float) -> None:
        """Check for interactions with all drums."""
        current_time = asyncio.get_event_loop().time()
        for drum in self.drums:
            drum.check_hit(hand_pos, hand_vel, current_time,
                           CONFIG['hit_velocity_threshold'])
