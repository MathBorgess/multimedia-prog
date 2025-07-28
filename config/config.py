CONFIG = {
    # Game settings
    'sequence_start_length': 2,
    'border_flash_duration': 1.0,  # seconds (slower)
    'color_pause_duration': 0.8,   # seconds between colors (slower)
    'success_pause_duration': 1.5,  # seconds after success
    'failure_pause_duration': 2.5,  # seconds after failure

    # Visual settings
    # transparency for areas (0.0 = transparent, 1.0 = opaque)
    'area_transparency': 0.6,
    'border_thickness': 25,        # border thickness for sequence display

    # Sound settings
    'enable_sounds': True,
    'sound_files': {
        'success': 'sounds/success.wav',
        'error': 'sounds/error.wav',
        'sequence': 'sounds/sequence.wav'
    },

    # Hand detection thresholds
    'hand_touch_threshold': 50,    # pixels distance to consider "touching"
    'hand_detection_confidence': 0.7,

    # Camera settings
    'camera_index': 0,
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 30,

    # MediaPipe Hands configuration
    'hands_config': {
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.7
    },

    # Performance settings
    'fps': 60,

    # Game colors (RGB) - semi-transparent versions will be calculated
    'game_colors': [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203)  # Pink
    ]
}
