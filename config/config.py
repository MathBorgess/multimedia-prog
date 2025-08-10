CONFIG = {
    # Game settings
    'sequence_start_length': 2,
    'border_flash_duration': 1.0,  # seconds (slower)
    'color_pause_duration': 0.8,   # seconds between colors (slower)
    'success_pause_duration': 1.5,  # seconds after success
    # seconds after failure (deprecated, using countdown now)
    'countdown_duration': 5,        # seconds countdown before restart

    # Visual settings
    # transparency for areas (0.0 = transparent, 1.0 = opaque)
    'area_transparency': 0.5,      # Slightly more transparent for better feedback
    'border_thickness': 25,        # border thickness for sequence display

    # Error message settings
    'error_display_duration': 2.5,  # seconds to show error message
    'error_flash_duration': 2.0,    # seconds for error flash effect

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

    # Game colors (BGR: OpenCV format) - semi-transparent versions will be calculated
    'game_colors': [
        (0, 0, 255),    
        (0, 255, 0),    
        (255, 0, 0),    
        (0, 255, 255),  
        (255, 0, 255),  
        (255, 255, 0),  
        (0, 165, 255),  
        (128, 0, 128),  
        (203, 192, 255) 
    ]
}
