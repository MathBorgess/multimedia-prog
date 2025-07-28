import numpy as np
import wave
import os


def write_wav_file(filename, data, sample_rate=22050):
    """Write numpy array data to a WAV file."""
    # Ensure data is 16-bit integers
    data = np.clip(data, -32767, 32767).astype(np.int16)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())


def generate_success_sound():
    """Generate a success sound."""
    sample_rate = 22050
    duration = 0.5
    frames = int(duration * sample_rate)

    t = np.linspace(0, duration, frames)
    # Ascending chord
    freq1 = 523  # C5
    freq2 = 659  # E5

    wave = (np.sin(2 * np.pi * freq1 * t) +
            np.sin(2 * np.pi * freq2 * t)) * 0.3

    # Envelope
    envelope = np.exp(-2 * t)
    wave *= envelope

    # Convert to 16-bit
    wave = (wave * 32767).astype(np.int16)

    return wave, sample_rate


def generate_error_sound():
    """Generate an error sound."""
    sample_rate = 22050
    duration = 0.3
    frames = int(duration * sample_rate)

    t = np.linspace(0, duration, frames)
    # Low harsh sound
    freq = 200

    wave = np.sin(2 * np.pi * freq * t) * 0.4
    # Add some buzzing
    buzz = np.sin(2 * np.pi * freq * 3 * t) * 0.1
    wave += buzz

    # Quick decay
    envelope = np.exp(-4 * t)
    wave *= envelope

    # Convert to 16-bit
    wave = (wave * 32767).astype(np.int16)

    return wave, sample_rate


def generate_sequence_sound():
    """Generate a sequence indicator sound."""
    sample_rate = 22050
    duration = 0.15
    frames = int(duration * sample_rate)

    t = np.linspace(0, duration, frames)
    # Single clear tone
    freq = 880  # A5

    wave = np.sin(2 * np.pi * freq * t) * 0.3

    # Bell-like envelope
    envelope = np.exp(-3 * t)
    wave *= envelope

    # Convert to 16-bit
    wave = (wave * 32767).astype(np.int16)

    return wave, sample_rate


def main():
    """Generate all sound files."""
    sounds_dir = "sounds"
    if not os.path.exists(sounds_dir):
        os.makedirs(sounds_dir)

    print("Generating game sounds...")

    # Generate success sound
    success_data, sr = generate_success_sound()
    write_wav_file(os.path.join(sounds_dir, "success.wav"), success_data, sr)

    # Generate error sound
    error_data, sr = generate_error_sound()
    write_wav_file(os.path.join(sounds_dir, "error.wav"), error_data, sr)

    # Generate sequence sound
    sequence_data, sr = generate_sequence_sound()
    write_wav_file(os.path.join(sounds_dir, "sequence.wav"), sequence_data, sr)

    print("Game sounds generated successfully!")
    print("Generated files:")
    print("- sounds/success.wav")
    print("- sounds/error.wav")
    print("- sounds/sequence.wav")


if __name__ == "__main__":
    main()
