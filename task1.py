from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

from audio_processing_error import AudioProcessingError


def fade_pydub(
    file_path: str | Path,
    start_ms: int = 10000,
    end_ms: int = 15000,
    fade_in_ms: int = 3000,
    fade_out_ms: int = 3000,
    repeats: int = 3,
) -> np.ndarray:
    """
    Apply fade-in and fade-out effects to an audio file using pydub.

    Args:
        file_path (str | Path): Path to the audio file.
        start_ms (int): Start time in milliseconds. Defaults to 10000.
        end_ms (int): End time in milliseconds. Defaults to 15000.
        fade_in_ms (int): Duration of fade-in in milliseconds. Defaults to 3000.
        fade_out_ms (int): Duration of fade-out in milliseconds. Defaults to 3000.
        repeats (int): Number of times to repeat the audio segment. Defaults to 3.

    Returns:
        np.ndarray: Processed audio samples as a numpy array.

    Raises:
        AudioProcessingError: If audio file loading or processing fails.
    """
    try:
        audio = AudioSegment.from_file(str(file_path))
        audio_segment = audio[start_ms:end_ms]
        audio_segment = audio_segment * repeats
        audio_segment = audio_segment.fade_in(fade_in_ms).fade_out(fade_out_ms)
        return np.array(audio_segment.get_array_of_samples())
    except Exception as e:
        raise AudioProcessingError(f'Error processing audio with pydub: {str(e)}')


def fade_librosa(
    file_path: str | Path,
    start_ms: int = 10000,
    end_ms: int = 15000,
    fade_in_ms: int = 3000,
    fade_out_ms: int = 3000,
    repeats: int = 3,
) -> np.ndarray:
    """
    Apply fade-in and fade-out effects to an audio file using librosa.

    Args:
        file_path (str | Path): Path to the audio file.
        start_ms (int): Start time in milliseconds. Defaults to 10000.
        end_ms (int): End time in milliseconds. Defaults to 15000.
        fade_in_ms (int): Duration of fade-in in milliseconds. Defaults to 3000.
        fade_out_ms (int): Duration of fade-out in milliseconds. Defaults to 3000.
        repeats (int): Number of times to repeat the audio segment. Defaults to 3.

    Returns:
        np.ndarray: Processed audio samples as a numpy array.

    Raises:
        AudioProcessingError: If audio file loading or processing fails.
    """
    try:
        audio, sample_rate = librosa.load(str(file_path), sr=None, mono=False)

        # Convert milliseconds to samples
        start_samples = int(start_ms / 1000 * sample_rate)
        end_samples = int(end_ms / 1000 * sample_rate)

        # Calculate fade-in and fade-out samples
        fade_in_samples = int(fade_in_ms / 1000 * sample_rate)
        fade_out_samples = int(fade_out_ms / 1000 * sample_rate)

        # Mono audio (samples,) ; Stereo audio (channels, samples)
        audio_segment = audio[..., start_samples:end_samples]

        # If audio is stereo, repeat the segment for both channels
        audio_segment = (
            np.tile(audio_segment, (1, repeats))
            if audio_segment.ndim == 2
            else np.tile(audio_segment, repeats)
        )

        if audio_segment.ndim == 2:
            for ch in range(audio_segment.shape[0]):
                # Apply fade-in and fade-out to each channel
                audio_segment[ch, :fade_in_samples] *= np.linspace(
                    0, 1, fade_in_samples
                )
                audio_segment[ch, -fade_out_samples:] *= np.linspace(
                    1, 0, fade_out_samples
                )
            # Flatten the stereo audio to mono
            audio_segment = audio_segment.T.flatten()
        else:
            audio_segment[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
            audio_segment[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)

        # 16-bit PCM
        audio_segment = np.clip(audio_segment, -1.0, 1.0)
        audio_segment = (audio_segment * 32767).astype(np.int16)
        return audio_segment
    except Exception as e:
        raise AudioProcessingError(f'Error processing audio with librosa: {str(e)}')


def plot_comparison(
    file_path: str,
    original: np.ndarray,
    pydub_processed: np.ndarray,
    librosa_processed: np.ndarray
) -> None:
    """
    Plot comparison of original and processed audio signals.

    Args:
        original (np.ndarray): Original audio samples.
        pydub_processed (np.ndarray): Audio processed with pydub.
        librosa_processed (np.ndarray): Audio processed with librosa.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 4))
    fig.suptitle(file_path)
    fig.subplots_adjust(hspace=0.5)

    axes[0].plot(original)
    axes[0].set_title('Original Audio')
    axes[0].set_xlabel('Sample (Hz)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(pydub_processed)
    axes[1].set_title('Pydub Faded Audio')
    axes[1].set_xlabel('Sample (Hz)')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(librosa_processed)
    axes[2].set_title('Librosa Faded Audio')
    axes[2].set_xlabel('Sample (Hz)')
    axes[2].set_ylabel('Amplitude')


def main() -> None:
    """
    Main function to read audio files, process with fade_in and fade_out filters,
    and plot them for comparison.
    """
    try:
        # --- Configuration ---
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'

        audio_files = [
            Path('samples/sample1.wav'),
            Path('samples/sample2.wav'),
        ]
        # --- End Configuration ---

        for path in audio_files:
            if not path.exists():
                print(f'Audio file not found: {path}')
                continue

            print(f'Processing "{path}"')

            # Processing audio sample
            original_audio = AudioSegment.from_file(path)
            repeated_segment = original_audio[10000:15000] * 3
            repeated_array = np.array(repeated_segment.get_array_of_samples())

            # Process with pydub and librosa
            pydub_array = fade_pydub(path)
            librosa_array = fade_librosa(path)

            # Show plot comparison
            plot_comparison(path, repeated_array, pydub_array, librosa_array)

        plt.show()

    except AudioProcessingError as e:
        print(f'Audio processing error: {str(e)}')
    except Exception as e:
        print(f'Unexpected error: {str(e)}')


if __name__ == '__main__':
    main()
