from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from audio_processing_error import AudioProcessingError


def plot_spectrogram(
    file_path: str | Path,
    sample_rate: int,
    data: np.ndarray
) -> None:
    """
    Calculates and plots the spectrograms using scipy.signal.spectrogram and matplotlib.pyplot.specgram.

    Args:
        file_path (str | Path): Path to the audio file (used for title).
        sample_rate (int): The sample rate of the audio signal.
        data (np.ndarray): The audio time series.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    fig.suptitle(file_path)
    fig.subplots_adjust(hspace=0.5)

    # --- scipy.signal.spectrogram ---
    frequencies, times, spectrogram_data = signal.spectrogram(data, sample_rate)
    # Plot with pcolormesh and store the mappable object for colorbar
    mesh = axes[0].pcolormesh(times, frequencies, np.log(spectrogram_data + 1e-9))
    axes[0].set_title('scipy.signal.spectrogram')
    axes[0].set_xlabel('Time [sec]')
    axes[0].set_ylabel('Frequency [Hz]')
    plt.colorbar(mesh, ax=axes[0], label='Log Power')
    # --- End scipy.signal.spectrogram ---

    # --- matplotlib.pyplot.specgram ---
    spec, freqs, t, im = axes[1].specgram(data, Fs=sample_rate)
    axes[1].set_title('matplotlib.pyplot.specgram')
    axes[1].set_xlabel('Time [sec]')
    axes[1].set_ylabel('Frequency [Hz]')
    plt.colorbar(im, ax=axes[1], label='Log Power')
    # --- End matplotlib.pyplot.specgram ---

def main() -> None:
    """
    Main function to read audio files, calculate spectrograms, and plot them for comparison.
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

            # Read the audio file
            sample_rate, data = wavfile.read(path)
            print(f'Processing "{path}": Sample Rate = {sample_rate}, Data Shape = {data.shape}')

            # Ensure signal is mono for feature extraction if it's stereo
            if data.ndim > 1:
                print('Stereo audio detected, converting to mono by averaging channels.')
                data = np.mean(data, axis=1)

            # Show plot spectrum
            plot_spectrogram(path, sample_rate, data)

        plt.show()

    except AudioProcessingError as e:
        print(f'Audio processing error: {str(e)}')
    except Exception as e:
        print(f'Unexpected error: {str(e)}')

if __name__ == "__main__":
    main()