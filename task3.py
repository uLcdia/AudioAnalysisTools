from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import logfbank, mfcc
from scipy.io import wavfile

from audio_processing_error import AudioProcessingError


def plot_features(
    features: np.ndarray,
    title: str,
    ylabel: str,
    ax: plt.Axes = None
) -> None:
    """
    Plots the calculated audio features (MFCC or FBanks).

    Args:
        features (np.ndarray): The feature matrix (time_frames x num_features).
        title (str): The title for the plot.
        ylabel (str): The label for the y-axis.
        ax (plt.Axes, optional): Matplotlib axes object to plot on. If None, a new figure is created.
    """
    if ax is None:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()

    # Transpose features so time is on the x-axis
    im = ax.imshow(features.T, aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='Amplitude')
    ax.set_title(title)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel(ylabel)

    # Set y-axis ticks to show all coefficients
    num_features = features.shape[1]
    ax.set_yticks(np.arange(0, num_features, max(1, num_features//10)))


def main() -> None:
    """
    Main function to read audio files, compute MFCCs and FBanks, and plot them for comparison.
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
            sample_rate, signal = wavfile.read(path)
            print(f'Processing "{path}": Sample Rate = {sample_rate}, Signal Shape = {signal.shape}')

            # Ensure signal is mono for feature extraction if it's stereo
            if signal.ndim > 1:
                print('Stereo audio detected, converting to mono by averaging channels.')
                signal = np.mean(signal, axis=1)

            # --- Feature Extraction ---
            # Calculate MFCCs
            mfcc_features = mfcc(signal, sample_rate, nfft=1024)
            print(f'MFCC Features Shape for {path.name}: {mfcc_features.shape}')

            # Calculate FBanks
            fbank_features = logfbank(signal, sample_rate, nfft=1024)
            print(f'Filter Bank Features Shape for {path.name}: {fbank_features.shape}')

            # --- Visualization ---
            # Create subplots for MFCC and Filter Banks
            fig, axes = plt.subplots(2, 1, figsize=(10, 4))
            fig.suptitle(f'Audio Features for {path.name}')
            fig.subplots_adjust(hspace=0.5)

            # Plot MFCCs
            plot_features(mfcc_features, 'MFCC Features', 'Cepstral Coefficient Index', axes[0])

            # Plot Filter Banks
            plot_features(fbank_features, 'Filter Banks', 'Filter Bank Index', axes[1])

        plt.show()

    except AudioProcessingError as e:
        print(f'Audio processing error: {str(e)}')
    except FileNotFoundError as e:
        print(f'Audio file not found: {str(e)}')
    except Exception as e:
        print(f'Unexpected error: {str(e)}')


if __name__ == '__main__':
    main()