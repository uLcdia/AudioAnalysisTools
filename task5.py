from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from audio_processing_error import AudioProcessingError


def load_pretrained_model(
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
) -> EncoderClassifier:
    """
    Load a pretrained SpeechBrain model for speaker verification.

    Args:
        source (str): Source of the pretrained model. Defaults to "speechbrain/spkrec-ecapa-voxceleb".

    Returns:
        EncoderClassifier: Loaded speaker verification model.

    Raises:
        AudioProcessingError: If loading the model fails.
    """
    # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    try:
        classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        return classifier
    except Exception as e:
        raise AudioProcessingError(f'Error loading pretrained model: {str(e)}')


def extract_speaker_embedding(
    model: EncoderClassifier,
    file_path: str | Path
) -> np.ndarray:
    """
    Extract speaker embedding from an audio file.

    Args:
        model (EncoderClassifier): Speaker verification model.
        file_path (str | Path): Path to the audio file.

    Returns:
        np.ndarray: Speaker embedding.

    Raises:
        AudioProcessingError: If embedding extraction fails.
    """
    try:
        waveform = model.load_audio(str(file_path))
        embedding = model.encode_batch(waveform).squeeze().detach().cpu().numpy()
        return embedding
    except Exception as e:
        raise AudioProcessingError(f'Error extracting speaker embedding: {str(e)}')


def compare_speakers(
    model: EncoderClassifier,
    file_path1: str | Path,
    file_path2: str | Path
) -> Tuple[float, bool]:
    """
    Compare two speakers and determine if they are the same person.

    Args:
        model (EncoderClassifier): Speaker verification model.
        file_path1 (str | Path): Path to the first audio file.
        file_path2 (str | Path): Path to the second audio file.

    Returns:
        Tuple[float, bool]: Similarity score (0-1) and boolean indicating if same speaker (True if same).

    Raises:
        AudioProcessingError: If speaker comparison fails.
    """
    try:
        # Extract embeddings
        embedding1 = extract_speaker_embedding(model, file_path1)
        embedding2 = extract_speaker_embedding(model, file_path2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        # Determine if same speaker
        threshold = 0.75
        is_same_speaker = similarity > threshold

        return similarity, is_same_speaker
    except Exception as e:
        raise AudioProcessingError(f'Error comparing speakers: {str(e)}')


def plot_comparison_results(
    file_paths: List[Path],
    results: List[Tuple[str, str, float, bool]]
) -> None:
    """
    Plot speaker verification results.

    Args:
        file_paths (List[Path]): List of audio file paths.
        results (List[Tuple[str, str, float, bool]]): List of comparison results.
    """
    plt.figure(figsize=(10, 6))

    # Matrix for comparison scores
    n = len(file_paths)
    scores_matrix = np.zeros((n, n))

    # Extract file names for labels
    labels = [path.stem for path in file_paths]

    # Fill the score matrix based on results
    for result in results:
        i = labels.index(Path(result[0]).stem)
        j = labels.index(Path(result[1]).stem)
        scores_matrix[i, j] = result[2]
        scores_matrix[j, i] = result[2]  # Make the matrix symmetric

    # Set diagonal to 1.0 (self-comparison)
    for i in range(n):
        scores_matrix[i, i] = 1.0

    # Plot heatmap
    plt.imshow(scores_matrix)
    plt.colorbar(label='Similarity Score')
    plt.xticks(range(n), labels, rotation=45)
    plt.yticks(range(n), labels)
    plt.title('Speaker Verification Similarity Scores')

    # Annotate the cells with scores
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{scores_matrix[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if scores_matrix[i, j] < 0.75 else 'black')

    plt.tight_layout()


def main() -> None:
    """
    Main function to perform speaker verification on audio samples.
    """
    try:
        # --- Configuration ---
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'

        audio_files = [
            Path('samples/sample5.wav'),
            Path('samples/sample6.wav'),
            Path('samples/sample.wav'),
        ]
        # --- End Configuration ---

        # Check if files exist
        for path in audio_files:
            if not path.exists():
                print(f'Audio file not found: {path}')
                del audio_files[audio_files.index(path)]

        # Load speaker verification model
        print("Loading pretrained speaker verification model...")
        model = load_pretrained_model()

        # Compare all pairs of audio files
        comparison_results = []
        for i in range(len(audio_files)):
            for j in range(i+1, len(audio_files)):
                print(f"\nComparing {audio_files[i].stem} and {audio_files[j].stem}:")
                similarity, is_same_speaker = compare_speakers(model, audio_files[i], audio_files[j])
                print(f"Similarity score: {similarity:.4f}")
                print(f"Same speaker: {is_same_speaker}")
                comparison_results.append((str(audio_files[i]), str(audio_files[j]), similarity, is_same_speaker))

        # Plot comparison results
        plot_comparison_results(audio_files, comparison_results)
        plt.show()

    except AudioProcessingError as e:
        print(f'Audio processing error: {str(e)}')
    except Exception as e:
        print(f'Unexpected error: {str(e)}')


if __name__ == '__main__':
    main()
