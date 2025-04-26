import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import whisper
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from scipy.io import wavfile
from TTS.api import TTS

from audio_processing_error import AudioProcessingError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()


def recognize_speech(
    file_path: str | Path,
    model_name: str = 'base',
    language: str = 'zh'
) -> str:
    """
    Recognize speech from an audio file using OpenAI-Whisper.

    Args:
        file_path (str | Path): Path to the audio file.
        model_name (str): Whisper model to use. Defaults to 'base'.
        language (str): Language code for transcription. Defaults to 'zh'.

    Returns:
        str: Recognized text from the audio.

    Raises:
        AudioProcessingError: If speech recognition fails.
    """
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(str(file_path), language=language)
        return result['text']
    except Exception as e:
        raise AudioProcessingError(f'Error recognizing speech with Whisper: {str(e)}')


def split_sentences(
    text: str,
    api_key: str,
    provider: str = 'nebius',
    model_name: str = 'Qwen/Qwen2.5-32B-Instruct'
) -> list:
    """
    Split Chinese text into sentences using Hugging Face's Inference API.

    Args:
        text (str): Text to split into sentences.
        api_key (str): API key for Hugging Face Inference API.
        provider (str): Provider for Hugging Face Inference API. Defaults to 'nebius'.
        model_name (str): Model to use for sentence splitting. Defaults to 'Qwen/Qwen2.5-32B-Instruct'.

    Returns:
        list: List of individual sentences.
    """
    try:
        if len(text) < 10:
            return [text]

        instruction = f"""
        你是一个专业的中文文本处理助手.你的任务是将一段没有标点符号的中文文本分割成独立的句子, 并 **仅以 JSON 格式返回结果**.输入的文本是一段没有标点符号的中文字符串, 可能会包含多个句子拼接在一起.请根据语义和语法逻辑, 将其拆分成一个句子列表.

        **重要说明:**
        - 不要输出任何与 JSON 无关的文本、说明或额外内容.
        - 输出必须严格遵循 JSON 格式, 包含一个键为 'sentences' 的列表, 列表中每个元素是一个独立的句子字符串.
        - 不要在输出中添加任何前言、结尾或其他非 JSON 内容.
        - 不要添加额外的标点符号, 只需分割成独立的句子.
        - 即使原文不符合中文语义, 也请尽量根据中文语义进行分割.
        - 必须将文本拆分为多个短句, 不要返回完整文本.

        **示例 1 (符合中文语义):**
        **输入:**
        '根据我们的最新研究多任务学习的优势在于速度和显存然而精度往往不如单任务模型所以HanLP预训练了许多单任务模型并设计了优雅的流水线模式将其组装起来'
        **输出:**
        *输出开始*
        {{
        'sentences': [
            '根据我们的最新研究',
            '多任务学习的优势在于速度和显存',
            '然而精度往往不如单任务模型',
            '所以HanLP预训练了许多单任务模型',
            '并设计了优雅的流水线模式将其组装起来'
        ]
        }}
        *输出结束*

        **示例 2 (不符合中文语义):**
        **输入:**
        '我看一件影子很怪的反正小鸣发现了我周围心理解发发起了大雨下性节续约电容有727秒它会看见自己的道理在跳舞房间里的物品都在说话然后刚刚说你来接下来的性节续流放手时间已经做出了很久'
        **输出:**
        *输出开始*
        {{
        'sentences': [
            '我看一件影子很怪的',
            '反正小鸣发现了我周围',
            '心理解发发起了大雨下',
            '性节续约电容有727秒',
            '它会看见自己的道理在跳舞',
            '房间里的物品都在说话',
            '然后刚刚说你来',
            '接下来的性节续流放手时间已经做出了很久'
        ]
        }}
        *输出结束*

        **实际任务:**
        以下是需要处理的中文文本:
        {text}

        请将这段文本分割成独立的句子, 每个句子不要超过30个字, 并 **仅以 JSON 格式** 返回结果.
        """

        client = InferenceClient(provider=provider, api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': instruction}],
            max_tokens=512,
            temperature=0.5
        )

        content = completion.choices[0].message.content

        try:
            json_data = json.loads(content)
            sentences = json_data.get('sentences', [])

            # Use fallback segmentation if needed
            if not sentences or (len(sentences) == 1 and len(sentences[0]) > 30) or (len(sentences) < len(text) // 20):
                logging.warning('LLM segmentation insufficient, using fallback segmentation')
                return fallback_split_sentences(text)

            return sentences
        except json.JSONDecodeError:
            logging.warning('Failed to parse LLM response as JSON, using fallback segmentation')
            return fallback_split_sentences(text)

    except Exception as e:
        logging.warning(f'Error splitting sentences using LLM: {str(e)}')
        return fallback_split_sentences(text)


def fallback_split_sentences(text: str, max_length: int = 30) -> list:
    """
    Fallback method to split text into sentences when LLM-based splitting fails.
    Uses regex to split by common punctuation first, then enforces a maximum length per segment.
    If no punctuation is present, splits by max_length.

    Args:
        text (str): Text to split into sentences.
        max_length (int): Maximum length of each sentence. Defaults to 30.

    Returns:
        list: List of sentence segments.
    """
    # Sentence terminators for splitting
    terminators_pattern = r'[,，.。!！?？;；\n\s]+'

    # First pass: split by terminators using regex
    initial_sentences = re.split(terminators_pattern, text)
    sentences = [s.strip() for s in initial_sentences if s.strip()]

    # If no splits occurred and text exists, split the tangled text
    if not sentences and text.strip():
        sentences = [text]

    # Second pass: ensure no segment exceeds max_length
    result = []
    for sentence in sentences:
        if len(sentence) > max_length:
            # Split long segments into chunks of max_length
            for i in range(0, len(sentence), max_length):
                result.append(sentence[i:i + max_length])
        elif sentence: # Avoid adding empty strings
            result.append(sentence)

    # If failed to split, log the error
    if not result:
        if text.strip():
            logging.warning(f'Failed to split text: {text}')
        else:
            logging.warning(f'Failed to split text, text is empty or whitespace: {text}')
        return []

    logging.info(f'Successfully split text into {len(result)} segments')
    logging.debug(f'Segments: {result}')
    return result


def generate_speech(
    sentences: List[str],
    model_name: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
    language: str = 'zh-cn',
    speaker_wav: str = None,
    speaker_name: str = None,
    is_cloning: bool = False
) -> List[tuple]:
    """
    Synthesize speech from sentences using Coqui-TTS.

    Args:
        sentences (List[str]): List of sentences to synthesize.
        model_name (str): TTS model to use. Defaults to 'tts_models/multilingual/multi-dataset/xtts_v2'.
        language (str): Language code for synthesis. Defaults to 'zh-cn'.
        speaker_wav (str, optional): Path to speaker's WAV file for voice cloning. Required if is_cloning=True.
        speaker_name (str, optional): Name of built-in speaker to use. Used if is_cloning=False.
        is_cloning (bool): Whether to use voice cloning or built-in speaker. Defaults to False.

    Returns:
        List[tuple]: List of tuples (sample_rate, audio_array) for each synthesized sentence.
    """
    try:
        tts = TTS(model_name=model_name)
        audio_arrays = []
        operation_type = "Cloning" if is_cloning else "Synthesizing"

        if tts.is_multi_speaker:
            if is_cloning and not speaker_wav:
                raise AudioProcessingError('Speaker WAV file is required for cloning')
            if not is_cloning and not speaker_name:
                raise AudioProcessingError('Speaker name is required for synthesizing')

        if tts.is_multi_lingual:
            if not language:
                raise AudioProcessingError('Language is required for multi-lingual synthesis')

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            logging.info(f'{operation_type} sentence {i+1}/{len(sentences)}')
            logging.debug(f'Sentence: {sentence}')

            # Generate audio
            if is_cloning and speaker_wav:
                wav = tts.tts(text=sentence, language=language, speaker_wav=speaker_wav)
            else:
                wav = tts.tts(text=sentence, speaker=speaker_name, language=language)

            # Convert to numpy array
            wav = np.array(wav, dtype=np.float64) if not isinstance(wav, np.ndarray) else wav

            sample_rate = tts.synthesizer.output_sample_rate
            audio_arrays.append((sample_rate, wav))

        return audio_arrays
    except Exception as e:
        logging.warning(f'Error {operation_type.lower()} speech: {str(e)}')
        return []


def save_audio_arrays(
    audio_arrays: List[tuple],
    output_path: str | Path
) -> None:
    """
    Merge audio arrays into a single WAV file.

    Args:
        audio_arrays (List[tuple]): List of tuples (sample_rate, audio_array) to merge.
        output_path (str | Path): Path to save the merged WAV file.
    """
    try:
        if not audio_arrays:
            logging.warning(f'No audio arrays to merge for {output_path}')
            return

        valid_arrays = []
        sample_rate = None

        # Validate arrays and prepare for concatenation
        for item in audio_arrays:
            try:
                if not isinstance(item, tuple) or len(item) != 2:
                    logging.warning(f'Invalid audio array: {item}')
                    continue

                sr, audio = item

                # Set sample rate from first valid array
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logging.warning(f'Sample rate mismatch: {sr} != {sample_rate}')
                    continue

                # Ensure audio is a numpy array
                if not isinstance(audio, np.ndarray):
                    logging.debug(f'Converting audio to numpy array: {audio}')
                    audio = np.array(audio, dtype=np.float64)

                valid_arrays.append((sr, audio))
            except Exception:
                continue

        if not valid_arrays:
            logging.warning(f'No valid audio arrays for {output_path}')
            return

        # Get first array and concatenate the rest
        sample_rate, concatenated_audio = valid_arrays[0]

        for _, audio in valid_arrays[1:]:
            try:
                concatenated_audio = np.concatenate((concatenated_audio, audio))
            except Exception as e:
                logging.warning(f'Error concatenating audio: {str(e)}')

        # Convert to int16 for WAV file saving
        if concatenated_audio.dtype in (np.float64, np.float32):
            logging.debug(f'Converting audio to int16 from {concatenated_audio.dtype}')
            concatenated_audio = (concatenated_audio * 32767).astype(np.int16)

        # Save the merged audio
        output_path = str(Path(output_path))
        wavfile.write(output_path, sample_rate, concatenated_audio)
        logging.info(f'Saved audio to {output_path}')

    except Exception as e:
        logging.warning(f'Error merging audio arrays: {str(e)}')


def save_text_to_file(
    text: str,
    output_path: str | Path
) -> None:
    """
    Save text to a file.

    Args:
        text (str): Text to save.
        output_path (str | Path): Path to save the text file.

    Raises:
        AudioProcessingError: If saving text fails.
    """
    try:
        with open(str(output_path), 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        raise AudioProcessingError(f'Error saving text to file: {str(e)}')


def main() -> None:
    """
    Process audio files through speech recognition, sentence splitting, and speech synthesis.
    """
    try:
        # --- Configuration ---
        audio_files = [
            Path('samples/sample3.wav'),
            Path('samples/sample4.wav'),
        ]
        output_dir = Path('task4_output')
        output_dir.mkdir(exist_ok=True, parents=True)
        # Hugging Face API token stored in .env file
        hf_token = os.getenv('HF_TOKEN')
        # Full list of speakers: `tts.speakers()`
        speakers = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara',
                    'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper',
                    'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black']

        for path in audio_files:
            try:
                if not path.exists():
                    logging.warning(f'Audio file not found: {path}')
                    continue

                logging.info(f'Processing "{path}"')

                # Step 1: Speech Recognition
                try:
                    recognized_text = recognize_speech(path)
                    logging.info(f'Recognized Text: {recognized_text}')
                    text_output_path = output_dir / f'recognized_{path.stem}.txt'
                    save_text_to_file(recognized_text, text_output_path)
                except Exception as e:
                    logging.warning(f'Speech recognition failed: {str(e)}')
                    recognized_text = ""

                if not recognized_text:
                    logging.warning(f'No text recognized for {path}, skipping further processing')
                    continue

                # Step 2: Split sentences
                sentences = split_sentences(recognized_text, hf_token)
                logging.info(f'Split into {len(sentences)} sentences')

                if not sentences:
                    logging.warning(f'No sentences splited for {path}, skipping further processing')
                    continue

                # Step 3: Speech cloning and synthesis
                try:
                    # Clone speech
                    audio_arrays = generate_speech(
                        sentences,
                        speaker_wav=str(path),
                        language='zh-cn',
                        is_cloning=True
                    )
                    if audio_arrays:
                        merged_output_path = output_dir / f'cloned_{path.stem}.wav'
                        save_audio_arrays(audio_arrays, merged_output_path)

                    # Synthesize speech
                    audio_arrays = generate_speech(
                        sentences,
                        speaker_name=random.choice(speakers),
                        language='zh-cn',
                        is_cloning=False
                    )
                    if audio_arrays:
                        merged_output_path = output_dir / f'synthesized_{path.stem}.wav'
                        save_audio_arrays(audio_arrays, merged_output_path)
                except Exception as e:
                    logging.warning(f'Speech synthesis failed: {str(e)}')

                logging.info(f'Finished processing "{path}"')

            except Exception as e:
                logging.warning(f'Error processing {path}: {str(e)}')

    except AudioProcessingError as e:
        logging.error(f'Audio processing error: {str(e)}')
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')


if __name__ == '__main__':
    main()