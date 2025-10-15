#!/usr/bin/env python3
"""
Generate podcast-style audio from SSML file, preserving timing information.

This script parses SSML <break> tags and inserts actual silence into the audio.
"""

import os
import sys
import re
import argparse
import torch
import torchaudio
from pathlib import Path


def parse_inline_emotions(text):
    """
    Parse text with inline emotion markers like (angry), (excited), etc.

    Returns:
        segments: List of tuples (text, emotion_description)

    Example:
        "Hello there! (excited) How are you? (calm)"
        -> [("Hello there!", "excited"), ("How are you?", "calm")]
    """
    segments = []

    # Split by emotion markers in parentheses
    parts = re.split(r'\s*\(([^)]+)\)\s*', text)

    current_emotion = None
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # This is text content
            if part.strip():
                segments.append((part.strip(), current_emotion))
        else:
            # This is an emotion marker - apply to next segment
            current_emotion = part.strip()

    return segments


def parse_ssml_with_breaks(input_file, use_inline_emotions=False):
    """
    Parse SSML or plain text file and extract text segments with break information.

    For SSML: Parses speaker markers and break tags
    For plain text: Splits by paragraphs, alternates speakers
    With inline_emotions: Parses (emotion) markers in text

    Returns:
        segments: List of tuples (speaker, text, emotion_text, break_after_ms)
    """
    with open(input_file, 'r') as f:
        content = f.read()

    # Check if it's SSML (has ## speaker markers or <break> tags)
    is_ssml = '## ' in content or '<break' in content

    segments = []

    if is_ssml:
        # Parse SSML format
        current_speaker = None
        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            # Check for speaker marker
            if line.startswith('## '):
                current_speaker = line[3:].strip().lower()
                continue

            if not line or not current_speaker:
                continue

            # Process line with SSML tags
            # Extract all break tags and text between them
            parts = re.split(r'(<break[^>]*?/>)', line)

            current_text = ""

            for part in parts:
                if part.startswith('<break'):
                    # Extract break duration
                    match = re.search(r'time="(\d+)ms"', part)
                    if match:
                        break_ms = int(match.group(1))
                    else:
                        break_ms = 0

                    # Add segment with accumulated text and break
                    if current_text.strip():
                        # Check for inline emotions
                        if use_inline_emotions and re.search(r'\([^)]+\)', current_text):
                            emotion_segments = parse_inline_emotions(current_text)
                            for text_part, emotion in emotion_segments:
                                segments.append((current_speaker, text_part, emotion, 0))
                            # Add break to last segment
                            if segments:
                                last = segments[-1]
                                segments[-1] = (last[0], last[1], last[2], break_ms)
                        else:
                            segments.append((current_speaker, current_text.strip(), None, break_ms))
                        current_text = ""
                else:
                    # Remove other SSML tags but keep content
                    part = re.sub(r'<emphasis[^>]*?>(.*?)</emphasis>', r'\1', part)
                    part = re.sub(r'<prosody[^>]*?>(.*?)</prosody>', r'\1', part)
                    current_text += part

            # Add remaining text without break
            if current_text.strip():
                # Check for inline emotions
                if use_inline_emotions and re.search(r'\([^)]+\)', current_text):
                    emotion_segments = parse_inline_emotions(current_text)
                    for text_part, emotion in emotion_segments:
                        segments.append((current_speaker, text_part, emotion, 0))
                else:
                    segments.append((current_speaker, current_text.strip(), None, 0))

    else:
        # Parse plain text format (alternating paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            # Alternate between speaker1 and speaker2
            speaker = 'speaker1' if i % 2 == 0 else 'speaker2'

            # Check for inline emotions
            if use_inline_emotions and re.search(r'\([^)]+\)', paragraph):
                emotion_segments = parse_inline_emotions(paragraph)
                for text_part, emotion in emotion_segments:
                    segments.append((speaker, text_part, emotion, 200))
            else:
                # Add 500ms break after each paragraph
                segments.append((speaker, paragraph, None, 500))

    return segments, is_ssml


def create_silence(duration_ms, sample_rate=22050):
    """Create silence tensor."""
    duration_samples = int(duration_ms * sample_rate / 1000)
    return torch.zeros(1, duration_samples, dtype=torch.int16)


def main():
    parser = argparse.ArgumentParser(
        description="Generate podcast audio from SSML with timing control"
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file (SSML or plain text)")
    parser.add_argument("--voice1", type=str, required=True,
                        help="Path to first voice sample (wav)")
    parser.add_argument("--voice2", type=str, required=True,
                        help="Path to second voice sample (wav)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml",
                        help="Config file path")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="Model directory")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 inference (default: True)")
    parser.add_argument("--speaker1", type=str, default="sarah",
                        help="First speaker name (default: sarah)")
    parser.add_argument("--speaker2", type=str, default="david",
                        help="Second speaker name (default: david)")
    parser.add_argument("--use_emotion", action="store_true", default=True,
                        help="Enable automatic emotion detection from text (default: True)")
    parser.add_argument("--inline_emotion", action="store_true", default=False,
                        help="Parse inline emotion markers like (angry), (excited) in text")
    parser.add_argument("--emo_alpha", type=float, default=0.6,
                        help="Emotion strength when using emotions (0.0-1.0, default: 0.6)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.voice1):
        print(f"❌ Voice1 file not found: {args.voice1}")
        sys.exit(1)

    if not os.path.exists(args.voice2):
        print(f"❌ Voice2 file not found: {args.voice2}")
        sys.exit(1)

    print("="*80)
    print("IndexTTS2 Podcast Generator with SSML Timing Support")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Voice 1 ({args.speaker1}): {args.voice1}")
    print(f"Voice 2 ({args.speaker2}): {args.voice2}")
    print(f"Output: {args.output}")
    print("="*80)

    # Parse input file
    print("\n📄 Parsing input file...")
    segments, is_ssml = parse_ssml_with_breaks(args.input, use_inline_emotions=args.inline_emotion)
    file_type = "SSML" if is_ssml else "plain text"
    emotion_mode = "inline emotions" if args.inline_emotion else "auto-detect" if args.use_emotion else "neutral"
    print(f"✓ Detected {file_type} format")
    print(f"✓ Emotion mode: {emotion_mode}")
    print(f"✓ Parsed {len(segments)} text segments with timing information")

    # Count segments per speaker and emotions
    speaker_counts = {}
    emotion_counts = {}
    for speaker, _, emotion, _ in segments:
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count} segments")

    if emotion_counts:
        print(f"\nDetected emotions:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} segments")

    # Set up environment for uv
    print("\n⚙️  Setting up environment...")
    venv_path = "/home/ubuntu/text-to-audio/index-tts/.venv"
    if os.path.exists(venv_path):
        os.environ["VIRTUAL_ENV"] = venv_path
        os.environ["PATH"] = f"{venv_path}/bin:{os.environ.get('PATH', '')}"

    # Import IndexTTS2
    try:
        from indextts.infer_v2 import IndexTTS2
    except ImportError as e:
        print(f"❌ Failed to import IndexTTS2: {e}")
        print("\nTry running:")
        print("  cd /home/ubuntu/text-to-audio/index-tts")
        print("  uv sync")
        sys.exit(1)

    # Initialize model
    print("\n📦 Loading IndexTTS2 model...")
    try:
        tts = IndexTTS2(
            cfg_path=args.config,
            model_dir=args.model_dir,
            use_fp16=args.fp16
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Voice mapping
    voice_map = {
        args.speaker1: args.voice1,
        args.speaker2: args.voice2,
        'speaker1': args.voice1,  # For plain text alternating mode
        'speaker2': args.voice2
    }

    # Generate audio for each segment
    print(f"\n🎙️  Synthesizing {len(segments)} segments...")

    import tempfile
    temp_files = []
    sample_rate = 22050  # IndexTTS2 default

    for i, (speaker, text, emotion_text, break_after_ms) in enumerate(segments):
        voice = voice_map.get(speaker, args.voice1)

        print(f"\n  Segment {i+1}/{len(segments)} ({speaker}):")
        print(f"    Text: {text[:60]}..." if len(text) > 60 else f"    Text: {text}")
        if emotion_text:
            print(f"    Emotion: {emotion_text}")
        print(f"    Break after: {break_after_ms}ms")

        # Generate audio for this segment
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_file.name)

        try:
            # Determine emotion mode
            if emotion_text:
                # Use explicit inline emotion
                tts.infer(
                    spk_audio_prompt=voice,
                    text=text,
                    output_path=temp_file.name,
                    use_emo_text=True,
                    emo_text=emotion_text,
                    emo_alpha=args.emo_alpha,
                    verbose=False
                )
            elif args.use_emotion:
                # Auto-detect emotion from text
                tts.infer(
                    spk_audio_prompt=voice,
                    text=text,
                    output_path=temp_file.name,
                    use_emo_text=True,
                    emo_alpha=args.emo_alpha,
                    verbose=False
                )
            else:
                # Neutral (no emotion)
                tts.infer(
                    spk_audio_prompt=voice,
                    text=text,
                    output_path=temp_file.name,
                    verbose=False
                )
        except Exception as e:
            print(f"    ❌ Failed to synthesize: {e}")
            # Clean up and exit
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
            sys.exit(1)

        print(f"    ✓ Generated")

    # Concatenate all segments with breaks
    print(f"\n🔗 Concatenating {len(temp_files)} audio segments with timing...")

    try:
        audio_segments = []

        for i, (temp_file, (speaker, text, emotion, break_after_ms)) in enumerate(zip(temp_files, segments)):
            # Load audio segment
            waveform, sr = torchaudio.load(temp_file)
            audio_segments.append(waveform)

            # Add silence if break is specified
            if break_after_ms > 0:
                silence = create_silence(break_after_ms, sample_rate=sr)
                audio_segments.append(silence)
                print(f"  Segment {i+1}: added {break_after_ms}ms silence")

        # Concatenate all
        final_audio = torch.cat(audio_segments, dim=1)

        # Save
        torchaudio.save(args.output, final_audio, sample_rate)

        # Clean up temp files
        for temp_file in temp_files:
            os.remove(temp_file)

        print(f"✓ Audio concatenated successfully")

    except Exception as e:
        print(f"❌ Failed to concatenate audio: {e}")
        print("Temp files saved:", temp_files)
        sys.exit(1)

    # Success
    print("\n" + "="*80)
    print("✓ SUCCESS!")
    print("="*80)
    print(f"Audio saved to: {args.output}")

    # Show file size and duration
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

        # Calculate duration
        waveform, sr = torchaudio.load(args.output)
        duration_sec = waveform.shape[1] / sr
        print(f"Duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

    print("\nYou can play it with:")
    print(f"  ffplay {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
