#!/usr/bin/env python3
"""
Generate podcast-style audio from text file using IndexTTS2 with multiple speakers.

Usage:
    python generate_podcast.py \
        --text story.txt \
        --voice1 examples/voice_01.wav \
        --voice2 examples/voice_02.wav \
        --output podcast.wav
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate podcast audio from text using two voices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple story narration (single voice)
  python generate_podcast.py \\
    --text story.txt \\
    --voice1 examples/voice_01.wav \\
    --output story_audio.wav

  # Podcast with two speakers (splits text by paragraphs/turns)
  python generate_podcast.py \\
    --text podcast_script.txt \\
    --voice1 examples/voice_01.wav \\
    --voice2 examples/voice_02.wav \\
    --output podcast.wav \\
    --alternating
        """
    )

    parser.add_argument("--text", type=str, required=True,
                        help="Path to text file to synthesize")
    parser.add_argument("--voice1", type=str, required=True,
                        help="Path to first voice sample (wav)")
    parser.add_argument("--voice2", type=str, default=None,
                        help="Path to second voice sample (wav) for multi-speaker")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--alternating", action="store_true",
                        help="Alternate between voice1 and voice2 for each paragraph")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml",
                        help="Config file path")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="Model directory")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 inference (default: True)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.text):
        print(f"❌ Text file not found: {args.text}")
        sys.exit(1)

    if not os.path.exists(args.voice1):
        print(f"❌ Voice1 file not found: {args.voice1}")
        sys.exit(1)

    if args.voice2 and not os.path.exists(args.voice2):
        print(f"❌ Voice2 file not found: {args.voice2}")
        sys.exit(1)

    # Read text
    with open(args.text, 'r') as f:
        text = f.read().strip()

    if not text:
        print("❌ Text file is empty")
        sys.exit(1)

    print("="*80)
    print("IndexTTS2 Podcast Generator")
    print("="*80)
    print(f"Text file: {args.text}")
    print(f"Text length: {len(text)} chars, {len(text.split())} words")
    print(f"Voice 1: {args.voice1}")
    if args.voice2:
        print(f"Voice 2: {args.voice2}")
        print(f"Mode: Multi-speaker {'(alternating)' if args.alternating else ''}")
    else:
        print("Mode: Single narrator")
    print(f"Output: {args.output}")
    print("="*80)

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

    # Generate audio
    print(f"\n🎙️  Synthesizing audio...")

    if args.voice2 and args.alternating:
        # Multi-speaker alternating mode
        print("Mode: Alternating between two voices per paragraph")

        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]  # Single paragraph fallback

        print(f"Found {len(paragraphs)} paragraphs")

        # Generate each paragraph with alternating voices
        import tempfile
        temp_files = []

        for i, para in enumerate(paragraphs):
            voice = args.voice1 if i % 2 == 0 else args.voice2
            voice_name = "Voice 1" if i % 2 == 0 else "Voice 2"

            print(f"  Paragraph {i+1}/{len(paragraphs)} ({voice_name}): {len(para.split())} words")

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_files.append(temp_file.name)

            tts.infer(
                spk_audio_prompt=voice,
                text=para,
                output_path=temp_file.name,
                verbose=False
            )

        # Concatenate audio files
        print(f"\n🔗 Concatenating {len(temp_files)} audio segments...")
        try:
            import torchaudio
            import torch

            audio_segments = []
            sr = None
            for temp_file in temp_files:
                waveform, sample_rate = torchaudio.load(temp_file)
                audio_segments.append(waveform)
                if sr is None:
                    sr = sample_rate

            # Concatenate
            final_audio = torch.cat(audio_segments, dim=1)
            torchaudio.save(args.output, final_audio, sr)

            # Clean up temp files
            for temp_file in temp_files:
                os.remove(temp_file)

            print(f"✓ Audio concatenated successfully")

        except Exception as e:
            print(f"❌ Failed to concatenate audio: {e}")
            print("Temp files saved:", temp_files)
            sys.exit(1)

    else:
        # Single voice mode
        print("Mode: Single narrator")
        try:
            output_path = tts.infer(
                spk_audio_prompt=args.voice1,
                text=text,
                output_path=args.output,
                verbose=True
            )
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            sys.exit(1)

    # Success
    print("\n" + "="*80)
    print("✓ SUCCESS!")
    print("="*80)
    print(f"Audio saved to: {args.output}")

    # Show file size
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

    print("\nYou can play it with:")
    print(f"  ffplay {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
