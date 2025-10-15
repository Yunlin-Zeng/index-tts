#!/usr/bin/env python3
"""
Simple CLI for IndexTTS2
"""
import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 Simple CLI")
    parser.add_argument("text", type=str, help="Text to be synthesized")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to the voice reference audio file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output audio file path")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model directory path")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for inference")

    args = parser.parse_args()

    # Validate inputs
    if not args.text.strip():
        print("ERROR: Text cannot be empty")
        sys.exit(1)

    if not os.path.exists(args.voice):
        print(f"ERROR: Voice file {args.voice} not found")
        sys.exit(1)

    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory {args.model_dir} not found")
        sys.exit(1)

    # Set output path if not provided
    if args.output is None:
        args.output = f"output_{int(time.time())}.wav"

    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Output: {args.output}")
    print("Loading IndexTTS2...")

    # Import and initialize
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.fp16,
    )

    print("Generating audio...")

    # Generate
    output_path = tts.infer(
        spk_audio_prompt=args.voice,
        text=args.text,
        output_path=args.output,
        verbose=True
    )

    print(f"Audio generated successfully: {output_path}")

if __name__ == "__main__":
    main()