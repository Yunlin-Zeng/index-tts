#!/usr/bin/env python3
"""
Convert SSML podcast script to plain text with speaker labels.
Removes SSML tags but keeps speaker turns.
"""

import re
import sys
import argparse


def convert_ssml_to_text(ssml_file, output_file):
    """Convert SSML to plain text, preserving speaker structure."""

    with open(ssml_file, 'r') as f:
        content = f.read()

    # Remove SSML tags but keep content
    # Remove <break> tags
    content = re.sub(r'<break[^>]*?/>', '', content)

    # Remove <emphasis> tags but keep content
    content = re.sub(r'<emphasis[^>]*?>(.*?)</emphasis>', r'\1', content)

    # Remove <prosody> tags but keep content
    content = re.sub(r'<prosody[^>]*?>(.*?)</prosody>', r'\1', content)

    # Parse speaker sections
    lines = content.split('\n')
    current_speaker = None
    speaker_texts = {'sarah': [], 'david': []}

    for line in lines:
        line = line.strip()

        # Check for speaker marker
        if line.startswith('## '):
            current_speaker = line[3:].strip().lower()
            continue

        # Add text to current speaker
        if line and current_speaker:
            speaker_texts[current_speaker].append(line)

    # Write output
    with open(output_file, 'w') as f:
        # Interleave speakers (assuming they alternate)
        max_len = max(len(speaker_texts['sarah']), len(speaker_texts['david']))

        for i in range(max_len):
            if i < len(speaker_texts['sarah']):
                f.write(speaker_texts['sarah'][i].strip() + '\n\n')
            if i < len(speaker_texts['david']):
                f.write(speaker_texts['david'][i].strip() + '\n\n')

    print(f"✓ Converted {ssml_file} -> {output_file}")
    print(f"  Sarah: {len(speaker_texts['sarah'])} segments")
    print(f"  David: {len(speaker_texts['david'])} segments")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SSML to plain text')
    parser.add_argument('input', help='Input SSML file')
    parser.add_argument('output', help='Output text file')

    args = parser.parse_args()
    convert_ssml_to_text(args.input, args.output)
