#!/usr/bin/env python3

import zipfile
import os
from pathlib import Path
import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M")


def extract_first_audio_files_from_zip(zip_path, output_dir, max_files=500) -> None:
    """
    Extract the first max_files (MP3 and WAV format)
    here from Zip Archive

    Args:
        zip_file: Path to the zip file
        output_dir: Directory to extract files to
        max_files: Maximum number of audio files to extract
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    audio_extensions = {".mp3", ".wav"}
    extracted_count = 0

    print(f"Opening zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get all file names in the zip
        all_files = zip_ref.namelist()

        audio_files = []

        for f in all_files:
            if Path(f).suffix.lower() in audio_extensions:
                audio_files.append(f)

        print(f"Found {len(audio_files)} audio files in the archive")
        print(f"Extracting first {min(max_files, len(audio_files))} files...")

        # Extracting first max_files audio files
        for file_path in audio_files[:max_files]:
            try:
                zip_ref.extract(file_path, output_path)
                extracted_count += 1

                # Move file to root of output_dir
                source_path = output_path / file_path
                file_name = Path(file_path).name
                dest_path = output_path / file_name

                # Handle duplicate names by adding a counter
                counter = 1
                original_dest = dest_path
                while dest_path.exists():
                    stem = original_dest.stem
                    suffix = original_dest.suffix
                    dest_path = output_path / f"{stem}_{counter}{suffix}"

                shutil.move(str(source_path), str(dest_path))

                # Clean up directories
                try:
                    parent_dir = source_path.parent
                    while parent_dir != output_path and parent_dir.exists():
                        parent_dir.rmdir()
                        parent_dir = parent_dir.parent
                except OSError:
                    pass

                if extracted_count % 50 == 0:
                    print(f"Extracted {extracted_count} files...")

            except Exception as e:
                print(f"Error extracting {file_path}: {e}")
                continue

    print(f"\nExtraction Complete!")
    print(f"Extracted {extracted_count} audio files to '{output_dir}' directory")

    mp3_count = len(list(output_path.glob("*.mp3")))
    wav_count = len(list(output_path.glob("*.wav")))
    print(f"MP3 files: {mp3_count}")
    print(f"WAV files: {wav_count}")


if __name__ == "__main__":
    zip_file = "fma_small.zip"
    folder_name = f"audio_files_{timestamp}"

    if not os.path.exists(zip_file):
        print(f"Error: {zip_file} not found!")
        exit(1)

    extract_first_audio_files_from_zip(zip_file, folder_name)
