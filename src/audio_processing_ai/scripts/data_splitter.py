import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List


def split_data_properly(source_folder: str, output_folder: str = None,
                       train_ratio: float = 0.7, val_ratio: float = 0.15, 
                       test_ratio: float = 0.15, seed: int = 42) -> str:
    """
    Split data into train/val/test sets BEFORE any training
    
    Args:
        source_folder: Path to folder containing 'real' and 'ai' subfolders
        output_folder: Where to create the split data (default: source_folder/data_split)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created split data folder
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    source_path = Path(source_folder)
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder {source_folder} does not exist")
    
    # Set output folder
    if output_folder is None:
        output_folder = source_path.parent / "data_split"
    else:
        output_folder = Path(output_folder)
    
    print(f"Splitting data from: {source_path}")
    print(f"Output will be saved to: {output_folder}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Find all audio files for each class
    real_files = []
    ai_files = []
    
    # Look for real files
    real_patterns = ["real/**/*.wav", "real/**/*.mp3", "real/**/*.flac", "real/**/*.m4a"]
    for pattern in real_patterns:
        real_files.extend(source_path.glob(pattern))
    
    # Look for AI files  
    ai_patterns = ["ai/**/*.wav", "ai/**/*.mp3", "ai/**/*.flac", "ai/**/*.m4a"]
    for pattern in ai_patterns:
        ai_files.extend(source_path.glob(pattern))
    
    print(f"Found {len(real_files)} real audio files")
    print(f"Found {len(ai_files)} AI audio files")
    
    if len(real_files) == 0:
        raise ValueError("No real audio files found. Check your folder structure.")
    if len(ai_files) == 0:
        raise ValueError("No AI audio files found. Check your folder structure.")
    
    # Shuffle files to ensure random distribution
    random.shuffle(real_files)
    random.shuffle(ai_files)
    
    def split_file_list(files: List[Path], train_r: float, val_r: float, test_r: float) -> Dict[str, List[Path]]:
        """Split a list of files into train/val/test"""
        n = len(files)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        
        return {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }
    
    # Split files for each class
    real_splits = split_file_list(real_files, train_ratio, val_ratio, test_ratio)
    ai_splits = split_file_list(ai_files, train_ratio, val_ratio, test_ratio)
    
    # Create directory structure
    print(f"\nCreating directory structure...")
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'ai']:
            split_dir = output_folder / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {split_dir}")
    
    def copy_files_to_split(file_dict: Dict[str, List[Path]], class_name: str):
        """Copy files to their respective split directories"""
        copied_counts = {}
        for split, files in file_dict.items():
            dest_dir = output_folder / split / class_name
            copied_count = 0
            
            for file_path in files:
                try:
                    dest_path = dest_dir / file_path.name
                    # Handle name conflicts
                    counter = 1
                    while dest_path.exists():
                        stem = file_path.stem
                        suffix = file_path.suffix
                        dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Warning: Failed to copy {file_path}: {e}")
            
            copied_counts[split] = copied_count
        return copied_counts
    
    # Copy files to new structure
    print(f"\nCopying files...")
    real_counts = copy_files_to_split(real_splits, 'real')
    ai_counts = copy_files_to_split(ai_splits, 'ai')
    
    # Print summary
    print(f"\nData splitting complete!")
    print("="*50)
    print("SPLIT SUMMARY:")
    print("="*50)
    
    total_real = sum(real_counts.values())
    total_ai = sum(ai_counts.values())
    
    for split in ['train', 'val', 'test']:
        real_count = real_counts[split]
        ai_count = ai_counts[split]
        total_split = real_count + ai_count
        real_pct = (real_count / total_real * 100) if total_real > 0 else 0
        ai_pct = (ai_count / total_ai * 100) if total_ai > 0 else 0
        
        print(f"{split.upper():>5}: {real_count:>4} real ({real_pct:>5.1f}%), "
              f"{ai_count:>4} AI ({ai_pct:>5.1f}%), "
              f"{total_split:>4} total")
    
    print("-" * 50)
    print(f"TOTAL: {total_real:>4} real, {total_ai:>4} AI, {total_real + total_ai:>4} total")
    
    # Create a summary file
    summary_file = output_folder / "split_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DATA SPLIT SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source folder: {source_path}\n")
        f.write(f"Split ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Split date: {os.popen('date').read().strip()}\n\n")
        
        for split in ['train', 'val', 'test']:
            f.write(f"{split.upper()}: {real_counts[split]} real, {ai_counts[split]} AI\n")
        
        f.write(f"\nTOTAL: {total_real} real, {total_ai} AI files\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    
    return str(output_folder)


def validate_split(split_folder: str):
    """Validate that the data split was successful"""
    split_path = Path(split_folder)
    
    if not split_path.exists():
        print(f"Split folder {split_folder} does not exist")
        return False
    
    print(f"\nValidating split at: {split_path}")
    
    all_good = True
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'ai']:
            split_dir = split_path / split / class_name
            if not split_dir.exists():
                print(f"Missing directory: {split_dir}")
                all_good = False
                continue
            
            files = list(split_dir.glob("*"))
            audio_files = [f for f in files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']]
            
            if len(audio_files) == 0:
                print(f"No audio files in: {split_dir}")
            else:
                print(f"{split}/{class_name}: {len(audio_files)} files")
    
    if all_good:
        print("Split validation passed!")
    else:
        print("Split validation failed!")
    
    return all_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split audio dataset into train/validation/test sets"
    )
    parser.add_argument(
        "--source", 
        required=True,
        help="Source folder containing 'real' and 'ai' subfolders"
    )
    parser.add_argument(
        "--output",
        help="Output folder for split data (default: source/../data_split)"
    )
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=0.7,
        help="Proportion for training set (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio", 
        type=float, 
        default=0.15,
        help="Proportion for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio", 
        type=float, 
        default=0.15,
        help="Proportion for test set (default: 0.15)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate existing split (don't create new split)"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        # Just validate existing split
        split_folder = args.output or Path(args.source).parent / "data_split"
        validate_split(split_folder)
    else:
        # Create new split
        try:
            split_folder = split_data_properly(
                source_folder=args.source,
                output_folder=args.output,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
            
            # Validate the created split
            validate_split(split_folder)
            
        except Exception as e:
            print(f"Error: {e}")
            exit(1)