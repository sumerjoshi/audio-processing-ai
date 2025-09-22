# Audio Processing AI Project

[![CI](https://github.com/yourusername/audio-processing-ai/workflows/CI/badge.svg)](https://github.com/yourusername/audio-processing-ai/actions)
[![PyPI version](https://badge.fury.io/py/audio-processing-ai.svg)](https://badge.fury.io/py/audio-processing-ai)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project uses deep learning to analyze audio files and detect AI-generated content. The goal of this project is to listen to an .mp3 or a .wav file and determine if it's AI generated or not.

## Installation

### Using uv (Recommended)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

2. Create and activate a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package in development mode:
```bash
uv pip install -e .
```

### Using pip (Alternative)

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install the package in development mode:
```bash
pip install -e .
```

This will install the `audio-processing-ai` package and all its dependencies.

## Development

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Quick Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-processing-ai.git
cd audio-processing-ai

# Switch to main branch (if not already there)
git checkout main-copy

# Create virtual environment and install dev dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

## Usage

### Training

For the training step, I used this file from here [Link text][https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/README.md] that is a 16khz model for inference to finetune against.

For the example here, I set up a data folder at the top level with /data/train/ai and /data/train/real
and would .mp3 and .wav files that I want to fintune against. I got the real data from
FMA [Link Text][https://github.com/mdeff/fma] for testing, and the AI generated data from
Facebook's Music Gen. There needs to be the word "ai" in the path of the ai folders and "real" in the 
path to the real songs.

**NOTE: In /model/pretrained/cnn14.py, I'm hardcoding the path to be /model/pretrained/pretrained_models/Cnn14_16k_mAP=0.438.pth.gz. This would have to be changed in the future. Cnn14 only takes in gzip files
so gzip your file beforehand**

Steps:
1. First place files in audio-processing-ai/data/train (if you are going to finetune data against your model) 
    **All AI Files should go in the /data/train/ai and all of the real files goes in /data/train/real. This is because we need to do supervised learning befor training the classfier which file is AI music and which is Real**
2. Figure out the model you are going to finetune against
3. Update this line (PRETRAINED_MODEL_PATH = 'model/pretrained/pretrained_models/Cnn14_16k_mAP=0.438.pth.gz') at cnn14.py to the .pth.gz file location of your choice

To train the model:
```bash
python train.py \
    --num-epochs 5 \
    --dataFolder data/train/ \
    --savedPath model/saved_models/your_model.pth \
    [--resume-from path/to/checkpoint.pth]  # Optional: resume from a checkpoint
```

Required arguments:
- `--savedPath`: Path where the model will be saved (must end in .pth)
- `--dataFolder`: Directory containing training data (default: "data/train/")
- `--num-epochs`: Number of training epochs (default: 5)

Optional arguments:
- `--resume-from`: Path to a checkpoint to resume training from

### Inference

To run predictions on audio files:
```bash
python predict.py \
    --folder path/to/audio/files \
    --model model/saved_models/your_model.pth
```

Required arguments:
- `--folder`: Directory containing .mp3/.wav files to analyze
- `--model`: Path to your trained model (.pth file)

The script will:
1. Process each audio file in the specified folder
2. Generate predictions for AI-generated content and audio scene tags
3. Save results to a CSV file named `predictions_YYYYMMDD_HHMM.csv`

## Project Structure

```
audio-processing-ai/
├── .github/
│   └── workflows/                    # GitHub Actions CI/CD workflows
├── src/
│   └── audio_processing_ai/          # Main package
│       ├── dataset/                  # Dataset loading and processing utilities
│       ├── model/                    # Model architecture and pretrained weights
│       ├── inference/                # Inference scripts and label files
│       └── scripts/                  # Utility scripts (including threshold_sweep.py)
├── tests/                            # Test files
├── train.py                          # Training script
├── predict.py                        # Prediction script
├── pyproject.toml                    # Package configuration
├── uv.lock                           # uv lock file (if using uv)
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
├── .gitignore                        # Git ignore rules
├── CONTRIBUTING.md                   # Contributing guidelines
├── CHANGELOG.md                      # Changelog
└── README.md                         # This file
```

## Notes

- The project uses PyTorch for deep learning
- Audio processing is done using torchaudio and librosa
- Model architecture is based on CNN14 with dual-head classification
- Training data should be organized in the `data/train/` directory
- Model checkpoints are saved in `model/saved_models/`
- The project is structured as a proper Python package following modern packaging standards
- All modules are organized under `src/audio_processing_ai/` for better code organization
- Uses `uv` for fast dependency management (recommended) or `pip` as an alternative
- Python 3.9+ is required for compatibility with all dependencies
- Includes comprehensive CI/CD with GitHub Actions for testing, linting, and deployment
- Pre-commit hooks ensure code quality and consistency
- Automated dependency updates and PyPI publishing workflows
