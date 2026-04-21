# Audio Processing AI Project

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

For the example here, I set up a data folder at the top level with `/data/train/ai` and `/data/train/real`
and would .mp3 and .wav files that I want to finetune against. I got the real data from
FMA [Link Text][https://github.com/mdeff/fma] for testing, and the AI generated data from
Facebook's Music Gen. There needs to be the word "ai" in the path of the ai folders and "real" in the 
path to the real songs.

**NOTE: In /model/pretrained/cnn14.py, I'm hardcoding the path to be /model/pretrained/pretrained_models/Cnn14_16k_mAP=0.438.pth.gz. This would have to be changed in the future. Cnn14 only takes in gzip files
so gzip your file beforehand**

Steps:
1. First place files in `data/train/` (if you are going to finetune data against your model) 
    **All AI Files should go in `/data/train/ai` and all of the real files goes in `/data/train/real`. This is because we need to do supervised learning before training the classifier which file is AI music and which is Real**
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

Place your audio files in the `data/predict/` folder, then run predictions:

```bash
python predict.py \
    --folder data/predict \
    --model model/saved_models/your_model.pth \
    --output results
```

Or specify any folder containing audio files:

```bash
python predict.py \
    --folder path/to/your/audio/files \
    --model model/saved_models/your_model.pth \
    --output results
```

Required arguments:
- `--folder`: Directory containing .mp3/.wav/.flac/.m4a files to analyze
- `--model`: Path to your trained model (.pth file)
- `--output`: Output directory for CSV and Excel results

The script will:
1. Process each audio file in the specified folder
2. Generate predictions for AI-generated content and audio scene tags
3. Save results to a CSV file named `predictions_YYYYMMDD_HHMM.csv` and an Excel file with summary statistics

### Evaluation

To evaluate your trained model on a test set, use the `evaluation_pipeline.py` script. This will generate comprehensive metrics including ROC curves, confusion matrices, and threshold analysis.

**Prerequisites:**
- Your data should be split into train/val/test folders
- Test folder should contain `real/` and `ai/` subfolders with audio files

Example evaluation:
```bash
python evaluation_pipeline.py \
    --model model/saved_models/your_model.pth \
    --data-split data/split/ \
    --output-dir evaluation_results
```

The evaluation will generate:
- `evaluation_plots.png`: ROC curve, logit distributions, threshold analysis, and confusion matrix
- `evaluation_thresholds.csv`: Performance metrics across different thresholds
- `evaluation_report.txt`: Detailed text report with all metrics

**Example Output:**
See `sample_runs/eval_sample_run/` for an example of evaluation results. This directory contains:
- `evaluation_plots.png`: Visual performance metrics
- `evaluation_report.txt`: Detailed evaluation report
- `evaluation_thresholds.csv`: Threshold analysis data

Required arguments:
- `--model`: Path to trained model (.pth file)
- `--data-split`: Path to split data folder (containing train/val/test subdirs)

Optional arguments:
- `--output-dir`: Output directory for results (default: auto-generated with timestamp)
- `--seed`: Random seed for reproducibility (default: 42)

### Gradio Web Interface

The project includes a Gradio web interface for interactive AI audio detection. You can run it locally or deploy it to Modal.

#### Running Locally

To run the Gradio app locally with a PyTorch model:

```bash
python gradio_app.py \
    --model model/saved_models/your_model.pth \
    [--threshold 0.35] \
    [--onnx]
```

To run with an ONNX model:

```bash
python gradio_app.py \
    --model model/saved_models/your_model.onnx \
    --onnx \
    [--threshold 0.35]
```

Required arguments:
- `--model`: Path to your trained model (.pth for PyTorch or .onnx for ONNX)

Optional arguments:
- `--threshold`: Threshold for AI detection (default: 0.35)
- `--onnx`: Use ONNX model instead of PyTorch (required if model is .onnx)

The app will launch at `http://127.0.0.1:7860` by default.

**Features:**
- **Single File Upload**: Upload one audio file (.mp3, .wav, .flac, .m4a) for instant prediction
- **Batch Processing**: Upload a ZIP file containing multiple audio files
- Results include AI detection confidence and music analysis (genre, mood, tempo, energy)
- Batch processing shows interactive tables, charts, and summary statistics
- Download full results as CSV/Excel for further analysis

#### Deploying to Modal

The Gradio app can be deployed to Modal for cloud hosting. The deployed app uses ONNX models for optimized inference.

**Prerequisites:**
1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

3. Convert your PyTorch model to ONNX (if not already done):
```bash
python scripts/setup_modal.py \
    --pth-path model/saved_models/your_model.pth \
    --onnx-path model/saved_models/your_model.onnx
```

4. Upload the ONNX model to Modal volume:
```bash
modal volume put ai-audio-models model/saved_models/your_model.onnx model.onnx
```

**Deploy to Modal:**
```bash
modal deploy gradio_app.py
```

**Access the deployed app:**
Once deployed, your app will be available at:
- **URL**: https://sumerjoshi--ai-audio-detection-gradio-app-modal.modal.run/

**Note:** The Modal deployment uses ONNX models for better performance and compatibility. Make sure you've uploaded your ONNX model to the `ai-audio-models` volume before deploying.

## Project Structure

```
audio-processing-ai/
├── .github/
│   └── workflows/                    # GitHub Actions CI/CD workflows
├── data/
│   ├── train/                       # Training data
│   │   ├── ai/                      # AI-generated audio files
│   │   └── real/                    # Real audio files
│   └── predict/                     # User audio files for prediction
├── model/
│   ├── pretrained/                  # Pretrained model weights
│   │   └── pretrained_models/
│   └── saved_models/                # Trained model checkpoints
├── sample_runs/                     # Example evaluation outputs
│   └── eval_sample_run/             # Sample evaluation results
├── src/
│   └── audio_processing_ai/          # Main package
│       ├── dataset/                  # Dataset loading and processing utilities
│       ├── model/                    # Model architecture and pretrained weights
│       ├── inference/                # Inference scripts and label files
│       └── scripts/                  # Utility scripts (including threshold_sweep.py)
├── tests/                            # Test files
├── train.py                          # Training script
├── predict.py                        # Prediction script
├── evaluation_pipeline.py            # Model evaluation script
├── gradio_app.py                     # Gradio web interface (local and Modal deployment)
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
- Prediction files can be placed in `data/predict/` for convenience
- Model checkpoints are saved in `model/saved_models/`
- Evaluation results are saved with timestamps for easy tracking
- The project is structured as a proper Python package following modern packaging standards
- All modules are organized under `src/audio_processing_ai/` for better code organization
- Uses `uv` for fast dependency management (recommended) or `pip` as an alternative
- Python 3.9+ is required for compatibility with all dependencies
- Includes comprehensive CI/CD with GitHub Actions for testing, linting, and deployment
- Pre-commit hooks ensure code quality and consistency
- Automated dependency updates and PyPI publishing workflows
