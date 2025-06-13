# Audio Processing AI

This project uses deep learning to analyze audio files and detect AI-generated content.

## Setup

You can set up this project using either `uv` (recommended) or `pip`.

### Option 1: Using uv (Recommended)

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Option 2: Using pip

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python train/train.py --epoch 5 --dataFolder data/train/ --savePath model/saved_models/your_model.pth
```

### Inference

To run predictions on audio files:
```bash
python inference/predict.py --folder path/to/audio/files --model path/to/model.pth --out predictions.csv
```

## Project Structure

- `train/`: Training scripts and utilities
- `inference/`: Inference scripts for prediction
- `model/`: Model architecture and pretrained weights
- `data/`: Training and test data
- `requirements.txt`: Project dependencies
- `.cursor.json`: Cursor IDE configuration (optional)

## Notes

- The project uses PyTorch for deep learning
- Audio processing is done using torchaudio and librosa
- Model architecture is based on CNN14 with dual-head classification
- Training data should be organized in the `data/train/` directory
- Model checkpoints are saved in `model/saved_models/`
