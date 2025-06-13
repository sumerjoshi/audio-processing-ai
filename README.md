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

3. Install the package in development mode:
```bash
uv pip install -e .
```

### Option 2: Using pip

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

## Usage

### Training

For the training step, I used this file from here [Link text][https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/README.md] that is a 16khz model for inference to finetune against.

For the example here, I set up a data folder at the top level with /data/train/ai and /data/train/real
and would .mp3 and .wav files that I want to fintune against. I got the real data from
FMA [Link Text][https://github.com/mdeff/fma] for testing, and the AI generated data from
Facebook's Music Gen.

**NOTE: In /model/pretrained/cnn14.py, I'm hardcoding the path to be /mode/pretrained/pretrained_models/Cnn14_16k_mAP=0.438.pth.gz. This would have to be changed in the future. Cnn14 only takes in gzip files
so gzip your file beforehand**

Steps:
1. First place files in audio-processing-ai/data/train (if you are going to finetune data against your model)
2. Figure out the model you are going to finetune against
3. Update this line (PRETRAINED_MODEL_PATH = 'model/pretrained/pretrained_models/Cnn14_16k_mAP=0.438.pth.gz') at cnn14.py to the .pth.gz file location of your choice

To train the model:
```bash
cd audio-processing-ai
python train.py --epoch 5 --dataFolder data/train/ --savePath model/saved_models/your_model.pth
```

### Inference

If you have an already trained/finetuned model and you just want to run the prediction,
run it as such.

Folder is the path to the audio files you want to test against.

Example lists the model path as model/saved_models/your_model.pth but that is changeable 
depending on where you saved it.

The outputted file is predictions_timestamp.csv

To run predictions on audio files:
```bash
python predict.py --folder path/to/audio/files --model model/saved_models/your_model.pth
```

## Project Structure

- `inference/`: Inference scripts for prediction
- `model/`: Model architecture and pretrained weights
- `dataset/`: Dataset loading and processing utilities
- `data/`: Training and test data
- `setup.py`: Package installation configuration
- `requirements.txt`: Project dependencies (for reference)
- `.cursor.json`: Cursor IDE configuration (optional)

## Notes

- The project uses PyTorch for deep learning
- Audio processing is done using torchaudio and librosa
- Model architecture is based on CNN14 with dual-head classification
- Training data should be organized in the `data/train/` directory
- Model checkpoints are saved in `model/saved_models/`
- The project is installed as a Python package for proper import handling

## Code Quality

This project uses Ruff for both linting and formatting Python code. Ruff is a fast Python linter and formatter written in Rust.

### Using Ruff

1. Install Ruff (it's already included in the dev dependencies):
```bash
# Using pip (recommended if you want to use your existing virtual environment)
pip install -e ".[dev]"

# OR using uv pip (if you want to use uv but keep your current virtual environment)
uv pip install -e ".[dev]"

# Note: Do NOT use 'uv venv' unless you want to create a new virtual environment
# with pyenv. If you want to use uv while keeping your current environment,
# use 'uv pip' instead.
```

2. Format your code:
```bash
ruff format .
```

3. Lint your code:
```bash
ruff check .
```

4. Fix linting issues automatically:
```bash
ruff check --fix .
```

The Ruff configuration is in `pyproject.toml`. Currently, it:
- Uses a line length of 88 characters (same as Black)
- Targets Python 3.9
- Enables pycodestyle (`E`) and Pyflakes (`F`) rules by default
- Ignores line length violations (`E501`)

You can customize the Ruff configuration by modifying the `[tool.ruff]` section in `pyproject.toml`.
