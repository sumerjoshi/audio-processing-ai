# Audio Processing AI Project

This project uses deep learning to analyze audio files and detect AI-generated content. The goal of this project is to listen to an .mp3 or a .wav file and determine if it's AI generated or not

## Setup

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
