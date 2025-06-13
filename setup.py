from setuptools import setup, find_packages

setup(
    name="audio-processing-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.7.1",
        "torchaudio>=2.7.1",
        "torchlibrosa>=0.1.0",
        "librosa>=0.11.0",
        "numpy>=2.0.2",
        "scipy>=1.13.1",
        "matplotlib>=3.9.4",
        "tqdm>=4.67.1",
        "soundfile>=0.13.1",
        "scikit-learn>=1.6.1",
    ],
    python_requires=">=3.8",
)
