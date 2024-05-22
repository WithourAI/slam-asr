# SLAM-ASR: Speech-Language Model for Automatic Speech Recognition

SLAM-ASR is an unofficial Python Lightning implementation that combines Whisper speech encoder and large language models (LLMs) for automatic speech recognition (ASR). It leverages the power of pre-trained models to transcribe speech into text.

## Features

- Utilizes Whisper speech encoder for speech feature extraction
- Supports various LLMs for text generation (e.g., Meta-Llama, Vicuna)
- Trains the model using PyTorch Lightning for scalability and ease of use
- Provides inference script for transcribing audio files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WithourAI/slam-asr.git
cd slam-asr
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the SLAM-ASR model, change the parameters and run the following command:
```bash
bash train.sh
```

This script will start the training process using the specified configuration, including the trainer settings, model hyperparameters, and callbacks.

From testing, the model can be trained on a single GPU with 24GB of memory with batch_size 1. With 40GB of memory, the batch_size can be increased to 6.

## Inference

To transcribe an audio file using the trained SLAM-ASR model, use the `inference.py` script:

## Model Architecture

The SLAM-ASR model consists of the following components:

- Whisper speech encoder: Extracts features from the input speech
- Projector: Transforms speech embeddings to the LLM embedding space
- LLM (e.g., Meta-Llama, Vicuna): Generates text based on the projected speech embeddings

## Dataset

The model can be trained on various speech datasets, such as LibriSpeech or custom datasets. Modify the `setup` method in `slam_asr.py` to load your desired dataset.

## Configuration

The training configuration can be customized by modifying the arguments in the `train.sh` script. Adjust the trainer settings, model hyperparameters, and callbacks according to your requirements.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Meta-Llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [PyTorch Lightning](https://lightning.ai/)
- The SLAM-ASR implementation is inspired by the paper ["An Embarrassingly Simple Approach for LLM with Strong ASR Capacity"](https://arxiv.org/abs/2402.08846) by Ziyang Ma et al.