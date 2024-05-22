import torch
import librosa
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel, WhisperFeatureExtractor
import torch.nn as nn


class Projector(nn.Module):
    """ A projector module to transform speech embeddings to LLM embedding space. """

    def __init__(self, speech_encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(speech_encoder_hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, llm_hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


class SLAM_ASR(torch.nn.Module):
    """ Speech to text module based on Whisper and LLM models. """

    def __init__(self, pretrained_model_name="openai/whisper-large-v3", llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name)
        self.speech_encoder = WhisperModel.from_pretrained(
            pretrained_model_name).encoder
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

        self.downsample_factor = 5
        self.projector = Projector(
            self.speech_encoder.config.hidden_size * self.downsample_factor,
            self.llm_model.config.hidden_size,
        )

        # Freeze the parameters of the pretrained models
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.init_prompts()

    def init_prompts(self):
        """ Initialize embeddings for static prompts used with the LLM. """
        user_prompt_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant_prompt_text = ". Transcribe speech to text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        self.user_prompt_embeds, self.user_mask = self.get_text_embedding(
            user_prompt_text, return_attention_mask=True)
        self.assistant_prompt_embeds, self.assistant_mask = self.get_text_embedding(
            assistant_prompt_text, return_attention_mask=True)

    def forward(self, input_features):
        """ Forward pass for predicting text from speech. """
        with torch.no_grad():
            encoded_features = model.speech_encoder(input_features)
            downsampled_features = downsample(
                encoded_features.last_hidden_state, k=model.downsample_factor)
            projected_features = model.projector(downsampled_features)

            # Concatenate user prompt, projected features, and assistant prompt
            inputs_embeds = torch.cat([
                self.user_prompt_embeds.to(model.llm_model.device, dtype=torch.bfloat16),
                projected_features,
                self.assistant_prompt_embeds.to(model.llm_model.device, dtype=torch.bfloat16)
            ], dim=1)

            # Generate the transcription
            outputs = model.llm_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=200,  # Adjust as needed
                pad_token_id=model.llm_tokenizer.pad_token_id,
                num_beams=5,
                 early_stopping=True,
            )

        return outputs

    @staticmethod
    def downsample(features, k):
        """ Downsample the feature dimension by a factor of k. """
        batch_size, seq_len, hidden_size = features.shape
        if seq_len % k != 0:
            raise ValueError(
                "Sequence length must be divisible by the downsample factor")
        downsampled_features = features.view(
            batch_size, seq_len // k, k * hidden_size)
        return downsampled_features

    def get_text_embedding(self, text, return_attention_mask=False):
        """ Generate embeddings for given text using the tokenizer and LLM. """
        tokens = self.llm_tokenizer(
            text, return_tensors="pt", padding=False, truncation=True, max_length=1024
        )
        token_ids = tokens.input_ids
        attention_mask = tokens.attention_mask if return_attention_mask else None
        embedding_layer = self.llm_model.get_input_embeddings()
        embeddings = embedding_layer(token_ids)
        return embeddings.to(self.llm_model.device), attention_mask.to(self.llm_model.device) if attention_mask is not None else None


@staticmethod
def downsample(features, k):
    """ Downsample the feature dimension by a factor of k. """
    batch_size, seq_len, hidden_size = features.shape
    if seq_len % k != 0:
        raise ValueError(
            "Sequence length must be divisible by the downsample factor")
    downsampled_features = features.view(
        batch_size, seq_len // k, k * hidden_size)
    return downsampled_features


def load_model(checkpoint_path):
    """ Load the model from a checkpoint. """
    model = SLAM_ASR()
    pretrained_dict = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model


def load_audio(audio_path, target_sr=16000):
    """ Load and resample an entire audio file. """
    audio, sr = librosa.load(audio_path, sr=None)  # Load the full audio
    audio = librosa.to_mono(audio)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def segment_audio(audio, segment_length=30, sr=16000):
    """ Segment audio into chunks of a specified length in seconds. """
    segment_samples = segment_length * sr
    return [audio[i:i+segment_samples] for i in range(0, len(audio), segment_samples)]

def transcribe_audio(model, audio_path, device='cuda'):
    """ Transcribe segmented audio using the trained model. """
    audio, sr = load_audio(audio_path)
    segments = segment_audio(audio, segment_length=20, sr=sr)
    full_transcription = []

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    for segment in segments:
        audio_features = feature_extractor(
            segment, return_tensors="pt", sampling_rate=sr, return_attention_mask=True)
        
        # Move to the correct device
        input_features = audio_features.input_features.to(device, dtype=torch.bfloat16)
        # attention_mask = audio_features.attention_mask.to(device)
        outputs = model.forward(input_features)

        transcription = model.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_transcription.append(transcription)

    return ' '.join(full_transcription)

# Main execution block
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '/home/x/x/epoch=1-step=30432.ckpt'
    audio_path = '/home/x/x/x.wav'

    # Load model and tokenizer
    model = load_model(checkpoint_path)
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    transcription = transcribe_audio(model, audio_path, device)
    print("Transcription:", transcription)
