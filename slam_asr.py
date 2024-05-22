import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics.text import CharErrorRate, WordErrorRate
import librosa
import torchaudio.transforms as T
from torch.optim.lr_scheduler import LambdaLR
import lightning as l
import unicodedata
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperModel,
    WhisperFeatureExtractor,
)
from datasets import load_dataset


class Projector(nn.Module):
    def __init__(self, speech_encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(speech_encoder_hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, llm_hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


def downsample(features: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, seq_len, hidden_size = features.shape
    # Check if seq_len is divisible by k to avoid losing data in reshaping
    if seq_len % k != 0:
        raise ValueError(
            "Sequence length must be divisible by the downsample factor")
    # Reshape to group every k elements
    features = features.view(batch_size, seq_len // k, k, hidden_size)
    # Reshape to merge the grouped elements into a single feature vector
    downsampled_features = features.reshape(
        batch_size, seq_len // k, k * hidden_size)
    return downsampled_features


def tokenize_text(text, tokenizer):
    return tokenizer(
        text, return_tensors="pt", padding=False, truncation=True, max_length=1024, add_special_tokens=False
    )


def get_text_embedding(text, model, tokenizer, return_attention_mask=False):
    tokens = tokenize_text(text, tokenizer)
    token_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(token_ids)

    if return_attention_mask:
        return embeddings.to(dtype=model.dtype), attention_mask
    else:
        return embeddings.to(dtype=model.dtype)


def get_token_embedding(token, model):
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(token)

    return embeddings.to(dtype=model.dtype)


def clean_text(text):
    chars_to_strip = '「」『』（）〔〕［］｛｝｟｠〈〉《》【】〖〗〘〙〚〛'
    text = unicodedata.normalize('NFKC', text)
    text = text.strip(chars_to_strip)
    text = text.replace("」。", "").replace(
        "』。", "").replace("♥", "").replace("�", "")
    # text = text.lower()
    return text


def data_collator(batch, llm_tokenizer, feature_extractor):
    speeches = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]

    texts_cleaned = [clean_text(text) for text in texts]

    speech_features = []
    speech_masks = []

    tokenized_texts = [tokenize_text(text, llm_tokenizer)
                       for text in texts_cleaned]

    labels_ids = []
    labels_masks = []

    max_text_length = max(
        len(tokenized_text['input_ids'][0]) for tokenized_text in tokenized_texts)

    for tokenized_text in tokenized_texts:
        length = len(tokenized_text['input_ids'][0])
        padded_input_ids = torch.cat([
            tokenized_text['input_ids'][0],
            torch.tensor([llm_tokenizer.eos_token_id]),  # Add EOS token
            torch.full((max_text_length - length,), llm_tokenizer.pad_token_id)
        ])
        padded_attention_mask = torch.cat([
            tokenized_text['attention_mask'][0],
            torch.tensor([1]),  # Add attention mask for EOS token
            torch.zeros((max_text_length - length,))
        ])

        labels_ids.append(padded_input_ids)
        labels_masks.append(padded_attention_mask)

    labels_ids = torch.stack(labels_ids)
    labels_masks = torch.stack(labels_masks)

    for speech in speeches:
        # Convert to mono if the audio has multiple channels
        if speech["array"].ndim > 1:
            speech_array = librosa.to_mono(speech["array"].T)
        else:
            speech_array = speech["array"]

        # Resample the audio to 16000 Hz using librosa
        speech_array = librosa.resample(
            speech_array, orig_sr=speech["sampling_rate"], target_sr=16000)
        speech["sampling_rate"] = 16000

        speech_feature = feature_extractor(
            speech_array, sampling_rate=speech["sampling_rate"], return_tensors="pt", return_attention_mask=True)
        spectrogram = speech_feature.input_features.squeeze(0)

        # Apply SpecAugment
        specaug = T.SpecAugment(
            freq_mask_param=15, time_mask_param=50, n_freq_masks=2, n_time_masks=10, p=0.5)
        augmented_spectrogram = specaug(spectrogram)

        speech_features.append(augmented_spectrogram)
        speech_masks.append(speech_feature.attention_mask.squeeze(0))

    speech_features = torch.stack(speech_features)
    speech_masks = torch.stack(speech_masks)

    return {"speeches": speech_features, "speeches_masks": speech_masks, "labels": labels_ids, "labels_masks": labels_masks}


class SLAM_ASR(l.LightningModule):
    def __init__(self, batch_size: int = 1, warmup_steps: int = 1000, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        speech_encoder_path = "openai/whisper-large-v3"
        self.speech_encoder = WhisperModel.from_pretrained(
            speech_encoder_path).encoder
        
        # llm_path = "Fugaku-LLM/Fugaku-LLM-13B-instruct"
        # llm_path = "lmsys/vicuna-7b-v1.5"
        # Only official llama3 instruct works for Japanese from testing so far
        llm_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        # llm_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # llm_path = "alfredplpl/Llama-3-8B-Instruct-Ja"
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            speech_encoder_path)

        self.downsample_factor = 5
        self.projector = Projector(
            self.speech_encoder.config.hidden_size * self.downsample_factor,
            self.llm_model.config.hidden_size,
        )

        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.user_prompt_embeds, self.user_mask = get_text_embedding(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", self.llm_model, self.llm_tokenizer, return_attention_mask=True)
        # self.user_prompt_embeds, self.user_mask = get_text_embedding(
        #     "### 指示:\n", self.llm_model, self.llm_tokenizer, return_attention_mask=True)

        self.assistant_prompt_embeds, self.assistant_mask = get_text_embedding(
            ". Transcribe speech to text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", self.llm_model, self.llm_tokenizer, return_attention_mask=True)
        # self.assistant_prompt_embeds, self.assistant_mask = get_text_embedding(
        #     "。 音声をテキストに書き起こす。<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", self.llm_model, self.llm_tokenizer, return_attention_mask=True)
        # self.assistant_prompt_embeds, self.assistant_mask = get_text_embedding(
        #     "。 音声をテキストに書き起こす。\n\n### 応答:\n", self.llm_model, self.llm_tokenizer, return_attention_mask=True)

    def setup(self, stage=None):
        # Load datasets
        combined_dataset = load_dataset(
            "withourai/withourai-asr-mix", token=True, split='train')

        # Split into training and validation sets
        train_val_split = combined_dataset.train_test_split(0.001)
        self.train_dataset = train_val_split["train"]
        self.val_dataset = train_val_split["test"]

        # self.train_dataset = load_dataset(
        #     "Raziullah/librispeech_small_asr_fine-tune", split="train")
        # self.val_dataset = load_dataset(
        #     "Raziullah/librispeech_small_asr_fine-tune", split="test")

        # self.train_dataset = load_dataset(
        #     "librispeech_asr", "clean", split="train.100")
        # self.val_dataset = load_dataset(
        #     "librispeech_asr", "clean", split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda batch: data_collator(
                batch, self.llm_tokenizer, self.feature_extractor),
            num_workers=20,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda batch: data_collator(
                batch, self.llm_tokenizer, self.feature_extractor),
            num_workers=20,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            drop_last=True,
        )

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint.get('state_dict', {})

        # Exclude keys that start with 'speech_encoder' or 'llm_model'
        filtered_state_dict = {k: v for k, v in state_dict.items(
        ) if not k.startswith(('speech_encoder', 'llm_model'))}

        # Update the checkpoint with the filtered state_dict
        checkpoint['state_dict'] = filtered_state_dict

    def on_load_checkpoint(self, checkpoint):
        pretrained_dict = checkpoint.get('state_dict', {})
        model_dict = self.state_dict()

        pretrained_dict = {
            k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}

        model_dict.update(pretrained_dict)

        # Load the filtered state_dict into the model
        self.load_state_dict(model_dict)

    def training_step(self, batch, batch_idx):
        batch_size = self.hparams.batch_size
        with torch.no_grad():
            user_prompt_embeds = self.user_prompt_embeds.expand(
                batch_size, -1, -1).to(device=self.llm_model.device, dtype=self.llm_model.dtype)
            user_mask = self.user_mask.expand(
                batch_size, -1).to(self.llm_model.device)

            assistant_prompt_embeds = self.assistant_prompt_embeds.expand(
                batch_size, -1, -1).to(device=self.llm_model.device, dtype=self.llm_model.dtype)
            assistant_mask = self.assistant_mask.expand(
                batch_size, -1).to(self.llm_model.device)

            labels_ids = batch["labels"].to(self.llm_model.device)
            labels_embed = get_token_embedding(labels_ids, self.llm_model)
            labels_mask = batch["labels_masks"].to(self.llm_model.device)

            speech_embeds = self.speech_encoder(batch["speeches"])
            speech_embeds = speech_embeds.last_hidden_state
            downsampled_embeds = downsample(
                speech_embeds, k=self.downsample_factor)

        projected_embeds = self.projector(downsampled_embeds)

        # Don't think the mask works for projected samples
        speech_mask = torch.ones_like(downsampled_embeds[:, :, 0])

        inputs_embeds = torch.cat([
            user_prompt_embeds,
            projected_embeds,
            assistant_prompt_embeds,
            labels_embed
        ], dim=1)

        attention_masks = torch.cat([
            user_mask,
            speech_mask,
            assistant_mask,
            labels_mask
        ], dim=1)

        labels = torch.full(
            (batch_size, inputs_embeds.shape[1]), fill_value=-100, device=labels_ids.device)

        labels[:, -labels_ids.shape[1]:] = torch.where(
            labels_mask.bool(), labels_ids, torch.full_like(labels_ids, -100))
        actual_label_length = (labels_mask == 1).sum(dim=1).max().detach()

        loss = self.llm_model(inputs_embeds=inputs_embeds,
                              attention_mask=attention_masks, labels=labels).loss
        self.log("train_loss", loss)

        if batch_idx % 200 == 0:
            # Select the first example from the batch
            input_embed = inputs_embeds[0].unsqueeze(0)
            attention_mask = attention_masks[0].unsqueeze(0)

            output = self.llm_model.generate(
                inputs_embeds=input_embed,
                attention_mask=attention_mask,
                max_new_tokens=actual_label_length + 10,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                num_beams=2,
                early_stopping=True,
            )

            decoded_pred = self.llm_tokenizer.decode(
                output[0], skip_special_tokens=True)
            decoded_label = self.llm_tokenizer.decode(
                labels_ids[0], skip_special_tokens=True)

            if dist.get_rank() == 0:
                print(f"\nReference: {decoded_label}")
                print(f"Predicted: {decoded_pred}")

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = self.hparams.batch_size
        with torch.no_grad():
            user_prompt_embeds = self.user_prompt_embeds.expand(
                batch_size, -1, -1).to(device=self.llm_model.device, dtype=self.llm_model.dtype)
            user_mask = self.user_mask.expand(
                batch_size, -1).to(self.llm_model.device)

            assistant_prompt_embeds = self.assistant_prompt_embeds.expand(
                batch_size, -1, -1).to(device=self.llm_model.device, dtype=self.llm_model.dtype)
            assistant_mask = self.assistant_mask.expand(
                batch_size, -1).to(self.llm_model.device)

            labels_ids = batch["labels"].to(self.llm_model.device)
            labels_embed = get_token_embedding(labels_ids, self.llm_model)
            labels_mask = batch["labels_masks"].to(self.llm_model.device)

            speech_embeds = self.speech_encoder(batch["speeches"])
            speech_embeds = speech_embeds.last_hidden_state
            downsampled_embeds = downsample(
                speech_embeds, k=self.downsample_factor)

            projected_embeds = self.projector(downsampled_embeds)
            # Don't think the mask works for projected samples
            speech_mask = torch.ones_like(downsampled_embeds[:, :, 0])

            inputs_embeds = torch.cat([
                user_prompt_embeds,
                projected_embeds,
                assistant_prompt_embeds,
                labels_embed
            ], dim=1)

            attention_masks = torch.cat([
                user_mask,
                speech_mask,
                assistant_mask,
                labels_mask
            ], dim=1)

            labels = torch.full(
                (batch_size, inputs_embeds.shape[1]), fill_value=-100, device=labels_ids.device)

            labels[:, -labels_ids.shape[1]:] = torch.where(
                labels_mask.bool(), labels_ids, torch.full_like(labels_ids, -100))
            actual_label_length = (labels_mask == 1).sum(dim=1).max().detach()

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_masks,
                max_new_tokens=actual_label_length + 10,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                num_beams=2,
                early_stopping=True,
            )

            decoded_preds = [self.llm_tokenizer.decode(
                ids, skip_special_tokens=True) for ids in outputs]
            decoded_labels = [self.llm_tokenizer.decode(
                ids, skip_special_tokens=True) for ids in labels_ids]

            cer = CharErrorRate()
            wer = WordErrorRate()
            cer_score = cer(decoded_preds, decoded_labels)
            wer_score = wer(decoded_preds, decoded_labels)

            loss = self.llm_model(
                inputs_embeds=inputs_embeds, labels=labels).loss
            self.log("val_loss", loss)
            self.log("val_cer", cer_score)
            self.log("val_wer", wer_score)

            if dist.get_rank() == 0:
                print(f"\nReference: {decoded_labels[0]}")
                print(f"Predicted: {decoded_preds[0]}")

            return {"val_loss": loss, "val_cer": cer_score, "val_wer": wer_score}

    def configure_optimizers(self):
        # Create the optimizer with the specified learning rate and no weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0)

        # Define a scheduler that warms up the learning rate for the first 1,000 steps
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return float(step) / float(max(1, self.hparams.warmup_steps))
            return 1.0

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',  # Update the learning rate after every step
            'frequency': 1
        }

        return [optimizer], [scheduler]


def cli_main():
    torch.set_float32_matmul_precision("medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    cli = LightningCLI(SLAM_ASR)


if __name__ == "__main__":
    cli_main()
